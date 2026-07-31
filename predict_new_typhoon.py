"""使用 algorithm_lab 最优深度学习方案预测并检索新台风。

用法：
    python predict_new_typhoon.py bavii.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import folium
import numpy as np

import algorithm_lab as lab
from query_import import parse_typhoon_from_csv


COLORS = [
    "blue", "green", "purple", "orange", "darkred",
    "cadetblue", "darkgreen", "pink", "darkblue", "lightred",
]


def _resample_to_three_hours(points: list[lab.TrackPoint]) -> list[lab.TrackPoint]:
    """将1h/3h混合观测统一到模型训练使用的3小时网格。"""
    if len(points) < 2:
        return points
    times = [
        datetime.fromisoformat(point.time.replace("Z", "+00:00"))
        for point in points
    ]
    timestamps = np.asarray([item.timestamp() for item in times], dtype=float)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    sorted_points = [points[index] for index in order]
    unique_timestamps, unique_indices = np.unique(
        timestamps, return_index=True
    )
    sorted_points = [sorted_points[index] for index in unique_indices]
    timestamps = unique_timestamps
    start, end = timestamps[0], timestamps[-1]
    grid = np.arange(start, end + 1.0, 3 * 3600.0)
    if end - grid[-1] > 60:
        grid = np.r_[grid, end]

    def interpolate(field):
        values = np.asarray([
            float(getattr(point, field)) for point in sorted_points
        ])
        return np.interp(grid, timestamps, values)

    latitudes = interpolate("lat")
    longitudes = interpolate("lon")
    pressures = interpolate("pressure")
    winds = interpolate("wind_speed")
    return [
        lab.TrackPoint(
            datetime.fromtimestamp(value, timezone.utc).isoformat(),
            float(lat), float(lon), float(pressure), float(wind),
        )
        for value, lat, lon, pressure, wind in zip(
            grid, latitudes, longitudes, pressures, winds
        )
    ]


def run(
    csv_path="bavii.csv",
    output_dir="output",
    predict_hours=72,
    prediction_mode="score61_restored",
):
    source = parse_typhoon_from_csv(csv_path)
    raw_known = [
        lab.TrackPoint(
            p.time.isoformat(), p.lat, p.lon, p.pressure, p.wind_speed
        )
        for p in source.points
    ]
    known = _resample_to_three_hours(raw_known)
    records = lab.clean_records(lab.SQLiteReader().load())
    strict_mode = prediction_mode == "strict_validation"
    algorithm = lab.DirectedOptimizedAlgorithm(
        records,
        fusion_weights=(1.0, 0.0, 0.0) if strict_mode else (0.65, 0.20, 0.15),
        forecast_time_weight_mode="score61",
    )
    stem = Path(csv_path).stem
    display_name = f"{stem.upper()} 新台风"
    query_started = time.perf_counter()
    if prediction_mode == "analog_fusion":
        if predict_hours != 72:
            raise ValueError("analog_fusion 当前只支持72小时预测")
        import analog_fusion_optimize as fusion

        config_path = Path("output/best_analog_fusion_config.json")
        if not config_path.exists():
            raise FileNotFoundError(
                "缺少融合配置，请先运行 python analog_fusion_optimize.py"
            )
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if not saved.get("accepted"):
            raise RuntimeError("融合方案未通过独立留出集，不允许接入")
        config = saved["config"]
        fusion_case = fusion.FusionCase(
            stem, display_name, int(known[-1].time[:4]), known,
            np.empty((0, 2), dtype=float),
        )
        base_values = fusion.base_predictions(fusion_case, algorithm)
        analog_pool = fusion.build_analog_pool(fusion_case, records)
        prediction, ranked = fusion.fuse_prediction(
            base_values[config["base_model"]], analog_pool, config
        )
        forecast_steps = len(prediction)
        result = {
            "typhoon_id": stem,
            "name": display_name,
            "known_points": len(known),
            "forecast_steps": forecast_steps,
            "predict_hours": predict_hours,
            "prediction": prediction.tolist(),
            "raw_prediction": base_values[config["base_model"]].tolist(),
            "forecast_wind": lab._forecast_scalar(
                known, "wind_speed", forecast_steps
            ).tolist(),
            "forecast_pressure": lab._forecast_scalar(
                known, "pressure", forecast_steps
            ).tolist(),
            "confidence": {"0_72h": 0.82},
            "long_range_correction": {"enabled": False},
            "top10": [
                {
                    **{
                        key: value for key, value in item.items()
                        if key != "predicted_continuation"
                    },
                    "comparison_start": item["match_start"],
                    "predicted_continuation": item[
                        "predicted_continuation"
                    ].tolist(),
                }
                for item in ranked[:10]
            ],
            "parameters": {
                "prediction_mode": "prediction_analog_feedback_fusion",
                "fusion_config": config,
                "truth_used_for_ranking": False,
                "target_typhoon_excluded_from_candidates": True,
            },
        }
    else:
        result = algorithm.predict_new_typhoon(
            known,
            predict_hours=predict_hours,
            typhoon_id=stem,
            name=display_name,
            prediction_mode=prediction_mode,
        )
    result["query_seconds"] = time.perf_counter() - query_started
    result["input_points_original"] = len(raw_known)
    result["input_points_resampled_3h"] = len(known)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"{predict_hours}h"
    forecast_csv = output / f"{stem}_predicted_path_{suffix}.csv"
    result_json = output / f"{stem}_prediction_result_{suffix}.json"
    map_html = output / f"{stem}_prediction_map_{suffix}.html"

    last_time = datetime.fromisoformat(known[-1].time.replace("Z", "+00:00"))
    forecast_rows = []
    for index, ((lon, lat), wind, pressure) in enumerate(zip(
        result["prediction"],
        result["forecast_wind"],
        result["forecast_pressure"],
    ), start=1):
        forecast_rows.append({
            "time": (last_time + timedelta(hours=6 * index)).isoformat(),
            "lat": lat,
            "lon": lon,
            "pressure": pressure,
            "wind_speed": wind,
        })
    with forecast_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["time", "lat", "lon", "pressure", "wind_speed"],
        )
        writer.writeheader()
        writer.writerows(forecast_rows)
    result_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    center = source.points[len(source.points) // 2]
    map_object = folium.Map(
        [center.lat, center.lon], zoom_start=5, tiles="OpenStreetMap"
    )
    observed = [(p.lat, p.lon) for p in source.points]
    predicted = [(row["lat"], row["lon"]) for row in forecast_rows]
    folium.PolyLine(
        observed, color="black", weight=5,
        tooltip=f"{display_name} 已观测路径"
    ).add_to(map_object)
    segments = [
        (0, min(12, len(predicted)), "red", None, "0-72h预测"),
        (12, min(20, len(predicted)), "darkorange", "8,8", "72-120h趋势"),
        (20, min(28, len(predicted)), "purple", "2,7", "120-168h趋势"),
    ]
    for start, end, color, dash, label in segments:
        if end <= start:
            continue
        coordinates = (
            [observed[-1], *predicted[start:end]]
            if start == 0
            else [predicted[start - 1], *predicted[start:end]]
        )
        folium.PolyLine(
            coordinates, color=color, weight=4,
            dash_array=dash, tooltip=label,
        ).add_to(map_object)
    for point in source.points:
        folium.CircleMarker(
            [point.lat, point.lon], radius=2.5, color="black",
            weight=1, fill=True, fill_opacity=0.8,
            tooltip=f"观测 {point.time.isoformat()}",
        ).add_to(map_object)
    for row in forecast_rows:
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=3, color="red",
            weight=1, fill=True, fill_opacity=0.85,
            tooltip=f"预测 {row['time']}",
        ).add_to(map_object)

    by_id = {record["id"]: record for record in records}
    for index, item in enumerate(result["top10"]):
        record = by_id[item["id"]]
        color = COLORS[index % len(COLORS)]
        start = int(item.get("comparison_start", 0))
        end = min(
            len(record["points"]),
            int(item.get("forecast_anchor", start)) + result["forecast_steps"],
        )
        display_points = record["points"][start:end] or record["points"]
        coordinates = [(p.lat, p.lon) for p in display_points]
        label = f"{index + 1}. {item['name']} ({item['year']}) {item['similarity']:.4f}"
        folium.PolyLine(
            coordinates, color=color, weight=2, tooltip=label
        ).add_to(map_object)
        for point in display_points:
            folium.CircleMarker(
                [point.lat, point.lon], radius=1.2, color=color,
                weight=1, fill=True, fill_opacity=0.65,
            ).add_to(map_object)
        end_point = display_points[-1]
        folium.Marker(
            [end_point.lat, end_point.lon], tooltip=label,
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:{color};white-space:nowrap">'
                     f'{item["name"]}</div>'
            ),
        ).add_to(map_object)

    legend_rows = [
        f'<div><span style="color:black">━</span> {display_name} 已观测路径</div>',
        '<div><span style="color:red">━</span> 0-72h预测</div>',
        '<div><span style="color:darkorange">┄</span> 72-120h趋势</div>',
        '<div><span style="color:purple">┈</span> 120-168h趋势</div>',
    ]
    legend_rows.extend(
        f'<div><span style="color:{COLORS[i % len(COLORS)]}">━</span> '
        f'{i + 1}. {item["name"]} ({item["year"]}) {item["similarity"]:.3f}</div>'
        for i, item in enumerate(result["top10"])
    )
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:10px;border:1px solid #888;font-size:12px;'
        'line-height:1.5;max-height:300px;overflow:auto">'
        + "".join(legend_rows) + "</div>"
    )
    map_object.get_root().html.add_child(folium.Element(legend))
    map_object.save(map_html)

    print(
        f"原始路径点：{len(raw_known)}；"
        f"3小时重采样后：{len(known)}"
    )
    print(
        f"预测节点：{len(forecast_rows)}"
        f"（未来{predict_hours}小时，每6小时一个节点）"
    )
    for index, item in enumerate(result["top10"], start=1):
        print(
            f"{index:2d}. {item['name']} ({item['year']}) "
            f"相似度={item['similarity']:.4f}"
        )
    print(f"预测路径：{forecast_csv.resolve()}")
    print(f"预测地图：{map_html.resolve()}")
    print(f"查询结果：{result_json.resolve()}")
    print(f"算法净查询耗时：{result['query_seconds']:.3f} 秒")


if __name__ == "__main__":
    run(
        sys.argv[1] if len(sys.argv) > 1 else "bavii.csv",
        predict_hours=int(sys.argv[2]) if len(sys.argv) > 2 else 72,
        prediction_mode=(
            sys.argv[3] if len(sys.argv) > 3 else "score61_restored"
        ),
    )
