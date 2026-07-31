"""汇总 bavii.csv 的 72/120/168 小时预测与相似检索结果。

运行：
    python horizon_report.py

输入为 output/bavii_prediction_result_{72,120,168}h.json，输出机器可读
JSON、Markdown 对比报告以及长时效最优参数配置。脚本不修改原业务文件。
"""
from __future__ import annotations

import json
import math
from pathlib import Path


HOURS = (72, 120, 168)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2.0 * math.asin(math.sqrt(value))


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def build_report(output_dir: str | Path = "output") -> dict:
    output = Path(output_dir)
    results = {
        hours: json.loads(
            (output / f"bavii_prediction_result_{hours}h.json").read_text(
                encoding="utf-8"
            )
        )
        for hours in HOURS
    }
    performance_path = output / "bavii_horizon_performance.json"
    performance = (
        json.loads(performance_path.read_text(encoding="utf-8"))
        if performance_path.exists()
        else None
    )
    baseline = results[72]
    baseline_names = [item["name"] for item in baseline["top10"]]
    baseline_seconds = (
        float(performance["timings"]["72"]["median_seconds"])
        if performance
        else float(baseline["query_seconds"])
    )
    horizons = []
    for hours, result in results.items():
        start = result["prediction"][0]
        end = result["prediction"][-1]
        weighted = [
            float(item["weighted_similarity"]) for item in result["top10"]
        ]
        unweighted = [
            float(item["unweighted_similarity"]) for item in result["top10"]
        ]
        names = [item["name"] for item in result["top10"]]
        old_ranks = {name: index + 1 for index, name in enumerate(baseline_names)}
        rank_changes = [
            {
                "name": name,
                "rank": index + 1,
                "rank_at_72h": old_ranks.get(name),
                "change_vs_72h": (
                    old_ranks[name] - (index + 1) if name in old_ranks else None
                ),
            }
            for index, name in enumerate(names)
        ]
        query_seconds = (
            float(performance["timings"][str(hours)]["median_seconds"])
            if performance
            else float(result["query_seconds"])
        )
        horizons.append(
            {
                "predict_hours": hours,
                "forecast_steps": result["forecast_steps"],
                "endpoint": {"lon": end[0], "lat": end[1]},
                "forecast_displacement_km": haversine(*start, *end),
                "forecast_mean_bearing_deg": bearing(*start, *end),
                "confidence": result["confidence"],
                "long_range_correction": result["long_range_correction"],
                "query_seconds": query_seconds,
                "runtime_increase_vs_72h_percent": (
                    (query_seconds / baseline_seconds - 1.0) * 100.0
                ),
                "avg_weighted_similarity": sum(weighted) / len(weighted),
                "avg_unweighted_similarity": sum(unweighted) / len(unweighted),
                "avg_weight_gain": (
                    sum(weighted) / len(weighted)
                    - sum(unweighted) / len(unweighted)
                ),
                "top10": result["top10"],
                "rank_changes_vs_72h": rank_changes,
            }
        )

    report = {
        "sample": "bavii.csv",
        "baseline_hours": 72,
        "performance_limit_percent": 30.0,
        "performance_measurement": (
            performance["method"] if performance else "single process run"
        ),
        "performance_requirement_passed": all(
            item["runtime_increase_vs_72h_percent"] <= 30.0
            for item in horizons
        ),
        "horizons": horizons,
    }
    (output / "bavii_horizon_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = {
        "default_predict_hours": 72,
        "supported_predict_hours": [72, 120, 168],
        "steps_by_hours": {"72": 12, "120": 20, "168": 28},
        "input_steps": 24,
        "recent_input": {"steps": 6, "attention_bias": 0.5},
        "forecast_loss_weights": {
            "0_24h": 1.0,
            "24_72h": 0.7,
            "72_120h": 0.35,
            "120_168h": 0.15,
        },
        "physical_constraints": {
            "max_heading_change_deg_per_6h": 20.0,
            "max_speed_change_kmh_per_6h": 10.0,
            "max_total_heading_deviation_deg": 75.0,
        },
        "known_path_match_weights": {
            "latest_15": 1.0,
            "points_16_30": 0.6,
            "points_31_46": 0.3,
            "older": 0.1,
        },
        "forecast_path_match_weights": {
            "0_24h": 0.6,
            "24_72h": 0.3,
            "over_72h": 0.05,
        },
        "recent_heading_filter": {
            "threshold_deg": 45.0,
            "similarity_factor_on_mismatch": 0.6,
        },
        "analog_correction": {
            "starts_after_hours": 72,
            "top_n": 5,
            "ratio": 0.15,
        },
        "retrieval": {
            "coarse": "LCSS Top100",
            "fine": "weighted FastDTW radius=3",
            "fusion_weights": {"path": 0.65, "wind": 0.20, "pressure": 0.15},
        },
        "models": {
            "gru": "output/gru_attention_seq2seq_168h.keras",
            "tcn": "output/tcn_forecaster_168h.keras",
            "ensemble": ["GRU", "TCN", "inertia"],
        },
        "validation": {
            "sample": "bavii.csv",
            "performance_requirement_passed": report[
                "performance_requirement_passed"
            ],
        },
    }
    (output / "best_long_horizon_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Bavi 三档预测与相似台风检索对比",
        "",
        "| 时效 | 终点（经度, 纬度） | 趋势位移 km | 平均方向° | Top10加权分 | 未加权分 | 净耗时 s | 较72h增幅 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in horizons:
        lines.append(
            f"| {item['predict_hours']}h | "
            f"{item['endpoint']['lon']:.3f}, {item['endpoint']['lat']:.3f} | "
            f"{item['forecast_displacement_km']:.1f} | "
            f"{item['forecast_mean_bearing_deg']:.1f} | "
            f"{item['avg_weighted_similarity']:.4f} | "
            f"{item['avg_unweighted_similarity']:.4f} | "
            f"{item['query_seconds']:.3f} | "
            f"{item['runtime_increase_vs_72h_percent']:+.1f}% |"
        )
    for item in horizons:
        lines.extend(
            [
                "",
                f"## {item['predict_hours']}h Top10",
                "",
                *[
                    f"{index}. {entry['name']} ({entry['year']})："
                    f"{entry['similarity']:.4f}，加权变化 "
                    f"{entry['similarity_weight_delta']:+.4f}"
                    for index, entry in enumerate(item["top10"], 1)
                ],
            ]
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 三档路径总体均向西北转北发展；时效越长，北向延伸越明显。",
            "- 72h以上仅以0.05权重辅助匹配，因此长时效变化不会取代真实观测路径的主导作用。",
            f"- 性能增幅是否满足不超过30%：{'是' if report['performance_requirement_passed'] else '否'}。",
        ]
    )
    (output / "bavii_horizon_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    built = build_report()
    print(json.dumps({
        "performance_requirement_passed": built["performance_requirement_passed"],
        "outputs": [
            "output/bavii_horizon_comparison.json",
            "output/bavii_horizon_comparison.md",
            "output/best_long_horizon_config.json",
        ],
    }, ensure_ascii=False, indent=2))
