"""预测路径 + 相似台风后续路径的无泄漏融合优化。

流程：
1. 从 SQLite 固定抽取 30 个完整历史台风，20 个调参、10 个独立留出。
2. 每个样本只暴露前 60% 路径；目标台风自身从相似候选库中排除。
3. 基础模型先预测72小时，再用历史候选的后续相对位移构造相似集合预报。
4. 融合参数只在20个调参样本选择，随后冻结并评估10个留出样本与Bavi断点。
5. 真实未来路径只用于最终误差计算，不参与候选排序、融合或轨迹校正。
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import algorithm_lab as lab
import bavi_rollback_optimize as bavi_lab


OUTPUT_DIR = Path("output")
SEED = 2026
MODEL_STEP_HOURS = 3
DISPLAY_STEP_HOURS = 6
NATIVE_STEPS = 24
DISPLAY_STEPS = 12


@dataclass
class FusionCase:
    typhoon_id: str
    name: str
    year: int
    known: list[lab.TrackPoint]
    truth: np.ndarray  # 12×[lon,lat]，6小时间隔


def _interval_hours(points: list[lab.TrackPoint]) -> float:
    return lab._average_step_hours(points)


def build_fixed_cases(records: list[dict]) -> tuple[list[FusionCase], list[FusionCase]]:
    eligible = []
    for record in records:
        points = record["points"]
        if len(points) < 64:
            continue
        interval = _interval_hours(points)
        if not 2.0 <= interval <= 4.0:
            continue
        cutoff = max(24, int(len(points) * 0.60))
        if cutoff + NATIVE_STEPS > len(points):
            cutoff = len(points) - NATIVE_STEPS
        if cutoff < 24:
            continue
        future = lab._path(points[cutoff:cutoff + NATIVE_STEPS])
        truth = future[1::2][:DISPLAY_STEPS]
        if len(truth) != DISPLAY_STEPS:
            continue
        eligible.append(FusionCase(
            record["id"], record["name"], record["year"],
            points[:cutoff], truth,
        ))
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(eligible))[:30]
    fixed = [eligible[int(index)] for index in indices]
    return fixed[:20], fixed[20:]


def _bearing_vector(path: np.ndarray) -> float:
    delta = np.mean(np.diff(path[-min(8, len(path)):], axis=0), axis=0)
    return float(math.atan2(delta[1], delta[0]))


def _rotate(values: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.asarray([[cosine, -sine], [sine, cosine]])
    return values @ matrix.T


def _window_distance(query: np.ndarray, candidate: np.ndarray) -> tuple[float, int]:
    """向量化滑窗平均Haversine距离。"""
    length = len(query)
    if len(candidate) < length + NATIVE_STEPS:
        return float("inf"), -1
    starts = np.arange(0, len(candidate) - length - NATIVE_STEPS + 1, 2)
    if len(starts) == 0:
        return float("inf"), -1
    best_distance, best_start = float("inf"), -1
    query_lon = np.radians(query[:, 0])
    query_lat = np.radians(query[:, 1])
    for start in starts:
        window = candidate[start:start + length]
        lon = np.radians(window[:, 0])
        lat = np.radians(window[:, 1])
        dlon, dlat = lon - query_lon, lat - query_lat
        value = (
            np.sin(dlat / 2) ** 2
            + np.cos(query_lat) * np.cos(lat) * np.sin(dlon / 2) ** 2
        )
        distance = float(np.mean(
            6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
        ))
        if distance < best_distance:
            best_distance, best_start = distance, int(start)
    return best_distance, best_start


def build_analog_pool(
    case: FusionCase, records: list[dict], limit=180
) -> list[dict]:
    """仅使用已知路径建立候选池，并明确排除目标台风自身。"""
    query = lab._path(case.known[-24:])
    query_start, query_end = query[0], query[-1]
    coarse = []
    for record in records:
        if record["id"] == case.typhoon_id:
            continue
        path = lab._path(record["points"])
        if len(path) < len(query) + NATIVE_STEPS:
            continue
        sampled = path[::max(1, len(path) // 24)]
        end_distance = min(
            lab.haversine(*query_end, *point) for point in sampled
        )
        start_distance = min(
            lab.haversine(*query_start, *point) for point in sampled
        )
        if end_distance <= 1300 and start_distance <= 1800:
            coarse.append((0.7 * end_distance + 0.3 * start_distance, record))
    coarse.sort(key=lambda item: item[0])

    query_heading = _bearing_vector(query)
    query_speed = float(np.mean(np.linalg.norm(np.diff(query[-8:], axis=0), axis=1)))
    pool = []
    for _, record in coarse[:350]:
        path = lab._path(record["points"])
        known_distance, start = _window_distance(query, path)
        if start < 0:
            continue
        anchor = start + len(query)
        continuation_native = path[anchor:anchor + NATIVE_STEPS]
        if len(continuation_native) < NATIVE_STEPS:
            continue
        candidate_anchor = path[anchor - 1]
        displacement_native = continuation_native - candidate_anchor
        raw = case_known_anchor(case) + displacement_native

        matched = path[start:anchor]
        candidate_heading = _bearing_vector(matched)
        candidate_speed = float(np.mean(
            np.linalg.norm(np.diff(matched[-8:], axis=0), axis=1)
        ))
        speed_scale = float(np.clip(
            query_speed / max(candidate_speed, 1e-6), 0.55, 1.65
        ))
        aligned_displacement = _rotate(
            displacement_native, query_heading - candidate_heading
        ) * speed_scale
        aligned = case_known_anchor(case) + aligned_displacement
        pool.append({
            "id": record["id"],
            "name": record["name"],
            "year": record["year"],
            "known_distance_km": known_distance,
            "match_start": start,
            "forecast_anchor": anchor,
            "raw_prediction": raw[1::2][:DISPLAY_STEPS],
            "aligned_prediction": aligned[1::2][:DISPLAY_STEPS],
        })
    pool.sort(key=lambda item: item["known_distance_km"])
    return pool[:limit]


def case_known_anchor(case: FusionCase) -> np.ndarray:
    return np.asarray([case.known[-1].lon, case.known[-1].lat], dtype=float)


def _recent_inertia(
    known: list[lab.TrackPoint], window: int, decay: float
) -> np.ndarray:
    path = lab._path(known)
    count = min(window, len(path) - 1)
    differences = np.diff(path[-count - 1:], axis=0)
    weights = np.exp(np.linspace(-1.2, 0.0, len(differences)))
    step = np.average(differences, axis=0, weights=weights)
    current, output = path[-1].copy(), []
    for _ in range(NATIVE_STEPS):
        current = current + step
        output.append(current.copy())
        step *= decay
    return np.asarray(output)[1::2][:DISPLAY_STEPS]


def base_predictions(
    case: FusionCase,
    algorithm: lab.DirectedOptimizedAlgorithm,
) -> dict[str, np.ndarray]:
    group = algorithm.store.features.get(case.typhoon_id, {}).get(
        "strength_group", 0
    )
    components = algorithm.forecaster.predict_components(
        case.known[-24:], NATIVE_STEPS, int(group)
    )
    current = np.average(
        components, axis=0, weights=np.asarray([0.45, 0.30, 0.25])
    )[1::2][:DISPLAY_STEPS]
    inertia6 = _recent_inertia(case.known, 6, 1.0)
    inertia24 = _recent_inertia(case.known, 24, 1.02)
    return {
        "current_aligned": current,
        "inertia6": inertia6,
        "inertia24_decay102": inertia24,
        "current_inertia50": 0.5 * current + 0.5 * inertia6,
    }


def mean_path_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean([
        lab.haversine(*a, *b) for a, b in zip(first, second)
    ]))


def fuse_prediction(
    base: np.ndarray,
    pool: list[dict],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    variant = (
        "aligned_prediction" if config["motion_align"] else "raw_prediction"
    )
    scored = []
    future_weight = config["future_match_weight"]
    for item in pool:
        candidate = item[variant]
        base_distance = mean_path_distance(base, candidate)
        distance = (
            (1.0 - future_weight) * item["known_distance_km"]
            + future_weight * base_distance
        )
        scored.append((distance, item))
    scored.sort(key=lambda value: value[0])
    selected = scored[:config["top_k"]]
    if not selected:
        return base.copy(), []
    weights = np.asarray([
        math.exp(-distance / 250.0) for distance, _ in selected
    ])
    analog = np.average(
        np.stack([item[variant] for _, item in selected]),
        axis=0,
        weights=weights,
    )
    alpha = np.linspace(
        config["alpha_short"], config["alpha_long"], DISPLAY_STEPS
    )[:, None]
    fused = (1.0 - alpha) * base + alpha * analog

    final_rank = []
    for item in pool:
        candidate = item[variant]
        future_distance = mean_path_distance(fused, candidate)
        distance = (
            (1.0 - future_weight) * item["known_distance_km"]
            + future_weight * future_distance
        )
        final_rank.append({
            **{key: value for key, value in item.items()
               if key not in ("raw_prediction", "aligned_prediction")},
            "predicted_continuation": candidate,
            "distance_km": distance,
            "similarity": math.exp(-distance / 250.0),
        })
    final_rank.sort(key=lambda item: item["distance_km"])
    return fused, final_rank


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    errors = np.asarray([
        lab.haversine(*predicted, *actual)
        for predicted, actual in zip(prediction, truth)
    ])
    return {
        "ade_24h_km": float(np.mean(errors[:4])),
        "fde_24h_km": float(errors[3]),
        "ade_48h_km": float(np.mean(errors[:8])),
        "fde_48h_km": float(errors[7]),
        "ade_72h_km": float(np.mean(errors)),
        "fde_72h_km": float(errors[-1]),
        "errors_km": errors.tolist(),
    }


def retrieval_metrics(ranked: list[dict], truth: np.ndarray) -> dict:
    top10 = ranked[:10]
    errors = [
        mean_path_distance(item["predicted_continuation"], truth)
        for item in top10
    ]
    scores = [item["similarity"] for item in top10]
    correlation = 0.0
    if len(errors) >= 2 and np.std(errors) > 0 and np.std(scores) > 0:
        correlation = float(np.corrcoef(scores, -np.asarray(errors))[0, 1])
    return {
        "top5_future_ade_km": float(np.mean(errors[:5])) if errors else float("inf"),
        "top10_overlap": float(np.mean(
            np.exp(-np.asarray(errors) / 300.0)
        )) if errors else 0.0,
        "rank_correlation": correlation,
    }


def summarize(outputs: list[dict]) -> dict:
    prediction_keys = (
        "ade_24h_km", "fde_24h_km", "ade_48h_km",
        "fde_48h_km", "ade_72h_km", "fde_72h_km",
    )
    retrieval_keys = (
        "top5_future_ade_km", "top10_overlap", "rank_correlation",
    )
    result = {
        key: float(np.mean([
            item["prediction_metrics"][key] for item in outputs
        ]))
        for key in prediction_keys
    }
    result.update({
        key: float(np.mean([
            item["retrieval_metrics"][key] for item in outputs
        ]))
        for key in retrieval_keys
    })
    result["objective"] = (
        result["ade_72h_km"]
        + 0.30 * result["fde_72h_km"]
        + 0.20 * result["top5_future_ade_km"]
        - 50.0 * result["top10_overlap"]
        - 10.0 * result["rank_correlation"]
    )
    return result


def evaluate_config(
    cases: list[FusionCase],
    bases: dict[str, dict[str, np.ndarray]],
    pools: dict[str, list[dict]],
    config: dict,
) -> tuple[dict, list[dict]]:
    outputs = []
    for case in cases:
        prediction, ranked = fuse_prediction(
            bases[case.typhoon_id][config["base_model"]],
            pools[case.typhoon_id],
            config,
        )
        outputs.append({
            "typhoon_id": case.typhoon_id,
            "name": case.name,
            "year": case.year,
            "prediction": prediction,
            "ranked": ranked,
            "prediction_metrics": prediction_metrics(prediction, case.truth),
            "retrieval_metrics": retrieval_metrics(ranked, case.truth),
        })
    return summarize(outputs), outputs


def parameter_grid() -> list[dict]:
    values = []
    for base_model in (
        "current_aligned", "inertia6",
        "inertia24_decay102", "current_inertia50",
    ):
        for top_k in (3, 5, 8):
            for future_weight in (0.25, 0.50):
                for alpha_short, alpha_long in (
                    (0.15, 0.35), (0.25, 0.50),
                    (0.35, 0.65), (0.50, 0.70),
                ):
                    for motion_align in (False, True):
                        values.append({
                            "base_model": base_model,
                            "top_k": top_k,
                            "future_match_weight": future_weight,
                            "alpha_short": alpha_short,
                            "alpha_long": alpha_long,
                            "motion_align": motion_align,
                        })
    return values


def baseline_config(base_model: str) -> dict:
    return {
        "base_model": base_model,
        "top_k": 3,
        "future_match_weight": 0.25,
        "alpha_short": 0.0,
        "alpha_long": 0.0,
        "motion_align": False,
    }


def bavi_case() -> FusionCase:
    track = bavi_lab.load_bavi()
    cutoff = datetime(2026, 7, 10, 14)
    known = track.grid(track.times[0], cutoff, MODEL_STEP_HOURS)
    truth = bavi_lab.truth_at_6h(track, cutoff)
    return FusionCase("BAVI_EXTERNAL", "Bavi外部检验", 2026, known, truth)


def write_case_csv(case: FusionCase, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "lat", "lon", "pressure", "wind_speed"])
        for point in case.known:
            writer.writerow([
                point.time, point.lat, point.lon,
                point.pressure, point.wind_speed,
            ])


def serializable_rank(ranked: list[dict], limit=10) -> list[dict]:
    return [
        {
            **{key: value for key, value in item.items()
               if key != "predicted_continuation"},
            "predicted_continuation": item[
                "predicted_continuation"
            ].tolist(),
        }
        for item in ranked[:limit]
    ]


def create_map(
    case: FusionCase,
    prediction: np.ndarray,
    ranked: list[dict],
    records_by_id: dict[str, dict],
    output: Path,
    title: str,
) -> None:
    end = case.known[-1]
    map_object = folium.Map([end.lat, end.lon], zoom_start=5)
    folium.PolyLine(
        [(point.lat, point.lon) for point in case.known],
        color="black", weight=5, tooltip="新台风已知路径",
    ).add_to(map_object)
    folium.PolyLine(
        [(lat, lon) for lon, lat in prediction],
        color="red", weight=4, tooltip="预测-相似路径融合结果",
    ).add_to(map_object)
    folium.PolyLine(
        [(lat, lon) for lon, lat in case.truth],
        color="green", weight=4, tooltip="留出真实路径（仅评分）",
    ).add_to(map_object)
    colors = [
        "blue", "purple", "orange", "darkred", "cadetblue",
        "darkgreen", "pink", "darkblue", "lightred", "beige",
    ]
    for index, item in enumerate(ranked[:10]):
        record = records_by_id[item["id"]]
        start = item["match_start"]
        anchor = item["forecast_anchor"]
        selected = record["points"][
            start:min(len(record["points"]), anchor + NATIVE_STEPS)
        ]
        color = colors[index % len(colors)]
        label = (
            f"{index + 1}. {item['name']} ({item['year']}) "
            f"{item['similarity']:.3f}"
        )
        folium.PolyLine(
            [(point.lat, point.lon) for point in selected],
            color=color, weight=1.6, opacity=0.75, tooltip=label,
        ).add_to(map_object)
        for point in selected:
            folium.CircleMarker(
                [point.lat, point.lon], radius=1.0, color=color,
                fill=True, fill_opacity=0.55, weight=1,
            ).add_to(map_object)
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:10px;border:1px solid #777;font-size:12px;'
        f'line-height:1.5"><b>{title}</b><br>'
        '<span style="color:black">━</span> 已知路径<br>'
        '<span style="color:red">━</span> 融合预测<br>'
        '<span style="color:green">━</span> 留出真值（仅评分）</div>'
    )
    map_object.get_root().html.add_child(folium.Element(legend))
    map_object.save(output)


def run() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    started = time.perf_counter()
    records = lab.clean_records(lab.SQLiteReader().load())
    records_by_id = {record["id"]: record for record in records}
    tuning_cases, holdout_cases = build_fixed_cases(records)
    all_cases = [*tuning_cases, *holdout_cases]
    algorithm = lab.DirectedOptimizedAlgorithm(records)

    bases, pools = {}, {}
    for index, case in enumerate(all_cases, start=1):
        bases[case.typhoon_id] = base_predictions(case, algorithm)
        pools[case.typhoon_id] = build_analog_pool(case, records)
        print(
            f"预计算 {index:02d}/{len(all_cases)} "
            f"{case.name}，候选{len(pools[case.typhoon_id])}条"
        )

    tuning_results = []
    for config in parameter_grid():
        metrics, _ = evaluate_config(tuning_cases, bases, pools, config)
        tuning_results.append({"config": config, "metrics": metrics})
    tuning_results.sort(key=lambda item: item["metrics"]["objective"])
    selected_config = tuning_results[0]["config"]
    tuning_metrics, tuning_outputs = evaluate_config(
        tuning_cases, bases, pools, selected_config
    )
    holdout_metrics, holdout_outputs = evaluate_config(
        holdout_cases, bases, pools, selected_config
    )

    baseline_results = []
    for base_model in bases[all_cases[0].typhoon_id]:
        config = baseline_config(base_model)
        tune_summary, _ = evaluate_config(tuning_cases, bases, pools, config)
        baseline_results.append({
            "config": config, "tuning_metrics": tune_summary,
        })
    baseline_results.sort(
        key=lambda item: item["tuning_metrics"]["objective"]
    )
    selected_baseline = baseline_results[0]["config"]
    baseline_holdout, baseline_holdout_outputs = evaluate_config(
        holdout_cases, bases, pools, selected_baseline
    )
    accepted = (
        holdout_metrics["objective"] < baseline_holdout["objective"]
        and holdout_metrics["ade_72h_km"] < baseline_holdout["ade_72h_km"]
    )

    external = bavi_case()
    external_bases = base_predictions(external, algorithm)
    external_pool = build_analog_pool(external, records)
    external_prediction, external_rank = fuse_prediction(
        external_bases[selected_config["base_model"]],
        external_pool,
        selected_config,
    )
    external_baseline, _ = fuse_prediction(
        external_bases[selected_baseline["base_model"]],
        external_pool,
        selected_baseline,
    )
    external_metrics = prediction_metrics(external_prediction, external.truth)
    external_baseline_metrics = prediction_metrics(
        external_baseline, external.truth
    )
    external_retrieval = retrieval_metrics(external_rank, external.truth)

    example = holdout_outputs[0]
    example_case = next(
        case for case in holdout_cases
        if case.typhoon_id == example["typhoon_id"]
    )
    write_case_csv(
        example_case, OUTPUT_DIR / "synthetic_new_typhoon.csv"
    )
    create_map(
        example_case, example["prediction"], example["ranked"],
        records_by_id, OUTPUT_DIR / "synthetic_analog_fusion_map.html",
        "数据库截断新台风：预测-相似路径融合",
    )
    create_map(
        external, external_prediction, external_rank,
        records_by_id, OUTPUT_DIR / "bavi_analog_fusion_map.html",
        "Bavi外部样本：预测-相似路径融合",
    )

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available_fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["24h ADE", "48h ADE", "72h ADE", "72h FDE"]
    baseline_values = [
        baseline_holdout["ade_24h_km"],
        baseline_holdout["ade_48h_km"],
        baseline_holdout["ade_72h_km"],
        baseline_holdout["fde_72h_km"],
    ]
    fusion_values = [
        holdout_metrics["ade_24h_km"],
        holdout_metrics["ade_48h_km"],
        holdout_metrics["ade_72h_km"],
        holdout_metrics["fde_72h_km"],
    ]
    positions = np.arange(len(labels))
    axes[0].bar(positions - 0.18, baseline_values, 0.36, label="基础模型")
    axes[0].bar(positions + 0.18, fusion_values, 0.36, label="相似路径融合")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("误差（km）")
    axes[0].set_title("10台风独立留出集")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    known_path = lab._path(external.known)
    axes[1].plot(
        known_path[:, 0], known_path[:, 1],
        color="black", linewidth=2.5, label="断点前观测",
    )
    axes[1].plot(
        external_prediction[:, 0], external_prediction[:, 1],
        color="red", marker="o", label="融合预测",
    )
    axes[1].plot(
        external_baseline[:, 0], external_baseline[:, 1],
        color="#2563EB", linestyle="--", label="基础预测",
    )
    axes[1].plot(
        external.truth[:, 0], external.truth[:, 1],
        color="green", marker="o", label="留出真值",
    )
    axes[1].set_title("Bavi外部断点检验")
    axes[1].set_xlabel("经度")
    axes[1].set_ylabel("纬度")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "analog_fusion_comparison.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "method": "prediction_analog_path_feedback_fusion",
        "seed": SEED,
        "tuning_size": len(tuning_cases),
        "holdout_size": len(holdout_cases),
        "target_typhoon_excluded_from_analog_pool": True,
        "truth_used_for_candidate_ranking": False,
        "truth_used_for_fusion": False,
        "selected_config": selected_config,
        "selected_tuning_metrics": tuning_metrics,
        "holdout_metrics": holdout_metrics,
        "baseline_config": selected_baseline,
        "baseline_holdout_metrics": baseline_holdout,
        "accepted_for_integration": accepted,
        "holdout_objective_improvement_pct": (
            (baseline_holdout["objective"] - holdout_metrics["objective"])
            / max(baseline_holdout["objective"], 1e-12) * 100.0
        ),
        "bavi_external_metrics": external_metrics,
        "bavi_external_baseline_metrics": external_baseline_metrics,
        "bavi_external_retrieval_metrics": external_retrieval,
        "tuning_ranking_top20": tuning_results[:20],
        "tuning_case_ids": [case.typhoon_id for case in tuning_cases],
        "holdout_case_ids": [case.typhoon_id for case in holdout_cases],
        "synthetic_example": {
            "id": example_case.typhoon_id,
            "name": example_case.name,
            "year": example_case.year,
            "known_points": len(example_case.known),
            "prediction_metrics": example["prediction_metrics"],
            "retrieval_metrics": example["retrieval_metrics"],
            "top10": serializable_rank(example["ranked"]),
        },
        "bavi_top10": serializable_rank(external_rank),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "synthetic_csv": "output/synthetic_new_typhoon.csv",
            "synthetic_map": "output/synthetic_analog_fusion_map.html",
            "bavi_map": "output/bavi_analog_fusion_map.html",
            "figure": "output/analog_fusion_comparison.png",
            "config": "output/best_analog_fusion_config.json",
        },
    }
    (OUTPUT_DIR / "analog_fusion_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "best_analog_fusion_config.json").write_text(
        json.dumps({
            "accepted": accepted,
            "config": selected_config,
            "validation": "20 tuning + 10 untouched holdout + Bavi external",
            "truth_fitting": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "selected_config": result["selected_config"],
        "baseline_holdout_metrics": result["baseline_holdout_metrics"],
        "fusion_holdout_metrics": result["holdout_metrics"],
        "accepted_for_integration": result["accepted_for_integration"],
        "holdout_objective_improvement_pct": result[
            "holdout_objective_improvement_pct"
        ],
        "bavi_external_baseline_metrics": result[
            "bavi_external_baseline_metrics"
        ],
        "bavi_external_metrics": result["bavi_external_metrics"],
    }, ensure_ascii=False, indent=2))
