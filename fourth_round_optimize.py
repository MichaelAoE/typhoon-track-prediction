"""第4轮：单台风截断预测、参数搜索与旧配置兼容审计。

规则：
1. 选择未参加既有开发/留出/前三轮实验的历史台风；
2. 仅依据断点前路径特征选样，不查看未来真值；
3. 前60%为新台风输入，后续72小时只用于候选参数评分；
4. 在单例最优值3%以内，优先选择与旧配置最接近的参数；
5. 候选只在固定10台风审计集上检查一次，审计不通过则保留旧配置。
"""
from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import algorithm_lab as lab
import analog_fusion_optimize as fusion
import three_round_iterative_optimize as previous_rounds


OUTPUT_DIR = Path("output")
REPORT_PATH = OUTPUT_DIR / "fourth_round_optimization_report.json"
CONFIG_PATH = OUTPUT_DIR / "best_analog_fusion_config.json"
SELECTION_SEED = 2041


def _case_from_record(record: dict) -> fusion.FusionCase | None:
    points = record["points"]
    if len(points) < 64:
        return None
    interval = lab._average_step_hours(points)
    if not 2.0 <= interval <= 4.0:
        return None
    cutoff = max(24, int(len(points) * 0.60))
    if cutoff + fusion.NATIVE_STEPS > len(points):
        cutoff = len(points) - fusion.NATIVE_STEPS
    if cutoff < 24:
        return None
    future_native = lab._path(points[cutoff:cutoff + fusion.NATIVE_STEPS])
    truth = future_native[1::2][:fusion.DISPLAY_STEPS]
    if len(truth) != fusion.DISPLAY_STEPS:
        return None
    return fusion.FusionCase(
        record["id"], record["name"], record["year"], points[:cutoff], truth
    )


def _bearing_degrees(first: lab.TrackPoint, second: lab.TrackPoint) -> float:
    lon1, lat1, lon2, lat2 = map(
        math.radians, [first.lon, first.lat, second.lon, second.lat]
    )
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _known_selection_features(case: fusion.FusionCase) -> dict:
    recent = case.known[-min(24, len(case.known)):]
    step_hours = max(lab._average_step_hours(recent), 1e-6)
    speeds = [
        lab.haversine(a.lon, a.lat, b.lon, b.lat) / step_hours
        for a, b in zip(recent, recent[1:])
    ]
    bearings = [
        _bearing_degrees(a, b) for a, b in zip(recent, recent[1:])
    ]
    turns = [
        abs((right - left + 180.0) % 360.0 - 180.0)
        for left, right in zip(bearings, bearings[1:])
    ]
    wind_valid = [
        p.wind_speed for p in case.known if p.wind_speed and p.wind_speed > 0
    ]
    pressure_valid = [
        p.pressure for p in case.known
        if p.pressure and 850 <= p.pressure < 1013
    ]
    return {
        "mean_recent_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "mean_recent_turn_deg": float(np.mean(turns)) if turns else 0.0,
        "known_path_distance_km": float(sum(
            lab.haversine(a.lon, a.lat, b.lon, b.lat)
            for a, b in zip(case.known, case.known[1:])
        )),
        "wind_valid_ratio": len(wind_valid) / max(len(case.known), 1),
        "pressure_valid_ratio": len(pressure_valid) / max(len(case.known), 1),
    }


def choose_new_case(
    records: list[dict], excluded_ids: set[str]
) -> tuple[fusion.FusionCase, dict]:
    """只按已知段选取一个现代、命名且运动特征居中的台风。"""
    candidates = []
    for record in records:
        if record["id"] in excluded_ids:
            continue
        case = _case_from_record(record)
        if case is None:
            continue
        if case.year < 1980:
            continue
        name = str(case.name).strip().upper()
        if not name or name in {"NOT NAMED", "UNNAMED"} or name == case.typhoon_id:
            continue
        features = _known_selection_features(case)
        if not 4.0 <= features["mean_recent_speed_kmh"] <= 60.0:
            continue
        if features["wind_valid_ratio"] < 0.50:
            continue
        candidates.append((case, features))
    if not candidates:
        raise RuntimeError("没有找到满足已知段质量条件且未参与旧实验的台风")

    turns = np.asarray([item[1]["mean_recent_turn_deg"] for item in candidates])
    speeds = np.asarray([item[1]["mean_recent_speed_kmh"] for item in candidates])
    median_turn = float(np.median(turns))
    median_speed = float(np.median(speeds))
    for case, features in candidates:
        features["selection_distance"] = (
            abs(features["mean_recent_turn_deg"] - median_turn)
            / max(float(np.std(turns)), 1.0)
            + 0.35 * abs(features["mean_recent_speed_kmh"] - median_speed)
            / max(float(np.std(speeds)), 1.0)
            - 0.03 * min(case.year - 1980, 40) / 40.0
        )
    candidates.sort(key=lambda item: (
        item[1]["selection_distance"], -item[0].year, item[0].typhoon_id
    ))
    case, features = candidates[0]
    features.update({
        "eligible_count": len(candidates),
        "median_turn_deg": median_turn,
        "median_speed_kmh": median_speed,
        "selection_uses_future_truth": False,
        "selection_seed": SELECTION_SEED,
    })
    return case, features


def fine_parameter_grid(previous: dict) -> list[dict]:
    values = []
    for base_model in (
        "current_aligned",
        "inertia6",
        "inertia24_decay102",
        "current_inertia50",
    ):
        for top_k in (3, 5, 8):
            for future_weight in (0.25, 0.35, 0.50, 0.65, 0.75):
                for alpha_short, alpha_long in (
                    (0.10, 0.25),
                    (0.15, 0.35),
                    (0.25, 0.50),
                    (0.35, 0.65),
                    (0.45, 0.75),
                    (0.55, 0.85),
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
    if previous not in values:
        values.append(dict(previous))
    return values


def parameter_distance(config: dict, previous: dict) -> float:
    base_distance = {
        ("current_inertia50", "current_inertia50"): 0.0,
        ("current_aligned", "current_inertia50"): 0.45,
        ("inertia6", "current_inertia50"): 0.45,
        ("inertia24_decay102", "current_inertia50"): 0.65,
    }.get((config["base_model"], previous["base_model"]), 0.75)
    return float(
        base_distance
        + abs(config["top_k"] - previous["top_k"]) / 5.0
        + abs(
            config["future_match_weight"]
            - previous["future_match_weight"]
        ) / 0.50
        + abs(config["alpha_short"] - previous["alpha_short"]) / 0.50
        + abs(config["alpha_long"] - previous["alpha_long"]) / 0.60
        + (0.35 if config["motion_align"] != previous["motion_align"] else 0.0)
    )


def write_known_csv(case: fusion.FusionCase, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "lat", "lon", "pressure", "wind_speed"])
        for point in case.known:
            writer.writerow([
                point.time, point.lat, point.lon,
                point.pressure, point.wind_speed,
            ])


def write_prediction_csv(
    case: fusion.FusionCase,
    previous_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    path: Path,
) -> None:
    last_time = case.known[-1].time
    if isinstance(last_time, str):
        # SQLite 读取层为兼容旧缓存可能保留 ISO 时间字符串；导出前统一转回 datetime。
        last_time = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time",
            "truth_lon", "truth_lat",
            "previous_lon", "previous_lat",
            "candidate_lon", "candidate_lat",
            "previous_error_km", "candidate_error_km",
        ])
        for index, (truth, old, new) in enumerate(zip(
            case.truth, previous_prediction, candidate_prediction
        ), start=1):
            writer.writerow([
                last_time + timedelta(hours=6 * index),
                truth[0], truth[1],
                old[0], old[1],
                new[0], new[1],
                lab.haversine(*old, *truth),
                lab.haversine(*new, *truth),
            ])


def configure_fonts() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    elif "SimHei" in available:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def create_comparison_figure(
    case: fusion.FusionCase,
    previous_config: dict,
    candidate_config: dict,
    previous_output: dict,
    candidate_output: dict,
    previous_audit: dict,
    candidate_audit: dict,
    path: Path,
) -> None:
    configure_fonts()
    known = lab._path(case.known)
    truth = case.truth
    previous_prediction = previous_output["prediction"]
    candidate_prediction = candidate_output["prediction"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(
        known[:, 0], known[:, 1], color="black", linewidth=2.5,
        label="断点前已知路径",
    )
    axes[0, 0].plot(
        truth[:, 0], truth[:, 1], "o-", color="#16803C",
        linewidth=2.2, markersize=4, label="断点后真实路径（仅评估）",
    )
    axes[0, 0].plot(
        previous_prediction[:, 0], previous_prediction[:, 1], "--",
        color="#2563EB", linewidth=2, label="先前参数预测",
    )
    axes[0, 0].plot(
        candidate_prediction[:, 0], candidate_prediction[:, 1], "o-",
        color="#DC2626", linewidth=2, markersize=3.5, label="本轮候选预测",
    )
    axes[0, 0].set_title(f"{case.name}（{case.year}）截断预测路径")
    axes[0, 0].set_xlabel("经度")
    axes[0, 0].set_ylabel("纬度")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    hours = np.arange(6, 73, 6)
    previous_errors = previous_output["prediction_metrics"]["errors_km"]
    candidate_errors = candidate_output["prediction_metrics"]["errors_km"]
    axes[0, 1].plot(
        hours, previous_errors, "o--", color="#2563EB", label="先前参数"
    )
    axes[0, 1].plot(
        hours, candidate_errors, "o-", color="#DC2626", label="本轮候选"
    )
    axes[0, 1].set_title("逐时效路径误差")
    axes[0, 1].set_xlabel("预报时效（小时）")
    axes[0, 1].set_ylabel("Haversine误差（km）")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    parameter_labels = ["未来匹配权重", "短时融合α", "长时融合α"]
    old_values = [
        previous_config["future_match_weight"],
        previous_config["alpha_short"],
        previous_config["alpha_long"],
    ]
    new_values = [
        candidate_config["future_match_weight"],
        candidate_config["alpha_short"],
        candidate_config["alpha_long"],
    ]
    positions = np.arange(len(parameter_labels))
    axes[1, 0].bar(
        positions - 0.18, old_values, 0.36, label="先前参数", color="#2563EB"
    )
    axes[1, 0].bar(
        positions + 0.18, new_values, 0.36, label="本轮候选", color="#DC2626"
    )
    axes[1, 0].set_xticks(positions, parameter_labels)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title(
        f"权重对比（TopK {previous_config['top_k']}→{candidate_config['top_k']}）"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    audit_labels = ["72h ADE", "72h FDE", "Top5后续ADE", "综合目标"]
    old_audit = [
        previous_audit["ade_72h_km"],
        previous_audit["fde_72h_km"],
        previous_audit["top5_future_ade_km"],
        previous_audit["objective"],
    ]
    new_audit = [
        candidate_audit["ade_72h_km"],
        candidate_audit["fde_72h_km"],
        candidate_audit["top5_future_ade_km"],
        candidate_audit["objective"],
    ]
    positions = np.arange(len(audit_labels))
    axes[1, 1].bar(
        positions - 0.18, old_audit, 0.36, label="先前参数", color="#2563EB"
    )
    axes[1, 1].bar(
        positions + 0.18, new_audit, 0.36, label="本轮候选", color="#DC2626"
    )
    axes[1, 1].set_xticks(positions, audit_labels, rotation=12)
    axes[1, 1].set_title("固定10台风审计集对比（越低越好）")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(round_number: int = 4) -> dict:
    """运行指定轮次；默认仍为第4轮，保持原调用方式兼容。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_names = {
        4: "fourth_round_optimization_report.json",
        5: "fifth_round_optimization_report.json",
    }
    report_path = OUTPUT_DIR / report_names.get(
        round_number, f"round{round_number}_optimization_report.json"
    )
    started = time.perf_counter()
    records = lab.clean_records(lab.SQLiteReader().load())
    records_by_id = {record["id"]: record for record in records}

    analog_report = json.loads(
        (OUTPUT_DIR / "analog_fusion_optimization_report.json").read_text(
            encoding="utf-8"
        )
    )
    old_excluded = set(analog_report.get("tuning_case_ids", []))
    old_excluded.update(analog_report.get("holdout_case_ids", []))
    old_development, fixed_audit = previous_rounds.build_new_cases(
        records, old_excluded
    )
    all_previous_cases = [*old_development, *fixed_audit]
    excluded = old_excluded | {
        case.typhoon_id for case in all_previous_cases
    }
    # 后续轮次自动排除已经完成单例训练的台风，保证每轮使用新样本。
    for completed_round in range(4, round_number):
        completed_name = report_names.get(
            completed_round,
            f"round{completed_round}_optimization_report.json",
        )
        completed_path = OUTPUT_DIR / completed_name
        if completed_path.exists():
            completed_report = json.loads(
                completed_path.read_text(encoding="utf-8")
            )
            completed_id = completed_report.get("selected_case", {}).get("id")
            if completed_id:
                excluded.add(completed_id)

    selected_case, selection_features = choose_new_case(records, excluded)
    print(
        f"选择台风：{selected_case.name} ({selected_case.year}) "
        f"{selected_case.typhoon_id}，已知点数={len(selected_case.known)}"
    )

    algorithm = lab.DirectedOptimizedAlgorithm(records)
    cases = [selected_case, *fixed_audit]
    bases: dict[str, dict[str, np.ndarray]] = {}
    pools: dict[str, list[dict]] = {}
    for index, case in enumerate(cases, start=1):
        bases[case.typhoon_id] = fusion.base_predictions(case, algorithm)
        pools[case.typhoon_id] = fusion.build_analog_pool(case, records)
        print(
            f"预计算 {index:02d}/{len(cases)}：{case.name}，"
            f"候选={len(pools[case.typhoon_id])}"
        )

    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    previous_config = dict(saved["config"])
    grid = fine_parameter_grid(previous_config)
    search = []
    for config in grid:
        metrics, _ = fusion.evaluate_config(
            [selected_case], bases, pools, config
        )
        search.append({
            "config": config,
            "metrics": metrics,
            "distance_to_previous": parameter_distance(
                config, previous_config
            ),
        })
    search.sort(key=lambda item: item["metrics"]["objective"])
    raw_best = search[0]
    tolerance = raw_best["metrics"]["objective"] + 0.03 * abs(
        raw_best["metrics"]["objective"]
    )
    near_best = [
        item for item in search if item["metrics"]["objective"] <= tolerance
    ]
    near_best.sort(key=lambda item: (
        item["distance_to_previous"],
        item["metrics"]["objective"],
    ))
    adapted_candidate = near_best[0]
    candidate_config = adapted_candidate["config"]

    previous_case_metrics, previous_case_outputs = fusion.evaluate_config(
        [selected_case], bases, pools, previous_config
    )
    candidate_case_metrics, candidate_case_outputs = fusion.evaluate_config(
        [selected_case], bases, pools, candidate_config
    )
    previous_audit_metrics, _ = fusion.evaluate_config(
        fixed_audit, bases, pools, previous_config
    )
    candidate_audit_metrics, _ = fusion.evaluate_config(
        fixed_audit, bases, pools, candidate_config
    )
    accepted = (
        candidate_audit_metrics["objective"]
        < previous_audit_metrics["objective"]
        and candidate_audit_metrics["ade_72h_km"]
        <= previous_audit_metrics["ade_72h_km"] * 1.01
        and candidate_audit_metrics["top10_overlap"]
        >= previous_audit_metrics["top10_overlap"] * 0.98
    )
    final_config = dict(candidate_config if accepted else previous_config)
    final_case_metrics, final_case_outputs = fusion.evaluate_config(
        [selected_case], bases, pools, final_config
    )

    previous_output = previous_case_outputs[0]
    candidate_output = candidate_case_outputs[0]
    final_output = final_case_outputs[0]
    prefix = f"iteration_round{round_number}"
    known_csv = OUTPUT_DIR / f"{prefix}_new_typhoon.csv"
    prediction_csv = OUTPUT_DIR / f"{prefix}_prediction_vs_truth.csv"
    map_path = OUTPUT_DIR / f"{prefix}_map.html"
    figure_path = OUTPUT_DIR / f"{prefix}_comparison.png"
    write_known_csv(selected_case, known_csv)
    write_prediction_csv(
        selected_case,
        previous_output["prediction"],
        candidate_output["prediction"],
        prediction_csv,
    )
    fusion.create_map(
        selected_case,
        final_output["prediction"],
        final_output["ranked"],
        records_by_id,
        map_path,
        (
            f"第{round_number}轮：{selected_case.name}截断预测与相似台风"
            f"（候选参数{'已接纳' if accepted else '未接纳'}）"
        ),
    )
    create_comparison_figure(
        selected_case,
        previous_config,
        candidate_config,
        previous_output,
        candidate_output,
        previous_audit_metrics,
        candidate_audit_metrics,
        figure_path,
    )

    audit_improvement = (
        (
            previous_audit_metrics["objective"]
            - candidate_audit_metrics["objective"]
        )
        / max(previous_audit_metrics["objective"], 1e-12)
        * 100.0
    )
    report = {
        "method": (
            f"round{round_number}_single_case_tuning_with_fixed_audit"
        ),
        "selected_case": {
            "id": selected_case.typhoon_id,
            "name": selected_case.name,
            "year": selected_case.year,
            "known_points": len(selected_case.known),
            "known_start": str(selected_case.known[0].time),
            "cutoff_time": str(selected_case.known[-1].time),
            "forecast_hours": 72,
        },
        "selection_features_known_only": selection_features,
        "excluded_previous_case_count": len(excluded),
        "fixed_audit_case_ids": [
            case.typhoon_id for case in fixed_audit
        ],
        "grid_size": len(grid),
        "near_best_tolerance": "single-case objective within raw best +3%",
        "previous_config": previous_config,
        "raw_single_case_best": raw_best,
        "adapted_candidate": adapted_candidate,
        "candidate_config": candidate_config,
        "final_config": final_config,
        "candidate_accepted": accepted,
        "acceptance_rule": {
            "audit_objective_must_improve": True,
            "audit_ade72_max_degradation": "1%",
            "audit_top10_overlap_max_degradation": "2%",
        },
        "selected_case_metrics": {
            "previous": previous_case_metrics,
            "candidate": candidate_case_metrics,
            "final": final_case_metrics,
        },
        "fixed_audit_metrics": {
            "previous": previous_audit_metrics,
            "candidate": candidate_audit_metrics,
            "objective_improvement_pct": audit_improvement,
        },
        "candidate_top10": fusion.serializable_rank(
            candidate_output["ranked"]
        ),
        "final_top10": fusion.serializable_rank(final_output["ranked"]),
        "search_top20": search[:20],
        "leakage_audit": {
            "case_selected_using_known_segment_only": True,
            "future_truth_used_for_parameter_scoring": True,
            "future_truth_used_for_prediction_generation": False,
            "future_truth_used_for_candidate_ranking": False,
            "target_typhoon_excluded_from_analog_pool": True,
            "fixed_audit_used_once_for_accept_reject": True,
            "residual_path_fitting_enabled": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "new_typhoon_csv": str(known_csv),
            "prediction_vs_truth_csv": str(prediction_csv),
            "map": str(map_path),
            "comparison_figure": str(figure_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if accepted:
        CONFIG_PATH.write_text(
            json.dumps({
                "accepted": True,
                "config": final_config,
                "validation": (
                    "20 tuning + 10 untouched holdout + Bavi external + "
                    f"round{round_number} single-case development with "
                    "fixed 10-case audit"
                ),
                "truth_fitting": False,
                "previous_config": previous_config,
                "latest_round": round_number,
                "latest_round_case": selected_case.typhoon_id,
                "latest_round_audit_objective_improvement_pct": (
                    audit_improvement
                ),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({
        "selected_case": report["selected_case"],
        "previous_config": previous_config,
        "raw_best_config": raw_best["config"],
        "adapted_candidate": candidate_config,
        "candidate_accepted": accepted,
        "selected_case_ade72": {
            "previous": previous_case_metrics["ade_72h_km"],
            "candidate": candidate_case_metrics["ade_72h_km"],
            "final": final_case_metrics["ade_72h_km"],
        },
        "audit_objective": {
            "previous": previous_audit_metrics["objective"],
            "candidate": candidate_audit_metrics["objective"],
            "improvement_pct": audit_improvement,
        },
        "final_config": final_config,
    }, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
