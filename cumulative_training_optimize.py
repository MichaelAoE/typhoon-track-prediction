"""第7轮：累积多台风训练与旧参数正则化。

训练阶段只使用：
1. 原始固定20台风开发集；
2. 第4~6轮已经截断过的开发台风；
3. 本轮新选择的一条截断台风。

固定10台风审计集不参与参数搜索，只在候选冻结后做一次验收。
真实未来路径仅用于回测评分，不参与预测生成或相似候选排序。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import algorithm_lab as lab
import analog_fusion_optimize as fusion
import fourth_round_optimize as single_round
import three_round_iterative_optimize as previous_rounds


OUTPUT_DIR = Path("output")
CONFIG_PATH = OUTPUT_DIR / "best_analog_fusion_config.json"
BASE_ROUND_REPORTS = (
    OUTPUT_DIR / "fourth_round_optimization_report.json",
    OUTPUT_DIR / "fifth_round_optimization_report.json",
    OUTPUT_DIR / "round6_optimization_report.json",
)

# 旧开发数据和旧参数合计占90%，本轮单台风只占10%。
TRAINING_WEIGHTS = {
    "original_20_cases": 0.40,
    "previous_round_cases": 0.20,
    "new_case": 0.10,
    "cross_case_p75": 0.15,
    "previous_parameter_prior": 0.15,
}


def case_objective(output: dict) -> float:
    prediction = output["prediction_metrics"]
    retrieval = output["retrieval_metrics"]
    return float(
        prediction["ade_72h_km"]
        + 0.30 * prediction["fde_72h_km"]
        + 0.20 * retrieval["top5_future_ade_km"]
        - 50.0 * retrieval["top10_overlap"]
        - 10.0 * retrieval["rank_correlation"]
    )


def ratio(value: float, baseline: float) -> float:
    return float(value / max(abs(baseline), 1e-9))


def load_previous_round_cases(
    records_by_id: dict[str, dict],
    round_number: int,
) -> tuple[list[fusion.FusionCase], list[dict]]:
    cases: list[fusion.FusionCase] = []
    provenance = []
    report_paths = list(BASE_ROUND_REPORTS)
    for completed_round in range(7, round_number):
        if completed_round == 7:
            report_paths.append(
                OUTPUT_DIR / "cumulative_training_report.json"
            )
        else:
            report_paths.append(
                OUTPUT_DIR
                / f"cumulative_training_round{completed_round}_report.json"
            )
    for report_path in report_paths:
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = (
            report.get("selected_case")
            or report.get("new_case")
            or {}
        )
        typhoon_id = selected.get("id")
        if not typhoon_id or typhoon_id not in records_by_id:
            continue
        case = single_round._case_from_record(records_by_id[typhoon_id])
        if case is None:
            continue
        cases.append(case)
        provenance.append({
            "report": str(report_path),
            "id": case.typhoon_id,
            "name": case.name,
            "year": case.year,
        })
    if not cases:
        raise RuntimeError("未找到第4~6轮累积训练样本。")
    return cases, provenance


def group_evaluation(
    cases: list[fusion.FusionCase],
    bases: dict[str, dict[str, np.ndarray]],
    pools: dict[str, list[dict]],
    config: dict,
) -> tuple[dict, list[dict]]:
    return fusion.evaluate_config(cases, bases, pools, config)


def local_parameter_grid(previous: dict) -> list[dict]:
    """围绕已收敛配置做细粒度搜索，避免重复粗网格。"""
    values = []
    for top_k in (4, 5, 6):
        for future_weight in np.arange(0.40, 0.6001, 0.025):
            for alpha_short in np.arange(0.30, 0.4001, 0.025):
                for alpha_long in np.arange(0.60, 0.7001, 0.025):
                    values.append({
                        "base_model": "current_inertia50",
                        "top_k": top_k,
                        "future_match_weight": round(
                            float(future_weight), 3
                        ),
                        "alpha_short": round(float(alpha_short), 3),
                        "alpha_long": round(float(alpha_long), 3),
                        "motion_align": False,
                    })
    if previous not in values:
        values.append(dict(previous))
    return values


def run(round_number: int = 7, grid_mode: str = "coarse") -> dict:
    """运行指定累积训练轮次；默认第7轮以兼容原入口。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = (
        OUTPUT_DIR / "cumulative_training_report.json"
        if round_number == 7
        else OUTPUT_DIR
        / f"cumulative_training_round{round_number}_report.json"
    )
    started = time.perf_counter()
    records = lab.clean_records(lab.SQLiteReader().load())
    records_by_id = {record["id"]: record for record in records}

    original_tuning, original_holdout = fusion.build_fixed_cases(records)
    analog_report = json.loads(
        (OUTPUT_DIR / "analog_fusion_optimization_report.json").read_text(
            encoding="utf-8"
        )
    )
    original_ids = set(analog_report.get("tuning_case_ids", []))
    original_ids.update(analog_report.get("holdout_case_ids", []))
    old_development, fixed_audit = previous_rounds.build_new_cases(
        records, original_ids
    )
    previous_cases, provenance = load_previous_round_cases(
        records_by_id, round_number
    )

    excluded = set(original_ids)
    excluded.update(case.typhoon_id for case in old_development)
    excluded.update(case.typhoon_id for case in fixed_audit)
    excluded.update(case.typhoon_id for case in previous_cases)
    new_case, selection_features = single_round.choose_new_case(
        records, excluded
    )

    print(
        f"累积训练：原始开发={len(original_tuning)}，"
        f"历史新增={len(previous_cases)}，本轮新增=1，"
        f"审计={len(fixed_audit)}"
    )
    print(
        f"本轮新台风：{new_case.name} ({new_case.year}) "
        f"{new_case.typhoon_id}，已知点数={len(new_case.known)}"
    )

    all_cases = [
        *original_tuning,
        *previous_cases,
        new_case,
        *fixed_audit,
    ]
    unique_cases = {
        case.typhoon_id: case for case in all_cases
    }
    algorithm = lab.DirectedOptimizedAlgorithm(records)
    bases: dict[str, dict[str, np.ndarray]] = {}
    pools: dict[str, list[dict]] = {}
    for index, case in enumerate(unique_cases.values(), start=1):
        bases[case.typhoon_id] = fusion.base_predictions(case, algorithm)
        pools[case.typhoon_id] = fusion.build_analog_pool(case, records)
        print(
            f"预计算 {index:02d}/{len(unique_cases)}：{case.name}，"
            f"候选={len(pools[case.typhoon_id])}"
        )

    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    previous_config = dict(saved["config"])
    if grid_mode == "local":
        grid = local_parameter_grid(previous_config)
    elif grid_mode == "coarse":
        grid = single_round.fine_parameter_grid(previous_config)
    else:
        raise ValueError(f"未知参数网格模式：{grid_mode}")

    previous_original, previous_original_outputs = group_evaluation(
        original_tuning, bases, pools, previous_config
    )
    previous_history, previous_history_outputs = group_evaluation(
        previous_cases, bases, pools, previous_config
    )
    previous_new, previous_new_outputs = group_evaluation(
        [new_case], bases, pools, previous_config
    )
    previous_case_objectives = {
        output["typhoon_id"]: case_objective(output)
        for output in (
            previous_original_outputs
            + previous_history_outputs
            + previous_new_outputs
        )
    }

    search = []
    for index, config in enumerate(grid, start=1):
        original_metrics, original_outputs = group_evaluation(
            original_tuning, bases, pools, config
        )
        history_metrics, history_outputs = group_evaluation(
            previous_cases, bases, pools, config
        )
        new_metrics, new_outputs = group_evaluation(
            [new_case], bases, pools, config
        )
        candidate_outputs = (
            original_outputs + history_outputs + new_outputs
        )
        per_case_ratios = [
            ratio(
                case_objective(output),
                previous_case_objectives[output["typhoon_id"]],
            )
            for output in candidate_outputs
        ]
        p75_ratio = float(np.percentile(per_case_ratios, 75))
        parameter_distance = single_round.parameter_distance(
            config, previous_config
        )
        prior_ratio = 1.0 + 0.10 * parameter_distance
        components = {
            "original_ratio": ratio(
                original_metrics["objective"],
                previous_original["objective"],
            ),
            "history_ratio": ratio(
                history_metrics["objective"],
                previous_history["objective"],
            ),
            "new_case_ratio": ratio(
                new_metrics["objective"],
                previous_new["objective"],
            ),
            "cross_case_p75_ratio": p75_ratio,
            "parameter_prior_ratio": prior_ratio,
        }
        weighted_score = float(
            TRAINING_WEIGHTS["original_20_cases"]
            * components["original_ratio"]
            + TRAINING_WEIGHTS["previous_round_cases"]
            * components["history_ratio"]
            + TRAINING_WEIGHTS["new_case"]
            * components["new_case_ratio"]
            + TRAINING_WEIGHTS["cross_case_p75"]
            * components["cross_case_p75_ratio"]
            + TRAINING_WEIGHTS["previous_parameter_prior"]
            * components["parameter_prior_ratio"]
        )
        search.append({
            "config": config,
            "weighted_score": weighted_score,
            "components": components,
            "parameter_distance": parameter_distance,
            "original_metrics": original_metrics,
            "history_metrics": history_metrics,
            "new_case_metrics": new_metrics,
            "degraded_over_10pct_fraction": float(np.mean(
                np.asarray(per_case_ratios) > 1.10
            )),
        })
        if index % 120 == 0:
            print(f"参数搜索 {index}/{len(grid)}")

    search.sort(key=lambda item: (
        item["weighted_score"],
        item["parameter_distance"],
    ))
    candidate = search[0]
    candidate_config = dict(candidate["config"])

    # 候选冻结后，固定审计集只评估一次。
    previous_audit, _ = group_evaluation(
        fixed_audit, bases, pools, previous_config
    )
    candidate_audit, _ = group_evaluation(
        fixed_audit, bases, pools, candidate_config
    )
    development_pass = (
        candidate["weighted_score"] < 1.0
        and candidate["components"]["original_ratio"] <= 1.01
        and candidate["components"]["cross_case_p75_ratio"] <= 1.02
        and candidate["degraded_over_10pct_fraction"] <= 0.25
    )
    audit_pass = (
        candidate_audit["objective"] < previous_audit["objective"]
        and candidate_audit["ade_72h_km"]
        <= previous_audit["ade_72h_km"] * 1.01
        and candidate_audit["top10_overlap"]
        >= previous_audit["top10_overlap"] * 0.98
    )
    accepted = bool(development_pass and audit_pass)
    final_config = dict(candidate_config if accepted else previous_config)

    previous_new_metrics, previous_new_outputs = group_evaluation(
        [new_case], bases, pools, previous_config
    )
    candidate_new_metrics, candidate_new_outputs = group_evaluation(
        [new_case], bases, pools, candidate_config
    )
    final_new_metrics, final_new_outputs = group_evaluation(
        [new_case], bases, pools, final_config
    )
    previous_output = previous_new_outputs[0]
    candidate_output = candidate_new_outputs[0]
    final_output = final_new_outputs[0]

    prefix = f"iteration_round{round_number}"
    known_csv = OUTPUT_DIR / f"{prefix}_new_typhoon.csv"
    prediction_csv = OUTPUT_DIR / f"{prefix}_prediction_vs_truth.csv"
    map_path = OUTPUT_DIR / f"{prefix}_map.html"
    figure_path = OUTPUT_DIR / f"{prefix}_comparison.png"
    single_round.write_known_csv(new_case, known_csv)
    single_round.write_prediction_csv(
        new_case,
        previous_output["prediction"],
        candidate_output["prediction"],
        prediction_csv,
    )
    fusion.create_map(
        new_case,
        final_output["prediction"],
        final_output["ranked"],
        records_by_id,
        map_path,
        (
            f"第{round_number}轮累积训练："
            f"{new_case.name}截断预测与相似台风"
            f"（候选参数{'已接纳' if accepted else '未接纳'}）"
        ),
    )
    single_round.create_comparison_figure(
        new_case,
        previous_config,
        candidate_config,
        previous_output,
        candidate_output,
        previous_audit,
        candidate_audit,
        figure_path,
    )

    audit_improvement = float(
        (
            previous_audit["objective"]
            - candidate_audit["objective"]
        )
        / max(abs(previous_audit["objective"]), 1e-9)
        * 100.0
    )
    report = {
        "method": "cumulative_multicase_training_with_parameter_prior",
        "training_weights": TRAINING_WEIGHTS,
        "development_case_counts": {
            "original_fixed": len(original_tuning),
            "previous_rounds": len(previous_cases),
            "new_case": 1,
            "total": (
                len(original_tuning) + len(previous_cases) + 1
            ),
        },
        "previous_round_cases": provenance,
        "new_case": {
            "id": new_case.typhoon_id,
            "name": new_case.name,
            "year": new_case.year,
            "known_points": len(new_case.known),
            "known_start": str(new_case.known[0].time),
            "cutoff_time": str(new_case.known[-1].time),
            "forecast_hours": 72,
        },
        "new_case_selection_features_known_only": selection_features,
        "fixed_audit_case_ids": [
            case.typhoon_id for case in fixed_audit
        ],
        "grid_size": len(grid),
        "grid_mode": grid_mode,
        "previous_config": previous_config,
        "candidate": candidate,
        "candidate_config": candidate_config,
        "development_pass": development_pass,
        "fixed_audit_metrics": {
            "previous": previous_audit,
            "candidate": candidate_audit,
            "objective_improvement_pct": audit_improvement,
        },
        "audit_pass": audit_pass,
        "candidate_accepted": accepted,
        "final_config": final_config,
        "new_case_metrics": {
            "previous": previous_new_metrics,
            "candidate": candidate_new_metrics,
            "final": final_new_metrics,
        },
        "candidate_top10": fusion.serializable_rank(
            candidate_output["ranked"]
        ),
        "final_top10": fusion.serializable_rank(final_output["ranked"]),
        "search_top20": search[:20],
        "leakage_audit": {
            "new_case_selected_using_known_segment_only": True,
            "future_truth_used_for_development_scoring": True,
            "future_truth_used_for_prediction_generation": False,
            "future_truth_used_for_candidate_ranking": False,
            "all_target_typhoons_excluded_from_analog_pools": True,
            "fixed_audit_used_once_after_candidate_freeze": True,
            "residual_truth_path_fitting_enabled": False,
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
                    f"cumulative "
                    f"{len(original_tuning) + len(previous_cases) + 1}"
                    "-case development + fixed 10-case audit"
                ),
                "truth_fitting": False,
                "previous_config": previous_config,
                "latest_round": round_number,
                "training_weights": TRAINING_WEIGHTS,
                "audit_objective_improvement_pct": audit_improvement,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({
        "new_case": report["new_case"],
        "development_case_count": report["development_case_counts"]["total"],
        "previous_config": previous_config,
        "candidate_config": candidate_config,
        "weighted_training_score": candidate["weighted_score"],
        "development_pass": development_pass,
        "audit_objective": {
            "previous": previous_audit["objective"],
            "candidate": candidate_audit["objective"],
            "improvement_pct": audit_improvement,
        },
        "audit_pass": audit_pass,
        "candidate_accepted": accepted,
        "final_config": final_config,
    }, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
