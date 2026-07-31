"""三轮逐台风预测—相似路径融合参数优化。

每轮使用一场新的数据库台风：
- 前60%路径作为新台风输入；
- 后续72小时只用于计算误差；
- 参数通过通用网格搜索更新，不拟合残差轨迹；
- 固定10场审计集指标变好才接受更新。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import algorithm_lab as lab
import analog_fusion_optimize as fusion


OUTPUT_DIR = Path("output")
SEED = 2037


def build_new_cases(
    records: list[dict], excluded_ids: set[str]
) -> tuple[list[fusion.FusionCase], list[fusion.FusionCase]]:
    eligible = []
    for record in records:
        if record["id"] in excluded_ids or len(record["points"]) < 64:
            continue
        if not 2.0 <= lab._average_step_hours(record["points"]) <= 4.0:
            continue
        cutoff = max(24, int(len(record["points"]) * 0.60))
        if cutoff + fusion.NATIVE_STEPS > len(record["points"]):
            cutoff = len(record["points"]) - fusion.NATIVE_STEPS
        if cutoff < 24:
            continue
        truth_native = lab._path(
            record["points"][cutoff:cutoff + fusion.NATIVE_STEPS]
        )
        truth = truth_native[1::2][:fusion.DISPLAY_STEPS]
        if len(truth) != fusion.DISPLAY_STEPS:
            continue
        eligible.append(fusion.FusionCase(
            record["id"], record["name"], record["year"],
            record["points"][:cutoff], truth,
        ))
    rng = np.random.default_rng(SEED)
    selected = [
        eligible[int(index)]
        for index in rng.permutation(len(eligible))[:13]
    ]
    return selected[:3], selected[3:]


def candidate_grid(initial: dict) -> list[dict]:
    values = []
    for base_model in (
        "current_aligned", "inertia6",
        "inertia24_decay102", "current_inertia50",
    ):
        for top_k in (3, 5, 8):
            for future_weight in (0.35, 0.50, 0.65):
                for alpha_short, alpha_long in (
                    (0.15, 0.35), (0.25, 0.50),
                    (0.35, 0.65), (0.45, 0.75),
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
    if initial not in values:
        values.append(initial)
    return values


def write_known(case: fusion.FusionCase, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "lat", "lon", "pressure", "wind_speed"])
        for point in case.known:
            writer.writerow([
                point.time, point.lat, point.lon,
                point.pressure, point.wind_speed,
            ])


def run() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    started = time.perf_counter()
    records = lab.clean_records(lab.SQLiteReader().load())
    records_by_id = {record["id"]: record for record in records}
    previous_report_path = OUTPUT_DIR / "analog_fusion_optimization_report.json"
    excluded = set()
    if previous_report_path.exists():
        previous_report = json.loads(
            previous_report_path.read_text(encoding="utf-8")
        )
        excluded.update(previous_report.get("tuning_case_ids", []))
        excluded.update(previous_report.get("holdout_case_ids", []))

    development_cases, audit_cases = build_new_cases(records, excluded)
    all_cases = [*development_cases, *audit_cases]
    algorithm = lab.DirectedOptimizedAlgorithm(records)
    bases, pools = {}, {}
    for index, case in enumerate(all_cases, start=1):
        bases[case.typhoon_id] = fusion.base_predictions(case, algorithm)
        pools[case.typhoon_id] = fusion.build_analog_pool(case, records)
        print(
            f"预计算 {index:02d}/{len(all_cases)} "
            f"{case.name} ({case.year})，候选{len(pools[case.typhoon_id])}条"
        )

    config_path = OUTPUT_DIR / "best_analog_fusion_config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    current_config = dict(saved_config["config"])
    initial_config = dict(current_config)
    current_audit_metrics, _ = fusion.evaluate_config(
        audit_cases, bases, pools, current_config
    )
    initial_audit_metrics = dict(current_audit_metrics)
    grid = candidate_grid(initial_config)
    rounds = []

    for round_index, development_case in enumerate(development_cases, start=1):
        ranked = []
        for config in grid:
            metrics, _ = fusion.evaluate_config(
                [development_case], bases, pools, config
            )
            ranked.append({"config": config, "development_metrics": metrics})
        ranked.sort(key=lambda item: item["development_metrics"]["objective"])
        candidate = ranked[0]
        candidate_config = candidate["config"]
        candidate_audit_metrics, _ = fusion.evaluate_config(
            audit_cases, bases, pools, candidate_config
        )

        before_metrics, _ = fusion.evaluate_config(
            [development_case], bases, pools, current_config
        )
        candidate_case_metrics, candidate_case_outputs = fusion.evaluate_config(
            [development_case], bases, pools, candidate_config
        )
        accepted = (
            candidate_audit_metrics["objective"]
            < current_audit_metrics["objective"]
        )
        previous_config = dict(current_config)
        previous_audit = dict(current_audit_metrics)
        if accepted:
            current_config = dict(candidate_config)
            current_audit_metrics = dict(candidate_audit_metrics)

        accepted_case_metrics, accepted_outputs = fusion.evaluate_config(
            [development_case], bases, pools, current_config
        )
        # 地图展示本轮单样本搜索得到的候选预测；是否接入由审计集另行决定。
        output = candidate_case_outputs[0]
        write_known(
            development_case,
            OUTPUT_DIR / f"iteration_round{round_index}_new_typhoon.csv",
        )
        fusion.create_map(
            development_case,
            output["prediction"],
            output["ranked"],
            records_by_id,
            OUTPUT_DIR / f"iteration_round{round_index}_map.html",
            (
                f"第{round_index}轮：{development_case.name}候选预测"
                f"（审计{'接受' if accepted else '拒绝'}）"
            ),
        )
        rounds.append({
            "round": round_index,
            "development_case": {
                "id": development_case.typhoon_id,
                "name": development_case.name,
                "year": development_case.year,
                "known_points": len(development_case.known),
            },
            "previous_config": previous_config,
            "candidate_config": candidate_config,
            "accepted_config": dict(current_config),
            "candidate_accepted": accepted,
            "case_before_metrics": before_metrics,
            "case_candidate_metrics": candidate_case_metrics,
            "case_after_acceptance_metrics": accepted_case_metrics,
            "audit_before_metrics": previous_audit,
            "audit_candidate_metrics": candidate_audit_metrics,
            "audit_after_metrics": dict(current_audit_metrics),
            "top10_after_acceptance": fusion.serializable_rank(
                accepted_outputs[0]["ranked"]
            ),
            "candidate_top10": fusion.serializable_rank(output["ranked"]),
            "candidate_search_top10": ranked[:10],
        })
        print(
            f"第{round_index}轮 {development_case.name}: "
            f"候选{'接受' if accepted else '拒绝'}，"
            f"审计目标 {previous_audit['objective']:.2f}"
            f" -> {current_audit_metrics['objective']:.2f}"
        )

    final_improvement = (
        (initial_audit_metrics["objective"] - current_audit_metrics["objective"])
        / max(initial_audit_metrics["objective"], 1e-12) * 100.0
    )
    final_accepted = current_audit_metrics["objective"] < initial_audit_metrics["objective"]
    if final_accepted:
        config_path.write_text(
            json.dumps({
                "accepted": True,
                "config": current_config,
                "validation": (
                    "三轮新台风顺序开发；每轮由固定10台风审计集决定是否接受"
                ),
                "truth_fitting": False,
                "previous_config": initial_config,
                "audit_objective_improvement_pct": final_improvement,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available_fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(0, 4)
    audit_objectives = [
        initial_audit_metrics["objective"],
        *[item["audit_after_metrics"]["objective"] for item in rounds],
    ]
    axes[0].plot(x, audit_objectives, marker="o", color="#2563EB")
    axes[0].set_xticks(x, ["初始", "第1轮", "第2轮", "第3轮"])
    axes[0].set_ylabel("固定审计集综合目标（越低越好）")
    axes[0].set_title("三轮参数接受轨迹")
    axes[0].grid(alpha=0.25)

    labels = [
        f"{item['development_case']['name']}\n第{item['round']}轮"
        for item in rounds
    ]
    before = [
        item["case_before_metrics"]["ade_72h_km"] for item in rounds
    ]
    after = [
        item["case_candidate_metrics"]["ade_72h_km"]
        for item in rounds
    ]
    positions = np.arange(3)
    axes[1].bar(positions - 0.18, before, 0.36, label="优化前")
    axes[1].bar(positions + 0.18, after, 0.36, label="单样本最优候选")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("72h ADE（km）")
    axes[1].set_title("三场截断新台风预测对比")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "three_round_iteration_comparison.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "method": "three_round_sequential_parameter_evolution",
        "seed": SEED,
        "development_cases": 3,
        "fixed_audit_cases": 10,
        "excluded_previous_experiment_cases": sorted(excluded),
        "initial_config": initial_config,
        "initial_audit_metrics": initial_audit_metrics,
        "rounds": rounds,
        "final_config": current_config,
        "final_audit_metrics": current_audit_metrics,
        "final_config_updated": final_accepted,
        "audit_objective_improvement_pct": final_improvement,
        "leakage_audit": {
            "target_typhoon_excluded_from_candidate_pool": True,
            "future_truth_used_for_candidate_ranking": False,
            "future_truth_used_for_path_fusion": False,
            "residual_path_fitting_enabled": False,
            "parameter_updates_are_generic_grid_values": True,
            "fixed_audit_set_used_for_accept_reject_only": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "round_csv_pattern": "output/iteration_round{1..3}_new_typhoon.csv",
            "round_map_pattern": "output/iteration_round{1..3}_map.html",
            "comparison_figure": "output/three_round_iteration_comparison.png",
            "best_config": "output/best_analog_fusion_config.json",
        },
    }
    (OUTPUT_DIR / "three_round_iteration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "initial_config": report["initial_config"],
        "rounds": [
            {
                "round": item["round"],
                "case": item["development_case"],
                "accepted": item["candidate_accepted"],
                "case_ade72_before": item[
                    "case_before_metrics"
                ]["ade_72h_km"],
                "case_ade72_after": item[
                    "case_after_acceptance_metrics"
                ]["ade_72h_km"],
                "audit_objective_after": item[
                    "audit_after_metrics"
                ]["objective"],
            }
            for item in report["rounds"]
        ],
        "final_config": report["final_config"],
        "final_audit_metrics": report["final_audit_metrics"],
        "audit_objective_improvement_pct": report[
            "audit_objective_improvement_pct"
        ],
    }, ensure_ascii=False, indent=2))
