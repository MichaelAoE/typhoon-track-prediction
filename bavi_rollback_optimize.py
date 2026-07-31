"""Bavi 严格无泄漏滚动优化与最终留出检验。

运行：
    python bavi_rollback_optimize.py

约束：
1. 最终断点后的真实路径不参与预测、参数选择、残差修正或相似台风排序。
2. 所有预测超参数只使用最终断点至少 72 小时以前的滚动断点选择。
3. 预测和检索配置冻结后，才读取最终 72 小时真值并计算误差。
4. 输出只包含“模型预测 vs 真实路径”，不生成任何真值拟合路径。
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


OUTPUT_DIR = Path("output")
FORECAST_HOURS = 72
MODEL_STEP_HOURS = 3
DISPLAY_STEP_HOURS = 6
MODEL_STEPS = FORECAST_HOURS // MODEL_STEP_HOURS
DISPLAY_STEPS = FORECAST_HOURS // DISPLAY_STEP_HOURS


@dataclass
class RawTrack:
    times: list[datetime]
    path: np.ndarray  # [[lon, lat], ...]

    def at(self, when: datetime) -> np.ndarray:
        source = np.asarray([item.timestamp() for item in self.times])
        target = when.timestamp()
        return np.asarray([
            np.interp(target, source, self.path[:, 0]),
            np.interp(target, source, self.path[:, 1]),
        ])

    def grid(self, start: datetime, end: datetime, hours=3) -> list[lab.TrackPoint]:
        points = []
        current = start
        while current <= end:
            lon, lat = self.at(current)
            points.append(lab.TrackPoint(
                current.replace(tzinfo=timezone.utc).isoformat(),
                float(lat), float(lon), 1013.0, 0.0,
            ))
            current += timedelta(hours=hours)
        return points


def load_bavi(path="bavi.csv") -> RawTrack:
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for values in reader:
            try:
                when = datetime.fromisoformat(values[0].strip())
                lat, lon = float(values[1]), float(values[2])
            except (IndexError, TypeError, ValueError):
                continue
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                rows.append((when, lon, lat))
    rows = sorted({item[0]: item for item in rows}.values())
    if len(rows) < 48:
        raise ValueError("bavi.csv 有效路径点不足，无法执行严格72小时回测")
    return RawTrack(
        [item[0] for item in rows],
        np.asarray([[item[1], item[2]] for item in rows], dtype=float),
    )


def choose_strict_cutoffs(track: RawTrack) -> tuple[list[datetime], datetime]:
    """最终断点留出；校准段的完整真值必须在最终断点前已经可知。"""
    latest = track.times[-1] - timedelta(hours=FORECAST_HOURS)
    test_cutoff = max(item for item in track.times if item <= latest)
    # 用12小时间隔增加滚动样本；最后一个校准段的完整72h真值恰好
    # 结束于最终预报时刻，因此没有使用最终预报发布后的信息。
    calibration = [
        test_cutoff - timedelta(hours=hours)
        for hours in (126, 114, 102, 90, 78, 72)
    ]
    earliest_allowed = track.times[0] + timedelta(hours=69)
    calibration = [
        item for item in calibration
        if item >= earliest_allowed
        and item + timedelta(hours=FORECAST_HOURS) <= test_cutoff
    ]
    if len(calibration) < 2:
        raise ValueError("严格时间隔离后可用校准断点不足")
    return calibration, test_cutoff


def truth_at_6h(track: RawTrack, cutoff: datetime) -> np.ndarray:
    return np.asarray([
        track.at(cutoff + timedelta(hours=index * DISPLAY_STEP_HOURS))
        for index in range(1, DISPLAY_STEPS + 1)
    ])


def to_display(native_prediction: np.ndarray) -> np.ndarray:
    return np.asarray(native_prediction, dtype=float)[1::2][:DISPLAY_STEPS]


def errors_km(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.asarray([
        lab.haversine(*predicted, *actual)
        for predicted, actual in zip(prediction, truth)
    ])


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    errors = errors_km(prediction, truth)
    return {
        "ade_24h_km": float(np.mean(errors[:4])),
        "fde_24h_km": float(errors[3]),
        "ade_48h_km": float(np.mean(errors[:8])),
        "fde_48h_km": float(errors[7]),
        "ade_72h_km": float(np.mean(errors)),
        "fde_72h_km": float(errors[-1]),
        "objective": float(np.mean(errors) + 0.5 * errors[-1]),
        "errors_km": errors.tolist(),
    }


def mean_metrics(values: list[dict]) -> dict:
    keys = (
        "ade_24h_km", "fde_24h_km", "ade_48h_km",
        "fde_48h_km", "ade_72h_km", "fde_72h_km", "objective",
    )
    return {key: float(np.mean([item[key] for item in values])) for key in keys}


def inertia_native(
    known: list[lab.TrackPoint], window: int, decay: float
) -> np.ndarray:
    path = lab._path(known)
    count = min(max(1, window), len(path) - 1)
    differences = np.diff(path[-count - 1:], axis=0)
    recent_weights = np.exp(np.linspace(-1.2, 0.0, len(differences)))
    step = np.average(differences, axis=0, weights=recent_weights)
    output, current = [], path[-1].copy()
    for _ in range(MODEL_STEPS):
        current = current + step
        output.append(current.copy())
        step *= decay
    return np.asarray(output)


def polynomial_native(
    known: list[lab.TrackPoint], window: int, degree: int
) -> np.ndarray:
    path = lab._path(known[-window:])
    source = np.arange(-len(path) + 1, 1, dtype=float) * MODEL_STEP_HOURS
    future = np.arange(1, MODEL_STEPS + 1, dtype=float) * MODEL_STEP_HOURS
    return np.column_stack([
        np.polyval(np.polyfit(source, path[:, axis], degree), future)
        for axis in range(2)
    ])


def turning_motion_native(
    known: list[lab.TrackPoint],
    window: int,
    turn_decay: float,
    speed_decay: float,
) -> np.ndarray:
    """从近期移向角和移速趋势外推，参数只由滚动校准段选择。"""
    path = lab._path(known[-window:])
    steps = np.diff(path, axis=0)
    speeds = np.linalg.norm(steps, axis=1)
    headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    recent_count = min(6, len(steps))
    recent_weights = np.exp(np.linspace(-1.4, 0.0, recent_count))
    speed = float(np.average(speeds[-recent_count:], weights=recent_weights))
    heading = float(np.average(
        headings[-recent_count:], weights=recent_weights
    ))
    if len(headings) >= 3:
        turn_rate = float(np.median(np.diff(headings[-recent_count:])))
    else:
        turn_rate = 0.0
    turn_rate = float(np.clip(turn_rate, -math.radians(12), math.radians(12)))
    current = path[-1].copy()
    output = []
    for _ in range(MODEL_STEPS):
        heading += turn_rate
        current = current + speed * np.asarray([
            math.cos(heading), math.sin(heading)
        ])
        output.append(current.copy())
        turn_rate *= turn_decay
        speed *= speed_decay
    return np.asarray(output)


def acceleration_native(
    known: list[lab.TrackPoint],
    window: int,
    acceleration_decay: float,
) -> np.ndarray:
    """二维速度—加速度外推，限制加速度防止长时效瞬移。"""
    path = lab._path(known[-window:])
    velocity = np.diff(path, axis=0)
    recent = velocity[-min(6, len(velocity)):]
    current_velocity = np.average(
        recent, axis=0, weights=np.exp(np.linspace(-1.2, 0.0, len(recent)))
    )
    if len(recent) >= 2:
        acceleration = np.median(np.diff(recent, axis=0), axis=0)
    else:
        acceleration = np.zeros(2)
    acceleration_norm = np.linalg.norm(acceleration)
    speed = max(np.linalg.norm(current_velocity), 1e-6)
    if acceleration_norm > 0.25 * speed:
        acceleration *= 0.25 * speed / acceleration_norm
    current = path[-1].copy()
    output = []
    for _ in range(MODEL_STEPS):
        current_velocity = current_velocity + acceleration
        current = current + current_velocity
        output.append(current.copy())
        acceleration *= acceleration_decay
    return np.asarray(output)


def model_input(
    known: list[lab.TrackPoint],
    normalizer: lab.StrengthGroupedNormalizer,
    group: int,
    recent_bias: float,
) -> np.ndarray:
    window = list(known[-24:])
    if len(window) < 24:
        window = [window[0]] * (24 - len(window)) + window
    coordinates = lab._path(window)
    tensor = np.hstack([
        (coordinates - coordinates[-1]) / 10.0,
        normalizer.transform(window, group),
    ]).astype(np.float32)
    if recent_bias:
        tensor[-6:] *= 1.0 + recent_bias
    return tensor


def direct_model_native(
    model,
    known: list[lab.TrackPoint],
    normalizer: lab.StrengthGroupedNormalizer,
    group: int,
    recent_bias: float,
) -> np.ndarray:
    anchor = np.asarray([known[-1].lon, known[-1].lat])
    return (
        model(
            model_input(known, normalizer, group, recent_bias)[None, ...],
            training=False,
        ).numpy()[0] * 10.0 + anchor
    )


def iterative_model_native(
    model,
    known: list[lab.TrackPoint],
    normalizer: lab.StrengthGroupedNormalizer,
    group: int,
) -> np.ndarray:
    history, output = list(known), []
    while len(output) < MODEL_STEPS:
        block = direct_model_native(model, history, normalizer, group, 0.0)
        for lon, lat in block:
            previous = datetime.fromisoformat(
                history[-1].time.replace("Z", "+00:00")
            )
            history.append(lab.TrackPoint(
                (previous + timedelta(hours=MODEL_STEP_HOURS)).isoformat(),
                float(lat), float(lon), 1013.0, 0.0,
            ))
            output.append([float(lon), float(lat)])
            if len(output) >= MODEL_STEPS:
                break
    return np.asarray(output)


def analog_rank(
    known: list[lab.TrackPoint], records: list[dict]
) -> list[tuple[float, np.ndarray]]:
    query_points = known[-24:]
    query = lab._path(query_points)
    case = lab.BenchmarkCase("BAVI_CALIBRATION", "Bavi", 0, query_points, [])
    ranked = []
    for record in lab._coarse_candidates(case, records, limit=400):
        path = lab._path(record["points"])
        distance, start = lab.sliding_window_overlap(query, path, stride=2)
        anchor = start + len(query)
        if start < 0 or anchor + MODEL_STEPS > len(path):
            continue
        matched_step = np.mean(np.diff(path[start:anchor], axis=0), axis=0)
        query_step = np.mean(np.diff(query[-8:], axis=0), axis=0)
        scale = np.linalg.norm(query_step) / max(np.linalg.norm(matched_step), 1e-6)
        scale = float(np.clip(scale, 0.5, 1.8))
        displacement = (path[anchor:anchor + MODEL_STEPS] - path[anchor - 1]) * scale
        ranked.append((distance, displacement))
    return sorted(ranked, key=lambda item: item[0])


def analog_native(
    known: list[lab.TrackPoint],
    ranked_analogs: list[tuple[float, np.ndarray]],
    top_k: int,
) -> np.ndarray:
    selected = ranked_analogs[:top_k]
    if not selected:
        return inertia_native(known, 6, 0.98)
    weights = np.asarray([
        math.exp(-distance / 250.0) for distance, _ in selected
    ])
    displacement = np.average(
        np.stack([item[1] for item in selected]), axis=0, weights=weights
    )
    anchor = np.asarray([known[-1].lon, known[-1].lat])
    return anchor + displacement


def build_base_predictions(
    known: list[lab.TrackPoint],
    records: list[dict],
    normalizer: lab.StrengthGroupedNormalizer,
    group: int,
    models: dict,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    predictions, metadata = {}, {}

    predictions["score61_restored"] = lab._score61_legacy_ensemble_prediction(
        known, normalizer, group
    )
    metadata["score61_restored"] = {
        "family": "legacy_gru_tcn_inertia",
        "weights": [0.45, 0.30, 0.25],
        "source": "固定30台风约60.13分版本",
    }

    legacy_gru = to_display(iterative_model_native(
        models["legacy_gru"], known, normalizer, group
    ))
    legacy_tcn = to_display(iterative_model_native(
        models["legacy_tcn"], known, normalizer, group
    ))
    inertia_default = to_display(inertia_native(known, 6, 1.0))
    predictions["legacy_time_aligned"] = (
        0.45 * legacy_gru + 0.30 * legacy_tcn + 0.25 * inertia_default
    )
    metadata["legacy_time_aligned"] = {
        "family": "legacy_time_aligned_ensemble",
        "weights": [0.45, 0.30, 0.25],
    }

    current_gru = to_display(direct_model_native(
        models["current_gru"], known, normalizer, group, 0.5
    )[:MODEL_STEPS])
    current_tcn = to_display(direct_model_native(
        models["current_tcn"], known, normalizer, group, 0.5
    )[:MODEL_STEPS])
    predictions["current_time_aligned"] = (
        0.45 * current_gru + 0.30 * current_tcn + 0.25 * inertia_default
    )
    metadata["current_time_aligned"] = {
        "family": "current_time_aligned_ensemble",
        "weights": [0.45, 0.30, 0.25],
    }

    for window in (3, 4, 5, 6, 8, 12, 18, 24):
        for decay in (0.92, 0.96, 1.00, 1.02):
            key = f"inertia_w{window}_d{decay:.2f}"
            predictions[key] = to_display(inertia_native(known, window, decay))
            metadata[key] = {
                "family": "recent_motion_inertia",
                "window": window,
                "decay": decay,
            }
    for window in (6, 9, 12, 18, 24):
        for degree in (1, 2):
            key = f"polynomial_w{window}_p{degree}"
            predictions[key] = to_display(
                polynomial_native(known, window, degree)
            )
            metadata[key] = {
                "family": "polynomial",
                "window": window,
                "degree": degree,
            }
    for window in (6, 12, 18, 24):
        for turn_decay in (0.70, 0.85, 1.00):
            for speed_decay in (0.98, 1.00):
                key = (
                    f"turn_w{window}_td{turn_decay:.2f}"
                    f"_sd{speed_decay:.2f}"
                )
                predictions[key] = to_display(turning_motion_native(
                    known, window, turn_decay, speed_decay
                ))
                metadata[key] = {
                    "family": "turning_motion",
                    "window": window,
                    "turn_decay": turn_decay,
                    "speed_decay": speed_decay,
                }
    for window in (6, 12, 18, 24):
        for acceleration_decay in (0.70, 0.85, 1.00):
            key = f"acceleration_w{window}_d{acceleration_decay:.2f}"
            predictions[key] = to_display(acceleration_native(
                known, window, acceleration_decay
            ))
            metadata[key] = {
                "family": "vector_acceleration",
                "window": window,
                "acceleration_decay": acceleration_decay,
            }

    analogs = analog_rank(known, records)
    for top_k in (3, 5, 10):
        key = f"analog_top{top_k}"
        predictions[key] = to_display(analog_native(known, analogs, top_k))
        metadata[key] = {
            "family": "historical_analog_ensemble",
            "top_k": top_k,
        }
    return predictions, metadata


def add_validation_blends(
    calibration_predictions: dict[datetime, dict[str, np.ndarray]],
    calibration_truth: dict[datetime, np.ndarray],
    metadata: dict[str, dict],
) -> tuple[list[dict], dict[datetime, dict[str, np.ndarray]]]:
    candidates = []
    for key in metadata:
        fold_metrics = [
            prediction_metrics(calibration_predictions[cutoff][key], truth)
            for cutoff, truth in calibration_truth.items()
        ]
        objectives = [item["objective"] for item in fold_metrics]
        candidates.append({
            "key": key,
            "config": metadata[key],
            "validation_metrics": mean_metrics(fold_metrics),
            "validation_std": float(np.std(objectives)),
            "selection_score": float(np.mean(objectives) + 0.25 * np.std(objectives)),
        })
    top_keys = [
        item["key"] for item in sorted(
            candidates, key=lambda item: item["selection_score"]
        )[:12]
    ]
    for left_index, left in enumerate(top_keys):
        for right in top_keys[left_index + 1:]:
            for weight in (0.25, 0.50, 0.75):
                key = f"blend__{left}__{right}__{weight:.2f}"
                for cutoff in calibration_predictions:
                    values = calibration_predictions[cutoff]
                    values[key] = weight * values[left] + (1.0 - weight) * values[right]
                fold_metrics = [
                    prediction_metrics(calibration_predictions[cutoff][key], truth)
                    for cutoff, truth in calibration_truth.items()
                ]
                objectives = [item["objective"] for item in fold_metrics]
                candidates.append({
                    "key": key,
                    "config": {
                        "family": "validation_blend",
                        "left": left,
                        "right": right,
                        "left_weight": weight,
                    },
                    "validation_metrics": mean_metrics(fold_metrics),
                    "validation_std": float(np.std(objectives)),
                    "selection_score": float(
                        np.mean(objectives) + 0.25 * np.std(objectives)
                    ),
                })
    return sorted(candidates, key=lambda item: item["selection_score"]), calibration_predictions


def prediction_from_config(
    config: dict,
    base_predictions: dict[str, np.ndarray],
    selected_key: str,
) -> np.ndarray:
    if selected_key in base_predictions:
        return base_predictions[selected_key]
    if config["family"] == "validation_blend":
        return (
            config["left_weight"] * base_predictions[config["left"]]
            + (1.0 - config["left_weight"]) * base_predictions[config["right"]]
        )
    raise KeyError(f"无法复现预测配置：{selected_key}")


def candidate_future(
    record: dict, anchor: int, count: int
) -> np.ndarray:
    points = record["points"]
    local = points[max(0, anchor - 4):min(len(points), anchor + 8)]
    step_hours = lab._average_step_hours(local) if len(local) >= 2 else 3.0
    stride = max(1, int(round(DISPLAY_STEP_HOURS / max(step_hours, 1e-6))))
    first = anchor + stride - 1
    path = lab._path(points)
    values = path[first:first + stride * count:stride]
    if len(values) == 0:
        values = path[-1:]
    if len(values) < count:
        values = np.vstack([
            values,
            np.repeat(values[-1][None, :], count - len(values), axis=0),
        ])
    return values[:count]


def retrieval_quality(
    result: dict, records_by_id: dict[str, dict], truth: np.ndarray
) -> dict:
    errors, scores = [], []
    for item in result["top10"]:
        record = records_by_id[item["id"]]
        continuation = candidate_future(
            record, int(item.get("forecast_anchor", 0)), len(truth)
        )
        error = float(np.mean(errors_km(continuation, truth)))
        errors.append(error)
        scores.append(float(item["similarity"]))
    correlation = 0.0
    if len(errors) >= 2 and np.std(errors) > 0 and np.std(scores) > 0:
        correlation = float(np.corrcoef(scores, -np.asarray(errors))[0, 1])
    top5_ade = float(np.mean(errors[:5])) if errors else float("inf")
    overlap = float(np.mean(np.exp(-np.asarray(errors) / 300.0))) if errors else 0.0
    return {
        "top5_future_ade_km": top5_ade,
        "top10_future_overlap": overlap,
        "rank_correlation": correlation,
        "objective": top5_ade - 150.0 * overlap - 30.0 * correlation,
        "candidate_future_ade_km": errors,
    }


def tune_retrieval(
    records: list[dict],
    full_known: dict[datetime, list[lab.TrackPoint]],
    predictions: dict[datetime, np.ndarray],
    truths: dict[datetime, np.ndarray],
) -> tuple[dict, list[dict]]:
    grid = [
        {
            "name": "score61_exact",
            "lcss_epsilon": 100.0, "heading_limit": 45.0,
            "speed_limit": 6.0, "lcss_blend": 0.0,
            "fusion_weights": (0.65, 0.20, 0.15),
            "forecast_time_weight_mode": "score61",
        },
        {
            "name": "score61_lcss20",
            "lcss_epsilon": 100.0, "heading_limit": 45.0,
            "speed_limit": 6.0, "lcss_blend": 0.2,
            "fusion_weights": (0.65, 0.20, 0.15),
            "forecast_time_weight_mode": "score61",
        },
        {
            "name": "missing_intensity_path90",
            "lcss_epsilon": 100.0, "heading_limit": 45.0,
            "speed_limit": 6.0, "lcss_blend": 0.0,
            "fusion_weights": (0.90, 0.05, 0.05),
            "forecast_time_weight_mode": "score61",
        },
        {
            "name": "missing_intensity_path_only",
            "lcss_epsilon": 100.0, "heading_limit": 45.0,
            "speed_limit": 6.0, "lcss_blend": 0.0,
            "fusion_weights": (1.0, 0.0, 0.0),
            "forecast_time_weight_mode": "score61",
        },
        {
            "name": "path_only_physical_time",
            "lcss_epsilon": 100.0, "heading_limit": 45.0,
            "speed_limit": 6.0, "lcss_blend": 0.0,
            "fusion_weights": (1.0, 0.0, 0.0),
            "forecast_time_weight_mode": "physical",
        },
        {
            "name": "relaxed_path90",
            "lcss_epsilon": 120.0, "heading_limit": 60.0,
            "speed_limit": 8.0, "lcss_blend": 0.0,
            "fusion_weights": (0.90, 0.05, 0.05),
            "forecast_time_weight_mode": "physical",
        },
    ]
    by_id = {record["id"]: record for record in records}
    evaluations = []
    for config in grid:
        algorithm = lab.DirectedOptimizedAlgorithm(
            records,
            lcss_epsilon=config["lcss_epsilon"],
            heading_limit=config["heading_limit"],
            speed_limit=config["speed_limit"],
            lcss_blend=config["lcss_blend"],
            fusion_weights=config["fusion_weights"],
            forecast_time_weight_mode=config["forecast_time_weight_mode"],
        )
        fold_metrics, seconds = [], []
        for cutoff in sorted(truths):
            started = time.perf_counter()
            result = algorithm.predict_new_typhoon(
                full_known[cutoff],
                predict_hours=72,
                prediction_override=predictions[cutoff],
                typhoon_id="BAVI_CALIBRATION",
                name="Bavi校准段",
            )
            seconds.append(time.perf_counter() - started)
            fold_metrics.append(retrieval_quality(result, by_id, truths[cutoff]))
        objectives = [item["objective"] for item in fold_metrics]
        evaluations.append({
            "config": {
                **config,
                "fusion_weights": list(config["fusion_weights"]),
            },
            "validation_metrics": {
                "top5_future_ade_km": float(np.mean([
                    item["top5_future_ade_km"] for item in fold_metrics
                ])),
                "top10_future_overlap": float(np.mean([
                    item["top10_future_overlap"] for item in fold_metrics
                ])),
                "rank_correlation": float(np.mean([
                    item["rank_correlation"] for item in fold_metrics
                ])),
                "objective": float(np.mean(objectives)),
                "objective_std": float(np.std(objectives)),
                "avg_query_seconds": float(np.mean(seconds)),
            },
            "selection_score": float(
                np.mean(objectives) + 0.25 * np.std(objectives)
            ),
        })
    evaluations.sort(key=lambda item: item["selection_score"])
    return evaluations[0]["config"], evaluations


def write_bavii(points: list[lab.TrackPoint], path=Path("bavii.csv")) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "lat", "lon", "pressure", "wind_speed"])
        for point in points:
            writer.writerow([
                point.time.replace("T", " ").replace("+00:00", ""),
                point.lat, point.lon, point.pressure, point.wind_speed,
            ])


def run() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    track = load_bavi()
    calibration_cutoffs, test_cutoff = choose_strict_cutoffs(track)
    records = lab.clean_records(lab.SQLiteReader().load())
    by_id = {record["id"]: record for record in records}
    store = lab.ScalarFeatureStore(records)
    normalizer = lab.StrengthGroupedNormalizer(records, store)
    wind_means = np.asarray([
        np.mean([point.wind_speed for point in record["points"]])
        for record in records
    ])
    q1, q2 = np.quantile(wind_means, [1 / 3, 2 / 3])
    group = 0 if 0 <= q1 else 1 if 0 <= q2 else 2

    import tensorflow as tf
    models = {
        "legacy_gru": tf.keras.models.load_model(
            OUTPUT_DIR / "gru_attention_seq2seq.keras", compile=False
        ),
        "legacy_tcn": tf.keras.models.load_model(
            OUTPUT_DIR / "tcn_forecaster.keras", compile=False
        ),
        "current_gru": tf.keras.models.load_model(
            OUTPUT_DIR / "gru_attention_seq2seq_168h.keras", compile=False
        ),
        "current_tcn": tf.keras.models.load_model(
            OUTPUT_DIR / "tcn_forecaster_168h.keras", compile=False
        ),
    }

    calibration_predictions = {}
    calibration_truth = {}
    calibration_full_known = {}
    metadata = None
    for cutoff in calibration_cutoffs:
        model_known = track.grid(
            cutoff - timedelta(hours=69), cutoff, MODEL_STEP_HOURS
        )
        values, current_metadata = build_base_predictions(
            model_known, records, normalizer, group, models
        )
        calibration_predictions[cutoff] = values
        calibration_truth[cutoff] = truth_at_6h(track, cutoff)
        calibration_full_known[cutoff] = track.grid(
            track.times[0], cutoff, MODEL_STEP_HOURS
        )
        metadata = current_metadata

    prediction_candidates, calibration_predictions = add_validation_blends(
        calibration_predictions, calibration_truth, metadata or {}
    )
    selected_prediction = prediction_candidates[0]
    selected_key = selected_prediction["key"]
    selected_config = selected_prediction["config"]
    selected_calibration_predictions = {
        cutoff: prediction_from_config(
            selected_config, values, selected_key
        )
        for cutoff, values in calibration_predictions.items()
    }

    selected_retrieval_config, retrieval_candidates = tune_retrieval(
        records,
        calibration_full_known,
        selected_calibration_predictions,
        calibration_truth,
    )

    # 到这里为止只访问了 calibration_cutoffs 的真值。现在冻结全部配置。
    test_model_known = track.grid(
        test_cutoff - timedelta(hours=69), test_cutoff, MODEL_STEP_HOURS
    )
    test_base, _ = build_base_predictions(
        test_model_known, records, normalizer, group, models
    )
    frozen_prediction = prediction_from_config(
        selected_config, test_base, selected_key
    )
    full_known = track.grid(track.times[0], test_cutoff, MODEL_STEP_HOURS)
    retrieval_algorithm = lab.DirectedOptimizedAlgorithm(
        records,
        lcss_epsilon=selected_retrieval_config["lcss_epsilon"],
        heading_limit=selected_retrieval_config["heading_limit"],
        speed_limit=selected_retrieval_config["speed_limit"],
        lcss_blend=selected_retrieval_config["lcss_blend"],
        fusion_weights=tuple(selected_retrieval_config["fusion_weights"]),
        forecast_time_weight_mode=selected_retrieval_config[
            "forecast_time_weight_mode"
        ],
    )
    query_started = time.perf_counter()
    retrieval_result = retrieval_algorithm.predict_new_typhoon(
        full_known,
        predict_hours=72,
        prediction_override=frozen_prediction,
        typhoon_id="bavii",
        name="Bavi新台风",
    )
    query_seconds = time.perf_counter() - query_started
    score61_prediction = test_base["score61_restored"]
    score61_algorithm = lab.DirectedOptimizedAlgorithm(records)
    score61_retrieval_result = score61_algorithm.predict_new_typhoon(
        full_known,
        predict_hours=72,
        prediction_override=score61_prediction,
        typhoon_id="bavii",
        name="Bavi新台风",
    )

    # 预测路径和检索排名已经冻结，此后才读取最终留出段真值。
    test_truth = truth_at_6h(track, test_cutoff)
    strict_metrics = prediction_metrics(frozen_prediction, test_truth)
    score61_metrics = prediction_metrics(score61_prediction, test_truth)
    retrieval_test_metrics = retrieval_quality(
        retrieval_result, by_id, test_truth
    )
    score61_retrieval_test_metrics = retrieval_quality(
        score61_retrieval_result, by_id, test_truth
    )

    write_bavii(full_known)
    last_observed = test_cutoff.replace(tzinfo=timezone.utc)
    with (OUTPUT_DIR / "bavii_strict_prediction_vs_truth.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "forecast_time", "predicted_lon", "predicted_lat",
            "actual_lon", "actual_lat", "error_km",
        ])
        for index, (predicted, actual, error) in enumerate(zip(
            frozen_prediction, test_truth,
            errors_km(frozen_prediction, test_truth),
        ), start=1):
            writer.writerow([
                (last_observed + timedelta(
                    hours=index * DISPLAY_STEP_HOURS
                )).isoformat(),
                predicted[0], predicted[1], actual[0], actual[1], error,
            ])

    report = {
        "method": "strict_temporal_holdout_without_truth_fitting",
        "source": "bavi.csv",
        "selected_cutoff": test_cutoff.isoformat(sep=" "),
        "calibration_cutoffs": [
            item.isoformat(sep=" ") for item in calibration_cutoffs
        ],
        "latest_calibration_truth_time": max(
            calibration_cutoffs
        ).replace(tzinfo=timezone.utc).isoformat(),
        "latest_calibration_truth_end_time": (
            max(calibration_cutoffs) + timedelta(hours=FORECAST_HOURS)
        ).replace(tzinfo=timezone.utc).isoformat(),
        "known_points": len(full_known),
        "selected_prediction_key": selected_key,
        "selected_prediction_config": selected_config,
        "selected_prediction_validation": {
            key: value for key, value in selected_prediction.items()
            if key != "config"
        },
        "selected_retrieval_config": selected_retrieval_config,
        "prediction_candidates": prediction_candidates,
        "retrieval_candidates": retrieval_candidates,
        "score61_holdout_metrics": score61_metrics,
        "score61_retrieval_holdout_metrics": score61_retrieval_test_metrics,
        "strict_selected_holdout_metrics": strict_metrics,
        "strict_retrieval_holdout_metrics": retrieval_test_metrics,
        "query_seconds": query_seconds,
        "prediction": frozen_prediction.tolist(),
        "truth_for_scoring_only": test_truth.tolist(),
        "top10": retrieval_result["top10"],
        "leakage_audit": {
            "test_truth_used_for_prediction": False,
            "test_truth_used_for_parameter_selection": False,
            "test_truth_used_for_retrieval_ranking": False,
            "prediction_frozen_before_test_truth_read": True,
            "retrieval_ranking_frozen_before_test_truth_read": True,
            "retrospective_residual_fit_enabled": False,
            "bavi_truth_calibration_enabled": False,
        },
    }
    (OUTPUT_DIR / "bavi_strict_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "bavii_strict_similarity.json").write_text(
        json.dumps(retrieval_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_output = {
        "prediction": {
            "key": selected_key,
            "config": selected_config,
            "validation_only": True,
        },
        "retrieval": selected_retrieval_config,
        "test_cutoff": test_cutoff.isoformat(sep=" "),
        "truth_fitting": False,
        "global_fallback": "score61_legacy_gru_tcn_inertia",
    }
    (OUTPUT_DIR / "best_bavi_strict_config.json").write_text(
        json.dumps(config_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available_fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    known_path = lab._path(full_known)
    axes[0].plot(
        known_path[:, 0], known_path[:, 1], color="black",
        linewidth=2.5, label="断点前观测",
    )
    axes[0].plot(
        frozen_prediction[:, 0], frozen_prediction[:, 1],
        color="#DC2626", marker="o", linewidth=2.5, label="严格盲测预测",
    )
    axes[0].plot(
        test_truth[:, 0], test_truth[:, 1],
        color="#16A34A", marker="o", linewidth=2.5, label="留出真实路径",
    )
    axes[0].set_title(f"Bavi 72小时严格留出对比（{test_cutoff:%m-%d %H:%M}）")
    axes[0].set_xlabel("经度")
    axes[0].set_ylabel("纬度")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    horizons = np.arange(6, 73, 6)
    axes[1].plot(
        horizons, errors_km(frozen_prediction, test_truth),
        color="#DC2626", marker="o", label="严格模型误差",
    )
    axes[1].plot(
        horizons, errors_km(score61_prediction, test_truth),
        color="#2563EB", marker="o", linestyle="--", label="约61分版本误差",
    )
    axes[1].set_title("各时效路径误差（真值仅用于评分）")
    axes[1].set_xlabel("预报时效（小时）")
    axes[1].set_ylabel("Haversine误差（km）")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "bavi_strict_prediction_compare.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close(figure)

    center = full_known[-1]
    map_object = folium.Map(
        [center.lat, center.lon], zoom_start=5, tiles="OpenStreetMap"
    )
    folium.PolyLine(
        [(point.lat, point.lon) for point in full_known],
        color="black", weight=5, tooltip="断点前真实观测",
    ).add_to(map_object)
    folium.PolyLine(
        [(lat, lon) for lon, lat in frozen_prediction],
        color="red", weight=4, tooltip="严格无泄漏预测",
    ).add_to(map_object)
    folium.PolyLine(
        [(lat, lon) for lon, lat in test_truth],
        color="green", weight=4, tooltip="断点后真实路径（仅评分）",
    ).add_to(map_object)
    colors = [
        "blue", "purple", "orange", "darkred", "cadetblue",
        "darkgreen", "pink", "darkblue", "lightred", "beige",
    ]
    for index, item in enumerate(retrieval_result["top10"]):
        record = by_id[item["id"]]
        start = int(item.get("comparison_start", 0))
        end = min(
            len(record["points"]),
            int(item.get("forecast_anchor", start)) + DISPLAY_STEPS,
        )
        selected = record["points"][start:end] or record["points"]
        color = colors[index % len(colors)]
        label = (
            f"{index + 1}. {item['name']} ({item['year']}) "
            f"{item['similarity']:.4f}"
        )
        folium.PolyLine(
            [(point.lat, point.lon) for point in selected],
            color=color, weight=1.8, opacity=0.75, tooltip=label,
        ).add_to(map_object)
        for point in selected:
            folium.CircleMarker(
                [point.lat, point.lon], radius=1.1, color=color,
                weight=1, fill=True, fill_opacity=0.6,
            ).add_to(map_object)
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:10px;border:1px solid #777;font-size:12px;'
        'line-height:1.55"><b>Bavi 严格无泄漏验证</b><br>'
        '<span style="color:black">━</span> 断点前观测<br>'
        '<span style="color:red">━</span> 自主模型预测<br>'
        '<span style="color:green">━</span> 留出真值（仅评分）<br>'
        '未生成任何真值拟合路径</div>'
    )
    map_object.get_root().html.add_child(folium.Element(legend))
    map_object.save(OUTPUT_DIR / "bavii_strict_prediction_map.html")

    report["outputs"] = {
        "new_typhoon_csv": "bavii.csv",
        "prediction_vs_truth_csv": "output/bavii_strict_prediction_vs_truth.csv",
        "comparison_figure": "output/bavi_strict_prediction_compare.png",
        "comparison_map": "output/bavii_strict_prediction_map.html",
        "similarity_json": "output/bavii_strict_similarity.json",
        "config": "output/best_bavi_strict_config.json",
    }
    (OUTPUT_DIR / "bavi_strict_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "selected_cutoff": result["selected_cutoff"],
        "selected_prediction_key": result["selected_prediction_key"],
        "score61_holdout_metrics": result["score61_holdout_metrics"],
        "strict_selected_holdout_metrics": result[
            "strict_selected_holdout_metrics"
        ],
        "strict_retrieval_holdout_metrics": result[
            "strict_retrieval_holdout_metrics"
        ],
        "query_seconds": result["query_seconds"],
        "top10": [
            {
                "name": item["name"],
                "year": item["year"],
                "similarity": item["similarity"],
            }
            for item in result["top10"]
        ],
    }, ensure_ascii=False, indent=2))
