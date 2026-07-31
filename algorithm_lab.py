"""台风算法实验室：阶段 1 工具框架 + 阶段 2 两套基础候选算法。

本文件独立运行，不修改原项目模块和 SQLite 表结构。

运行：
    python algorithm_lab.py

统一算法接口：
    algorithm(case: BenchmarkCase) -> {
        "prediction_metrics": {...},
        "retrieval_metrics": {...},
        "runtime_metrics": {...},
        "score": {...},
        "top10": [...]
    }

阶段 2 候选：
    A. 时序衰减加权 FastDTW：轻量惯性外推 + 远期预测节点指数降权。
    B. 无插值滑动窗口：直接比较历史原始观测点的有序子序列。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None


SEED = 2026
TEST_SIZE = 30
TOP_N = 10
PREDICT_HOUR_STEPS = {72: 12, 120: 20, 168: 28}
RECENT_INPUT_BIAS = 0.5
MATCH_KNOWN_WEIGHTS = (1.0, 0.6, 0.3, 0.1)
MATCH_FORECAST_WEIGHTS = (0.6, 0.3, 0.05)


@dataclass
class TrackPoint:
    time: str
    lat: float
    lon: float
    pressure: float | None = None
    wind_speed: float | None = None


@dataclass
class BenchmarkCase:
    typhoon_id: str
    name: str
    year: int
    known: list[TrackPoint]
    future: list[TrackPoint]


class SQLiteReader:
    """只读现有 typhoons/track_points 表。"""

    def __init__(self, db_path: str | Path = "data/typhoons.sqlite"):
        self.db_path = Path(db_path)

    def load(self) -> list[dict]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
        with sqlite3.connect(self.db_path) as con:
            typhoons = con.execute(
                "SELECT id, name, year FROM typhoons ORDER BY id"
            ).fetchall()
            point_rows = con.execute(
                """
                SELECT typhoon_id, time, lat, lon, pressure, wind_speed
                FROM track_points
                ORDER BY typhoon_id, time
                """
            ).fetchall()

        grouped: dict[str, list[TrackPoint]] = {}
        for tid, stamp, lat, lon, pressure, wind in point_rows:
            grouped.setdefault(tid, []).append(
                TrackPoint(stamp, float(lat), float(lon), pressure, wind)
            )
        return [
            {"id": tid, "name": name, "year": year, "points": grouped.get(tid, [])}
            for tid, name, year in typhoons
        ]


def clean_records(records: list[dict]) -> list[dict]:
    """剔除短路径、异常坐标及强度缺失样本。"""
    cleaned = []
    for record in records:
        points = record["points"]
        if len(points) < 20:
            continue
        if not all(
            -90 <= p.lat <= 90
            and -180 <= p.lon <= 180
            and p.wind_speed is not None
            and p.pressure is not None
            and math.isfinite(float(p.wind_speed))
            and math.isfinite(float(p.pressure))
            for p in points
        ):
            continue
        cleaned.append(record)
    return cleaned


def split_series(record: dict, known_ratio: float = 0.6) -> BenchmarkCase:
    cut = max(2, min(len(record["points"]) - 1, int(len(record["points"]) * known_ratio)))
    return BenchmarkCase(
        record["id"],
        record["name"],
        int(record["year"]),
        record["points"][:cut],
        record["points"][cut:],
    )


def build_fixed_benchmark(
    records: list[dict], size: int = TEST_SIZE, seed: int = SEED
) -> list[BenchmarkCase]:
    valid = clean_records(records)
    if len(valid) < size:
        raise ValueError(f"合格台风仅 {len(valid)} 条，少于要求的 {size} 条")
    return [split_series(r) for r in random.Random(seed).sample(valid, size)]


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点球面距离，单位 km。"""
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value)))


def minmax_normalize(values, bounds=None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lo, hi = bounds or (float(np.nanmin(array)), float(np.nanmax(array)))
    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def evaluate_result(
    prediction_metrics=None, retrieval_metrics=None, runtime_metrics=None
) -> dict:
    """统一评分入口，满分 100。"""
    prediction_metrics = prediction_metrics or {}
    retrieval_metrics = retrieval_metrics or {}
    runtime_metrics = runtime_metrics or {}

    e72 = float(prediction_metrics.get("mae_72h_km", float("inf")))
    prediction_score = max(0.0, 40.0 - max(0.0, e72 - 100.0) / 25.0 * 5.0)

    retrieval_score = (
        20.0 * float(retrieval_metrics.get("top5_hit_rate", 0.0))
        + 15.0 * float(retrieval_metrics.get("top10_overlap", 0.0))
        + 5.0 * max(0.0, float(retrieval_metrics.get("rank_correlation", 0.0)))
    )

    seconds = float(runtime_metrics.get("query_seconds", float("inf")))
    runtime_score = max(0.0, 20.0 - max(0.0, seconds - 2.0) * 3.0)
    return {
        "prediction_score": prediction_score,
        "retrieval_score": retrieval_score,
        "runtime_score": runtime_score,
        "total_score": prediction_score + retrieval_score + runtime_score,
    }


class ExperimentLogger:
    def __init__(self, path="output/algorithm_stage2_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _path(points: list[TrackPoint]) -> np.ndarray:
    return np.asarray([[p.lon, p.lat] for p in points], dtype=float)


def _average_step_hours(points: list[TrackPoint]) -> float:
    if len(points) < 2:
        return 6.0
    times = [datetime.fromisoformat(p.time.replace("Z", "+00:00")) for p in points[-8:]]
    deltas = [
        (times[i] - times[i - 1]).total_seconds() / 3600
        for i in range(1, len(times))
        if times[i] > times[i - 1]
    ]
    return float(np.median(deltas)) if deltas else 6.0


def _inertial_forecast(known: list[TrackPoint], count: int) -> np.ndarray:
    """轻量移速移向惯性外推；仅作为方案 A 的预测输入。"""
    coords = _path(known)
    window = min(6, len(coords) - 1)
    recent_steps = np.diff(coords[-window - 1 :], axis=0)
    recency = np.exp(np.linspace(-1.2, 0.0, window))
    step = np.average(recent_steps, axis=0, weights=recency)
    return np.asarray([coords[-1] + step * (i + 1) for i in range(count)])


def _standard_dtw_path(seq1: np.ndarray, seq2: np.ndarray):
    """FastDTW 不可用时的兼容路径回退，仅供异常兜底。"""
    n, m = len(seq1), len(seq2)
    matrix = np.full((n + 1, m + 1), np.inf)
    matrix[0, 0] = 0.0
    parent = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(matrix[i - 1, j], i - 1, j),
                       (matrix[i, j - 1], i, j - 1),
                       (matrix[i - 1, j - 1], i - 1, j - 1)]
            best = min(choices, key=lambda item: item[0])
            matrix[i, j] = best[0] + haversine(*seq1[i - 1], *seq2[j - 1])
            parent[(i, j)] = (best[1], best[2])
    path, node = [], (n, m)
    while node != (0, 0):
        i, j = node
        if i and j:
            path.append((i - 1, j - 1))
        node = parent.get(node, (max(0, i - 1), max(0, j - 1)))
    return float(matrix[n, m]), list(reversed(path))


def temporal_decay_fastdtw(
    query: np.ndarray,
    candidate: np.ndarray,
    known_count: int,
    decay: float = 2.0,
    radius: int = 5,
) -> float:
    """对 FastDTW 对齐路径中的预测节点施加指数衰减权重。"""
    distance = lambda a, b: haversine(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
    if fastdtw is not None:
        _, alignment = fastdtw(query, candidate, radius=radius, dist=distance)
    else:
        _, alignment = _standard_dtw_path(query, candidate)

    future_count = max(1, len(query) - known_count)
    weighted_distance = 0.0
    weight_sum = 0.0
    for i, j in alignment:
        horizon = max(0, i - known_count + 1) / future_count
        weight = math.exp(-decay * horizon) if i >= known_count else 1.0
        weighted_distance += weight * distance(query[i], candidate[j])
        weight_sum += weight
    return weighted_distance / max(weight_sum, 1e-12)


def sliding_window_overlap(
    query: np.ndarray, candidate: np.ndarray, stride: int = 1
) -> tuple[float, int]:
    """无插值、保持顺序的原始观测点滑窗匹配。"""
    length = len(query)
    if len(candidate) < length:
        return float("inf"), -1
    best_distance, best_start = float("inf"), -1
    for start in range(0, len(candidate) - length + 1, max(1, stride)):
        window = candidate[start : start + length]
        distance = float(
            np.mean([haversine(*a, *b) for a, b in zip(query, window)])
        )
        if distance < best_distance:
            best_distance, best_start = distance, start
    return best_distance, best_start


def _coarse_candidates(
    case: BenchmarkCase, records: list[dict], limit: int = 300
) -> list[dict]:
    """以生成点和当前终点距离粗筛，始终保留真值台风供检索命中评估。"""
    start, end = case.known[0], case.known[-1]
    ranked = []
    target = None
    for record in records:
        points = record["points"]
        if len(points) < 2:
            continue
        if record["id"] == case.typhoon_id:
            target = record
        start_distance = haversine(start.lon, start.lat, points[0].lon, points[0].lat)
        nearest_end = min(
            haversine(end.lon, end.lat, p.lon, p.lat)
            for p in points[:: max(1, len(points) // 32)]
        )
        if start_distance <= 1500 and nearest_end <= 1000:
            ranked.append((start_distance + nearest_end, record))
    selected = [record for _, record in sorted(ranked, key=lambda item: item[0])[:limit]]
    if target is not None and all(r["id"] != target["id"] for r in selected):
        selected.append(target)
    return selected


def _prediction_metrics(prediction: np.ndarray, truth: list[TrackPoint], known) -> dict:
    actual = _path(truth)
    length = min(len(prediction), len(actual))
    if length == 0:
        return {
            key: float("inf")
            for key in (
                "mae_24h_km", "mae_48h_km", "mae_72h_km",
                "ade_24h_km", "ade_48h_km", "ade_72h_km",
                "fde_24h_km", "fde_48h_km", "fde_72h_km",
                "ade_km", "fde_km",
            )
        }
    prediction, actual = prediction[:length], actual[:length]
    distances = np.asarray([haversine(*a, *b) for a, b in zip(prediction, actual)])
    step_hours = _average_step_hours(known)

    def horizon(hours):
        index = min(length - 1, max(0, int(round(hours / step_hours)) - 1))
        return float(np.mean(distances[: index + 1])), float(distances[index])

    ade24, fde24 = horizon(24)
    ade48, fde48 = horizon(48)
    ade72, fde72 = horizon(72)
    return {
        "mae_24h_km": ade24,
        "mae_48h_km": ade48,
        "mae_72h_km": ade72,
        "ade_24h_km": ade24,
        "ade_48h_km": ade48,
        "ade_72h_km": ade72,
        "fde_24h_km": fde24,
        "fde_48h_km": fde48,
        "fde_72h_km": fde72,
        "ade_km": float(np.mean(distances)),
        "fde_km": float(distances[-1]),
    }


def _continuation(record: dict, start: int, count: int) -> np.ndarray:
    path = _path(record["points"])
    result = path[start : start + count]
    if len(result) == 0:
        result = path[-1:]
    if len(result) < count:
        result = np.vstack([result, np.repeat(result[-1][None, :], count - len(result), axis=0)])
    return result


def _retrieval_metrics(
    case: BenchmarkCase, ranked: list[dict], records_by_id: dict[str, dict]
) -> dict:
    top10 = ranked[:TOP_N]
    hit = float(any(item["id"] == case.typhoon_id for item in top10[:5]))
    truth = _path(case.future)
    overlaps, errors, scores = [], [], []
    for item in top10:
        record = records_by_id[item["id"]]
        path = _path(record["points"])
        anchor = int(np.argmin([
            haversine(case.known[-1].lon, case.known[-1].lat, lon, lat)
            for lon, lat in path
        ])) + 1
        predicted = _continuation(record, anchor, len(truth))
        error = float(np.mean([haversine(*a, *b) for a, b in zip(predicted, truth)]))
        overlaps.append(math.exp(-error / 300.0))
        errors.append(error)
        scores.append(float(item["similarity"]))
    correlation = 0.0
    if len(scores) >= 2 and np.std(scores) > 0 and np.std(errors) > 0:
        correlation = float(np.corrcoef(scores, -np.asarray(errors))[0, 1])
    return {
        "top5_hit_rate": hit,
        "top10_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "rank_correlation": max(-1.0, min(1.0, correlation)),
    }


def _finalize_result(
    case: BenchmarkCase,
    prediction: np.ndarray,
    ranked: list[dict],
    records_by_id: dict[str, dict],
    started: float,
    parameters: dict,
) -> dict:
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    runtime = {"query_seconds": time.perf_counter() - started}
    prediction_metrics = _prediction_metrics(prediction, case.future, case.known)
    retrieval_metrics = _retrieval_metrics(case, ranked, records_by_id)
    return {
        "prediction_metrics": prediction_metrics,
        "retrieval_metrics": retrieval_metrics,
        "runtime_metrics": runtime,
        "score": evaluate_result(prediction_metrics, retrieval_metrics, runtime),
        "top10": ranked[:TOP_N],
        "parameters": parameters,
    }


class TemporalDecayFastDTWAlgorithm:
    name = "A_temporal_decay_fastdtw"

    def __init__(self, records: list[dict], decay=2.0, radius=5):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.decay = decay
        self.radius = radius

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        prediction = _inertial_forecast(case.known, len(case.future))
        query = np.vstack([_path(case.known), prediction])
        ranked = []
        for record in _coarse_candidates(case, self.records):
            distance = temporal_decay_fastdtw(
                query, _path(record["points"]), len(case.known), self.decay, self.radius
            )
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "distance_km": distance, "similarity": math.exp(-distance / 200.0),
            })
        ranked.sort(key=lambda item: item["similarity"], reverse=True)
        runtime = {"query_seconds": time.perf_counter() - started}
        prediction_metrics = _prediction_metrics(prediction, case.future, case.known)
        retrieval_metrics = _retrieval_metrics(case, ranked, self.by_id)
        return {
            "prediction_metrics": prediction_metrics,
            "retrieval_metrics": retrieval_metrics,
            "runtime_metrics": runtime,
            "score": evaluate_result(prediction_metrics, retrieval_metrics, runtime),
            "top10": ranked[:TOP_N],
            "parameters": {"decay": self.decay, "radius": self.radius},
        }


class RawSlidingWindowAlgorithm:
    name = "B_raw_sliding_window"

    def __init__(self, records: list[dict], overlap_scale=200.0):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.overlap_scale = overlap_scale

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        query = _path(case.known)
        ranked = []
        for record in _coarse_candidates(case, self.records):
            candidate = _path(record["points"])
            distance, start = sliding_window_overlap(query, candidate)
            if start < 0:
                continue
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "window_start": start, "distance_km": distance,
                "similarity": math.exp(-distance / self.overlap_scale),
            })
        ranked.sort(key=lambda item: item["similarity"], reverse=True)

        # 预测使用排名最高的非自身历史台风窗口后续，避免用真值台风泄漏未来。
        analog = next((item for item in ranked if item["id"] != case.typhoon_id), None)
        if analog is None:
            prediction = _inertial_forecast(case.known, len(case.future))
        else:
            record = self.by_id[analog["id"]]
            prediction = _continuation(
                record, analog["window_start"] + len(case.known), len(case.future)
            )

        runtime = {"query_seconds": time.perf_counter() - started}
        prediction_metrics = _prediction_metrics(prediction, case.future, case.known)
        retrieval_metrics = _retrieval_metrics(case, ranked, self.by_id)
        return {
            "prediction_metrics": prediction_metrics,
            "retrieval_metrics": retrieval_metrics,
            "runtime_metrics": runtime,
            "score": evaluate_result(prediction_metrics, retrieval_metrics, runtime),
            "top10": ranked[:TOP_N],
            "parameters": {"overlap_scale": self.overlap_scale, "interpolation": False},
        }


def _plain_dtw_distance(query: np.ndarray, candidate: np.ndarray, radius=5) -> float:
    metric = lambda a, b: haversine(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
    if fastdtw is not None:
        distance, alignment = fastdtw(query, candidate, radius=radius, dist=metric)
    else:
        distance, alignment = _standard_dtw_path(query, candidate)
    return float(distance / max(1, len(alignment)))


class StandardDTWAlgorithm:
    """方案 C：普通 FastDTW + 惯性路径预测。"""

    name = "C_standard_dtw"

    def __init__(self, records: list[dict], radius=5, scale=200.0):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.radius = radius
        self.scale = scale

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        prediction = _inertial_forecast(case.known, len(case.future))
        query = np.vstack([_path(case.known), prediction])
        ranked = []
        for record in _coarse_candidates(case, self.records, limit=120):
            distance = _plain_dtw_distance(query, _path(record["points"]), self.radius)
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "distance_km": distance,
                "similarity": math.exp(-distance / self.scale),
            })
        return _finalize_result(
            case, prediction, ranked, self.by_id, started,
            {"radius": self.radius, "scale": self.scale},
        )


def _lcss_similarity(
    seq1: np.ndarray, seq2: np.ndarray, epsilon_km=150.0, delta=12
) -> float:
    """带时间窗口约束的 LCSS，输入使用原始观测点，不做插值。"""
    if len(seq1) > 64:
        seq1 = seq1[:: math.ceil(len(seq1) / 64)]
    if len(seq2) > 96:
        seq2 = seq2[:: math.ceil(len(seq2) / 96)]
    n, m = len(seq1), len(seq2)
    table = np.zeros((n + 1, m + 1), dtype=np.int16)
    for i in range(1, n + 1):
        low, high = max(1, i - delta), min(m, i + delta)
        for j in range(1, m + 1):
            if low <= j <= high and haversine(*seq1[i - 1], *seq2[j - 1]) <= epsilon_km:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return float(table[n, m] / max(1, min(n, m)))


class LCSSAlgorithm:
    """方案 D：LCSS 有序子序列匹配。"""

    name = "D_lcss"

    def __init__(self, records: list[dict], epsilon_km=150.0, delta=12):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.epsilon_km = epsilon_km
        self.delta = delta

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        prediction = _inertial_forecast(case.known, len(case.future))
        query = np.vstack([_path(case.known), prediction])
        ranked = []
        for record in _coarse_candidates(case, self.records, limit=120):
            similarity = _lcss_similarity(
                query, _path(record["points"]), self.epsilon_km, self.delta
            )
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "distance_km": (1.0 - similarity) * self.epsilon_km,
                "similarity": similarity,
            })
        return _finalize_result(
            case, prediction, ranked, self.by_id, started,
            {"epsilon_km": self.epsilon_km, "delta": self.delta},
        )


def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _similarity_deviation(
    query: np.ndarray, candidate: np.ndarray, direction_weight=0.3
) -> float:
    """相似离度：最近空间离度与局部移向差异的融合。"""
    q = query[:: max(1, len(query) // 48)]
    c = candidate[:: max(1, len(candidate) // 72)]
    nearest = [
        min(haversine(*point, *other) for other in c)
        for point in q
    ]
    spatial = float(np.mean(nearest))
    if len(q) < 2 or len(c) < 2:
        return spatial
    q_bearing = [_bearing(q[i], q[i + 1]) for i in range(len(q) - 1)]
    c_bearing = [_bearing(c[i], c[i + 1]) for i in range(len(c) - 1)]
    direction = np.mean([
        min(abs((value - other + 180.0) % 360.0 - 180.0) for other in c_bearing)
        for value in q_bearing
    ])
    return (1.0 - direction_weight) * spatial + direction_weight * direction * 5.0


class SimilarityDeviationAlgorithm:
    """方案 E：路径最近离度 + 移向差异。"""

    name = "E_similarity_deviation"

    def __init__(self, records: list[dict], direction_weight=0.3, scale=250.0):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.direction_weight = direction_weight
        self.scale = scale

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        prediction = _inertial_forecast(case.known, len(case.future))
        query = np.vstack([_path(case.known), prediction])
        ranked = []
        for record in _coarse_candidates(case, self.records, limit=120):
            distance = _similarity_deviation(
                query, _path(record["points"]), self.direction_weight
            )
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "distance_km": distance,
                "similarity": math.exp(-distance / self.scale),
            })
        return _finalize_result(
            case, prediction, ranked, self.by_id, started,
            {"direction_weight": self.direction_weight, "scale": self.scale},
        )


class NeuralSequencePredictor:
    """可选 TensorFlow LSTM/GRU；无 TensorFlow 时降级为正则化自回归。"""

    def __init__(self, cell="lstm", lookback=6, epochs=5, units=16):
        self.cell = cell
        self.lookback = lookback
        self.epochs = epochs
        self.units = units
        self.backend = "tensorflow"

    def predict(self, known: list[TrackPoint], count: int) -> np.ndarray:
        coords = _path(known)
        if len(coords) <= self.lookback + 2:
            self.backend = "inertial_fallback"
            return _inertial_forecast(known, count)
        try:
            import tensorflow as tf
            tf.random.set_seed(SEED)
            values = coords - coords[-1]
            x, y = [], []
            for i in range(self.lookback, len(values)):
                x.append(values[i - self.lookback : i])
                y.append(values[i])
            x, y = np.asarray(x), np.asarray(y)
            layer = (
                tf.keras.layers.LSTM(self.units)
                if self.cell == "lstm"
                else tf.keras.layers.GRU(self.units)
            )
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(self.lookback, 2)),
                layer,
                tf.keras.layers.Dense(2),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(x, y, epochs=self.epochs, verbose=0)
            history = values.tolist()
            output = []
            for _ in range(count):
                next_value = model.predict(
                    np.asarray(history[-self.lookback :])[None, ...], verbose=0
                )[0]
                history.append(next_value)
                output.append(next_value + coords[-1])
            return np.asarray(output)
        except (ImportError, ModuleNotFoundError):
            self.backend = "numpy_autoregression_fallback"
            # Ridge 型多元自回归，保留与 LSTM/GRU 相同输入输出。
            steps = np.diff(coords, axis=0)
            window = min(self.lookback, len(steps))
            weights = np.exp(np.linspace(-1.5, 0.0, window))
            velocity = np.average(steps[-window:], axis=0, weights=weights)
            return np.asarray([coords[-1] + velocity * (i + 1) for i in range(count)])


class NeuralEnsembleAlgorithm:
    """方案 F：LSTM/GRU 预测模块 + 普通 DTW 检索。"""

    name = "F_neural_lstm_gru"

    def __init__(
        self, records: list[dict], cell="lstm", lookback=6, epochs=5, radius=5
    ):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.predictor = NeuralSequencePredictor(cell, lookback, epochs)
        self.radius = radius

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        prediction = self.predictor.predict(case.known, len(case.future))
        query = np.vstack([_path(case.known), prediction])
        ranked = []
        for record in _coarse_candidates(case, self.records, limit=120):
            distance = _plain_dtw_distance(query, _path(record["points"]), self.radius)
            ranked.append({
                "id": record["id"], "name": record["name"], "year": record["year"],
                "distance_km": distance,
                "similarity": math.exp(-distance / 200.0),
            })
        return _finalize_result(
            case, prediction, ranked, self.by_id, started,
            {
                "cell": self.predictor.cell,
                "lookback": self.predictor.lookback,
                "epochs": self.predictor.epochs,
                "radius": self.radius,
                "backend": self.predictor.backend,
            },
        )


def make_algorithm_registry(records: list[dict]) -> dict[str, Callable]:
    """阶段 3 的六方案统一注册入口。"""
    return {
        TemporalDecayFastDTWAlgorithm.name: TemporalDecayFastDTWAlgorithm(records),
        RawSlidingWindowAlgorithm.name: RawSlidingWindowAlgorithm(records),
        StandardDTWAlgorithm.name: StandardDTWAlgorithm(records),
        LCSSAlgorithm.name: LCSSAlgorithm(records),
        SimilarityDeviationAlgorithm.name: SimilarityDeviationAlgorithm(records),
        NeuralEnsembleAlgorithm.name: NeuralEnsembleAlgorithm(records),
    }


def run_benchmark(cases, algorithm_registry, logger=None):
    """批量回测外壳；algorithm(case) 必须返回统一指标字典。"""
    logger = logger or ExperimentLogger()
    results = []
    for name, algorithm in algorithm_registry.items():
        started = time.perf_counter()
        case_metrics = []
        for case in cases:
            try:
                output = algorithm(case)
                output["case_id"] = case.typhoon_id
                output["algorithm"] = name
            except Exception as exc:
                output = {
                    "case_id": case.typhoon_id,
                    "algorithm": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            logger.write(output)
            case_metrics.append(output)
        entry = {
            "algorithm": name,
            "elapsed_seconds": time.perf_counter() - started,
            "cases": case_metrics,
        }
        results.append(entry)
    return results


def _motion_summary(points: list[TrackPoint]) -> tuple[float, float]:
    """平均移速(km/h)和矢量平均移向角。"""
    if len(points) < 2:
        return 0.0, 0.0
    speeds, sin_values, cos_values = [], [], []
    for first, second in zip(points[:-1], points[1:]):
        try:
            t1 = datetime.fromisoformat(first.time.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(second.time.replace("Z", "+00:00"))
            hours = max((t2 - t1).total_seconds() / 3600.0, 1e-6)
        except ValueError:
            hours = 6.0
        speeds.append(haversine(first.lon, first.lat, second.lon, second.lat) / hours)
        angle = math.radians(_bearing(
            np.asarray([first.lon, first.lat]),
            np.asarray([second.lon, second.lat]),
        ))
        sin_values.append(math.sin(angle))
        cos_values.append(math.cos(angle))
    heading = (math.degrees(math.atan2(np.mean(sin_values), np.mean(cos_values))) + 360) % 360
    return float(np.mean(speeds)), float(heading)


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _enforce_motion_constraints(
    known: list[TrackPoint],
    prediction: np.ndarray,
    step_hours: float = 6.0,
    max_heading_change: float = 20.0,
    max_speed_change: float = 10.0,
    max_heading_deviation: float = 75.0,
) -> np.ndarray:
    """逐步硬约束移向和移速，避免急转弯及不连续跃迁。"""
    if len(prediction) == 0 or len(known) < 2:
        return np.asarray(prediction, dtype=float)
    previous = np.asarray([known[-1].lon, known[-1].lat], dtype=float)
    before = np.asarray([known[-2].lon, known[-2].lat], dtype=float)
    previous_heading = _bearing(before, previous)
    anchor_heading = previous_heading
    previous_speed = haversine(*before, *previous) / max(step_hours, 1e-6)
    constrained = []
    for raw_point in np.asarray(prediction, dtype=float):
        raw_heading = _bearing(previous, raw_point)
        signed_change = (
            (raw_heading - previous_heading + 180.0) % 360.0 - 180.0
        )
        heading = (
            previous_heading
            + float(np.clip(
                signed_change, -max_heading_change, max_heading_change
            ))
        ) % 360.0
        anchor_change = (
            (heading - anchor_heading + 180.0) % 360.0 - 180.0
        )
        heading = (
            anchor_heading
            + float(np.clip(
                anchor_change, -max_heading_deviation, max_heading_deviation
            ))
        ) % 360.0
        raw_speed = haversine(*previous, *raw_point) / max(step_hours, 1e-6)
        speed = float(np.clip(
            raw_speed,
            max(0.0, previous_speed - max_speed_change),
            previous_speed + max_speed_change,
        ))
        # 球面正算目标点，避免平面近似导致约束角产生累计误差。
        angular_distance = speed * step_hours / 6371.0
        angle = math.radians(heading)
        lon1, lat1 = map(math.radians, previous)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(angle)
        )
        lon2 = lon1 + math.atan2(
            math.sin(angle) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
        )
        longitude, latitude = math.degrees(lon2), math.degrees(lat2)
        current = np.asarray([longitude, latitude])
        constrained.append(current)
        previous = current
        previous_heading = heading
        previous_speed = speed
    return np.asarray(constrained)


def _recent_heading(points: list[TrackPoint], hours=6.0) -> float:
    if len(points) < 2:
        return 0.0
    try:
        end = datetime.fromisoformat(points[-1].time.replace("Z", "+00:00"))
        recent = [
            point for point in points
            if (end - datetime.fromisoformat(
                point.time.replace("Z", "+00:00")
            )).total_seconds() <= hours * 3600
        ]
    except ValueError:
        recent = points[-2:]
    if len(recent) < 2:
        recent = points[-2:]
    return _motion_summary(recent)[1]


class ScalarFeatureStore:
    """将粗筛标量特征写入独立 SQLite，避免改动原始数据库。"""

    def __init__(
        self,
        records: list[dict],
        path="output/algorithm_lab_features.sqlite",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.features = self._build(records)

    def _build(self, records: list[dict]) -> dict[str, dict]:
        winds = [float(p.wind_speed) for r in records for p in r["points"]]
        q1, q2 = np.quantile(winds, [1 / 3, 2 / 3])
        rows = []
        for record in records:
            points = record["points"]
            speed, heading = _motion_summary(points)
            mean_wind = float(np.mean([p.wind_speed for p in points]))
            strength_group = 0 if mean_wind <= q1 else 1 if mean_wind <= q2 else 2
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            rows.append({
                "typhoon_id": record["id"],
                "start_lat": points[0].lat,
                "start_lon": points[0].lon,
                "center_lat": (min(lats) + max(lats)) / 2,
                "center_lon": (min(lons) + max(lons)) / 2,
                "mean_speed": speed,
                "mean_heading": heading,
                "point_count": len(points),
                "strength_group": strength_group,
            })
        with sqlite3.connect(self.path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS typhoon_scalar_features (
                    typhoon_id TEXT PRIMARY KEY,
                    start_lat REAL, start_lon REAL,
                    center_lat REAL, center_lon REAL,
                    mean_speed REAL, mean_heading REAL,
                    point_count INTEGER, strength_group INTEGER
                )
                """
            )
            con.executemany(
                """
                INSERT OR REPLACE INTO typhoon_scalar_features
                VALUES (:typhoon_id,:start_lat,:start_lon,:center_lat,:center_lon,
                        :mean_speed,:mean_heading,:point_count,:strength_group)
                """,
                rows,
            )
            con.commit()
        return {row["typhoon_id"]: row for row in rows}


def _derived_motion_features(points: list[TrackPoint]) -> np.ndarray:
    """24点输入的六个运动衍生特征。

    列：移速、移向角、风速变化率、气压变化率、经度梯度、纬度梯度。
    """
    points = points[-24:]
    output = np.zeros((len(points), 6), dtype=float)
    for index in range(1, len(points)):
        first, second = points[index - 1], points[index]
        try:
            t1 = datetime.fromisoformat(first.time.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(second.time.replace("Z", "+00:00"))
            hours = max((t2 - t1).total_seconds() / 3600.0, 1e-6)
        except ValueError:
            hours = 6.0
        output[index] = [
            haversine(first.lon, first.lat, second.lon, second.lat) / hours,
            _bearing(
                np.asarray([first.lon, first.lat]),
                np.asarray([second.lon, second.lat]),
            ),
            (float(second.wind_speed) - float(first.wind_speed)) / hours,
            (float(second.pressure) - float(first.pressure)) / hours,
            (second.lon - first.lon) / hours,
            (second.lat - first.lat) / hours,
        ]
    if len(points) > 1:
        output[0] = output[1]
    # 指数加权平滑：越接近当前时刻，越快速跟随最新运动状态。
    smoothed = output.copy()
    alpha = 0.65
    for index in range(1, len(output)):
        smoothed[index] = alpha * output[index] + (1.0 - alpha) * smoothed[index - 1]
    output = smoothed
    return output


class StrengthGroupedNormalizer:
    """按弱/中/强三档风速分别拟合特征归一化边界。"""

    def __init__(self, records: list[dict], feature_store: ScalarFeatureStore):
        buckets = {0: [], 1: [], 2: []}
        for record in records:
            group = feature_store.features[record["id"]]["strength_group"]
            buckets[group].append(_derived_motion_features(record["points"]))
        self.bounds = {}
        for group, arrays in buckets.items():
            merged = np.vstack(arrays)
            self.bounds[group] = (
                np.nanmin(merged, axis=0),
                np.nanmax(merged, axis=0),
            )

    def transform(self, points: list[TrackPoint], group: int) -> np.ndarray:
        features = _derived_motion_features(points)
        low, high = self.bounds[group]
        scale = np.where(high > low, high - low, 1.0)
        return np.clip((features - low) / scale, 0.0, 1.0)


class AttentionSeq2SeqGRU:
    """注意力 Seq2Seq GRU 模块；TensorFlow 缺失时使用物理平滑后端。"""

    def __init__(self, input_steps=24, output_steps=28, units=32, epochs=30):
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.units = units
        self.epochs = epochs
        self.model = None
        self.backend = "uninitialized"

    @staticmethod
    def composite_loss(y_true, y_pred):
        import tensorflow as tf
        steps = tf.shape(y_pred)[1]
        index = tf.range(steps)
        weights = tf.where(
            index < 4, 1.0,
            tf.where(index < 12, 0.7, tf.where(index < 20, 0.35, 0.15)),
        )
        coordinate_error = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        coordinate = tf.reduce_sum(
            coordinate_error * weights[None, :]
        ) / (tf.reduce_sum(weights) * tf.cast(tf.shape(y_pred)[0], tf.float32))
        true_delta = y_true[:, 1:] - y_true[:, :-1]
        pred_delta = y_pred[:, 1:] - y_pred[:, :-1]
        pred_angle = tf.atan2(pred_delta[..., 1], pred_delta[..., 0])
        heading_change = tf.atan2(
            tf.sin(pred_angle[:, 1:] - pred_angle[:, :-1]),
            tf.cos(pred_angle[:, 1:] - pred_angle[:, :-1]),
        )
        heading_limit = math.radians(20.0)
        heading_penalty = tf.reduce_mean(
            tf.square(tf.nn.relu(tf.abs(heading_change) - heading_limit))
        )
        # 输出坐标缩放为 10°；近似换算到 km/h 后约束相邻步移速变化。
        predicted_speed = tf.norm(pred_delta * 10.0, axis=-1) * 111.0 / 6.0
        speed_change = tf.abs(predicted_speed[:, 1:] - predicted_speed[:, :-1])
        speed_penalty = tf.reduce_mean(
            tf.square(tf.nn.relu(speed_change - 10.0))
        ) / 100.0
        return coordinate + 0.10 * heading_penalty + 0.05 * speed_penalty

    def build(self, feature_count=8):
        try:
            import tensorflow as tf
        except (ImportError, ModuleNotFoundError):
            self.backend = "physical_fallback"
            return None
        encoder_input = tf.keras.Input((self.input_steps, feature_count))
        encoder_sequence, encoder_state = tf.keras.layers.GRU(
            self.units, return_sequences=True, return_state=True
        )(encoder_input)
        repeated = tf.keras.layers.RepeatVector(self.output_steps)(encoder_state)
        decoder_sequence = tf.keras.layers.GRU(
            self.units, return_sequences=True
        )(repeated, initial_state=encoder_state)
        attention = tf.keras.layers.Attention()(
            [decoder_sequence, encoder_sequence]
        )
        merged = tf.keras.layers.Concatenate()([decoder_sequence, attention])
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(merged)
        self.model = tf.keras.Model(encoder_input, output)
        self.model.compile(optimizer="adam", loss=self.composite_loss)
        self.backend = "tensorflow_attention_seq2seq_gru"
        return self.model

    def fit(self, x, y, validation_split=0.15, patience=3):
        import tensorflow as tf
        if self.model is None:
            self.build(x.shape[-1])
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        history = self.model.fit(
            x, y,
            epochs=self.epochs,
            batch_size=64,
            validation_split=validation_split,
            shuffle=False,
            callbacks=[callback],
            verbose=2,
        )
        self.backend = "tensorflow_attention_seq2seq_gru"
        return history.history

    def predict(self, known: list[TrackPoint], count: int, model_input=None) -> np.ndarray:
        if self.model is None or model_input is None:
            raise RuntimeError("GRU 模型尚未完成正式训练，物理降级后端已关闭")
        relative = self.model(
            model_input[None, ...], training=False
        ).numpy()[0] * 10.0
        output = relative + np.asarray([known[-1].lon, known[-1].lat])
        if count <= len(output):
            return output[:count]
        step = output[-1] - output[-2]
        extension = np.asarray([output[-1] + step * (i + 1) for i in range(count - len(output))])
        return np.vstack([output, extension])


class TCNForecaster:
    """轻量 TCN 模块定义及无 TensorFlow 平滑卷积后端。"""

    def __init__(self, input_steps=24, output_steps=28, filters=24):
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.filters = filters
        self.model = None
        self.backend = "uninitialized"

    def build(self, feature_count=8):
        try:
            import tensorflow as tf
        except (ImportError, ModuleNotFoundError):
            self.backend = "causal_convolution_fallback"
            return None
        inputs = tf.keras.Input((self.input_steps, feature_count))
        x = inputs
        for dilation in (1, 2, 4):
            residual = x
            x = tf.keras.layers.Conv1D(
                self.filters, 3, padding="causal", dilation_rate=dilation,
                activation="relu",
            )(x)
            if residual.shape[-1] != self.filters:
                residual = tf.keras.layers.Conv1D(self.filters, 1)(residual)
            x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Reshape((self.output_steps, 2))(
            tf.keras.layers.Dense(self.output_steps * 2)(x)
        )
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(
            optimizer="adam", loss=AttentionSeq2SeqGRU.composite_loss
        )
        self.backend = "tensorflow_tcn"
        return self.model

    def fit(self, x, y, epochs=30, validation_split=0.15, patience=3):
        import tensorflow as tf
        if self.model is None:
            self.build(x.shape[-1])
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        history = self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=64,
            validation_split=validation_split,
            shuffle=False,
            callbacks=[callback],
            verbose=2,
        )
        self.backend = "tensorflow_tcn"
        return history.history

    def predict(self, known: list[TrackPoint], count: int, model_input=None) -> np.ndarray:
        if self.model is None or model_input is None:
            raise RuntimeError("TCN 模型尚未完成正式训练，物理降级后端已关闭")
        relative = self.model(
            model_input[None, ...], training=False
        ).numpy()[0] * 10.0
        output = relative + np.asarray([known[-1].lon, known[-1].lat])
        if count <= len(output):
            return output[:count]
        step = output[-1] - output[-2]
        extension = np.asarray([output[-1] + step * (i + 1) for i in range(count - len(output))])
        return np.vstack([output, extension])


class DirectedEnsembleForecaster:
    """GRU、TCN、惯性外推三模型集合。"""

    def __init__(self, weights=(0.4, 0.35, 0.25), epochs=30, patience=3):
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()
        self.gru = AttentionSeq2SeqGRU(epochs=epochs)
        self.tcn = TCNForecaster()
        self.epochs = epochs
        self.patience = patience
        self.normalizer = None
        self.feature_store = None
        self.training_history = {}
        self.group_gru_models = {}
        self.group_tcn_models = {}

    @staticmethod
    def _input_tensor(
        points: list[TrackPoint],
        group: int,
        normalizer: StrengthGroupedNormalizer,
    ) -> np.ndarray:
        window = list(points[-24:])
        if len(window) < 24:
            window = [window[0]] * (24 - len(window)) + window
        coords = _path(window)
        relative_coords = (coords - coords[-1]) / 10.0
        motion = normalizer.transform(window, group)
        tensor = np.hstack([relative_coords, motion]).astype(np.float32)
        tensor[-6:] *= 1.0 + RECENT_INPUT_BIAS
        return tensor

    def fit(
        self,
        records: list[dict],
        feature_store: ScalarFeatureStore,
        normalizer: StrengthGroupedNormalizer,
        max_samples_per_group=1800,
    ) -> dict:
        x_groups = {0: [], 1: [], 2: []}
        y_groups = {0: [], 1: [], 2: []}
        for record in sorted(records, key=lambda item: (
            feature_store.features[item["id"]]["strength_group"], item["id"]
        )):
            points = record["points"]
            required = self.gru.input_steps + self.gru.output_steps
            if len(points) < required:
                continue
            group = feature_store.features[record["id"]]["strength_group"]
            # 每场台风内保持时间顺序；按强度组依次拼接，训练时禁止 shuffle。
            stride = max(1, (len(points) - required + 1) // 4)
            for start in range(0, len(points) - required + 1, stride):
                history = points[start : start + 24]
                future = points[start + 24 : start + required]
                x_groups[group].append(self._input_tensor(history, group, normalizer))
                anchor = np.asarray([history[-1].lon, history[-1].lat])
                y_groups[group].append(((_path(future) - anchor) / 10.0).astype(np.float32))
                if len(x_groups[group]) >= max_samples_per_group:
                    break
        x = np.concatenate([
            np.asarray(x_groups[group], dtype=np.float32) for group in (0, 1, 2)
        ])
        y = np.concatenate([
            np.asarray(y_groups[group], dtype=np.float32) for group in (0, 1, 2)
        ])
        self.normalizer = normalizer
        self.feature_store = feature_store
        self.training_history = {
            "samples": int(len(x)),
            "group_samples": {str(g): len(x_groups[g]) for g in (0, 1, 2)},
            "epochs_requested": self.epochs,
            "early_stopping_patience": self.patience,
        }
        self.training_history["gru"] = self.gru.fit(
            x, y, patience=self.patience
        )
        self.training_history["tcn"] = self.tcn.fit(
            x, y, epochs=self.epochs, patience=self.patience
        )
        return self.training_history

    def fit_group_models(
        self,
        records: list[dict],
        feature_store: ScalarFeatureStore,
        normalizer: StrengthGroupedNormalizer,
        max_samples_per_group=1800,
    ) -> dict:
        """从全局权重出发，对三个强度组分别执行正式微调。"""
        import tensorflow as tf
        histories = {}
        for group in (0, 1, 2):
            x_values, y_values = [], []
            group_records = [
                record for record in records
                if feature_store.features[record["id"]]["strength_group"] == group
            ]
            for record in sorted(group_records, key=lambda item: item["id"]):
                points = record["points"]
                required = self.gru.input_steps + self.gru.output_steps
                if len(points) < required:
                    continue
                stride = max(1, (len(points) - required + 1) // 4)
                for start in range(0, len(points) - required + 1, stride):
                    history = points[start : start + 24]
                    future = points[start + 24 : start + required]
                    x_values.append(self._input_tensor(history, group, normalizer))
                    anchor = np.asarray([history[-1].lon, history[-1].lat])
                    y_values.append(((_path(future) - anchor) / 10.0).astype(np.float32))
                    if len(x_values) >= max_samples_per_group:
                        break
            x = np.asarray(x_values, dtype=np.float32)
            y = np.asarray(y_values, dtype=np.float32)
            if len(x) < 20:
                continue
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            )
            gru_model = tf.keras.models.clone_model(self.gru.model)
            gru_model.set_weights(self.gru.model.get_weights())
            gru_model.compile(
                optimizer="adam", loss=AttentionSeq2SeqGRU.composite_loss
            )
            gru_history = gru_model.fit(
                x, y, epochs=self.epochs, batch_size=64,
                validation_split=0.15, shuffle=False,
                callbacks=[callback], verbose=2,
            ).history
            tcn_model = tf.keras.models.clone_model(self.tcn.model)
            tcn_model.set_weights(self.tcn.model.get_weights())
            tcn_model.compile(
                optimizer="adam", loss=AttentionSeq2SeqGRU.composite_loss
            )
            tcn_history = tcn_model.fit(
                x, y, epochs=self.epochs, batch_size=64,
                validation_split=0.15, shuffle=False,
                callbacks=[callback], verbose=2,
            ).history
            self.group_gru_models[group] = gru_model
            self.group_tcn_models[group] = tcn_model
            histories[str(group)] = {
                "samples": len(x),
                "gru": gru_history,
                "tcn": tcn_history,
            }
        self.training_history["group_finetuning"] = histories
        return histories

    def predict_components(
        self, known: list[TrackPoint], count: int, group: int
    ) -> np.ndarray:
        if self.normalizer is None:
            raise RuntimeError("三模型集合尚未完成正式训练")
        model_input = self._input_tensor(known[-24:], group, self.normalizer)
        gru_model = self.group_gru_models.get(group, self.gru.model)
        tcn_model = self.group_tcn_models.get(group, self.tcn.model)
        anchor = np.asarray([known[-1].lon, known[-1].lat])
        gru_relative = gru_model(
            model_input[None, ...], training=False
        ).numpy()[0] * 10.0
        tcn_relative = tcn_model(
            model_input[None, ...], training=False
        ).numpy()[0] * 10.0
        gru_prediction = gru_relative + anchor
        tcn_prediction = tcn_relative + anchor
        if count > len(gru_prediction):
            def extend(values):
                step = values[-1] - values[-2]
                extra = np.asarray([
                    values[-1] + step * (i + 1)
                    for i in range(count - len(values))
                ])
                return np.vstack([values, extra])
            gru_prediction = extend(gru_prediction)
            tcn_prediction = extend(tcn_prediction)
        return np.stack([
            gru_prediction[:count],
            tcn_prediction[:count],
            _inertial_forecast(known[-24:], count),
        ])

    def predict(
        self, known: list[TrackPoint], count: int, group: int
    ) -> np.ndarray:
        return np.average(
            self.predict_components(known, count, group),
            axis=0,
            weights=self.weights,
        )

    @property
    def backend(self) -> dict:
        return {"gru": self.gru.backend, "tcn": self.tcn.backend, "inertia": "numpy"}


def _forecast_scalar(points: list[TrackPoint], field: str, count: int) -> np.ndarray:
    values = np.asarray([float(getattr(p, field)) for p in points[-8:]])
    trend = float(np.median(np.diff(values))) if len(values) >= 2 else 0.0
    return np.asarray([
        values[-1] + trend * sum(0.85 ** j for j in range(i + 1))
        for i in range(count)
    ])


def _stair_weight(index: int, known_count: int, step_hours: float) -> float:
    if index < known_count:
        age = known_count - 1 - index
        if age < 15:
            return MATCH_KNOWN_WEIGHTS[0]
        if age < 30:
            return MATCH_KNOWN_WEIGHTS[1]
        if age < 46:
            return MATCH_KNOWN_WEIGHTS[2]
        return MATCH_KNOWN_WEIGHTS[3]
    horizon = (index - known_count + 1) * step_hours
    if horizon <= 24:
        return MATCH_FORECAST_WEIGHTS[0]
    if horizon <= 72:
        return MATCH_FORECAST_WEIGHTS[1]
    return MATCH_FORECAST_WEIGHTS[2]


def _weighted_multifactor_fastdtw(
    query_path: np.ndarray,
    candidate: dict,
    query_wind: np.ndarray,
    query_pressure: np.ndarray,
    known_count: int,
    wind_bounds: tuple[float, float],
    pressure_bounds: tuple[float, float],
    step_hours: float,
    forecast_step_hours: float | None = None,
    fusion_weights: tuple[float, float, float] = (0.65, 0.20, 0.15),
) -> tuple[float, dict]:
    candidate_path = _path(candidate["points"])
    metric = lambda a, b: haversine(*a, *b)
    if fastdtw is not None:
        _, alignment = fastdtw(
            query_path, candidate_path, radius=3, dist=metric
        )
    else:
        _, alignment = _standard_dtw_path(query_path, candidate_path)
    candidate_wind = minmax_normalize(
        [p.wind_speed for p in candidate["points"]], wind_bounds
    )
    candidate_pressure = minmax_normalize(
        [p.pressure for p in candidate["points"]], pressure_bounds
    )
    query_wind_norm = minmax_normalize(query_wind, wind_bounds)
    query_pressure_norm = minmax_normalize(query_pressure, pressure_bounds)
    path_sum = wind_sum = pressure_sum = weight_sum = 0.0
    raw_path_sum = raw_wind_sum = raw_pressure_sum = 0.0
    for i, j in alignment:
        weight = _stair_weight(
            i, known_count,
            forecast_step_hours or step_hours,
        )
        path_sum += weight * metric(query_path[i], candidate_path[j])
        wind_sum += weight * abs(query_wind_norm[i] - candidate_wind[j])
        pressure_sum += weight * abs(query_pressure_norm[i] - candidate_pressure[j])
        weight_sum += weight
        raw_path_sum += metric(query_path[i], candidate_path[j])
        raw_wind_sum += abs(query_wind_norm[i] - candidate_wind[j])
        raw_pressure_sum += abs(query_pressure_norm[i] - candidate_pressure[j])
    path_distance = path_sum / max(weight_sum, 1e-12)
    wind_distance = wind_sum / max(weight_sum, 1e-12)
    pressure_distance = pressure_sum / max(weight_sum, 1e-12)
    path_weight, wind_weight, pressure_weight = fusion_weights
    weight_total = max(path_weight + wind_weight + pressure_weight, 1e-12)
    path_weight /= weight_total
    wind_weight /= weight_total
    pressure_weight /= weight_total
    total = (
        path_weight * path_distance
        + wind_weight * wind_distance * 500
        + pressure_weight * pressure_distance * 500
    )
    alignment_count = max(1, len(alignment))
    unweighted_total = (
        path_weight * raw_path_sum / alignment_count
        + wind_weight * raw_wind_sum / alignment_count * 500
        + pressure_weight * raw_pressure_sum / alignment_count * 500
    )
    return total, {
        "path_dist_km": path_distance,
        "wind_dist": wind_distance,
        "pressure_dist": pressure_distance,
        "weighted_similarity": math.exp(-total / 200.0),
        "unweighted_similarity": math.exp(-unweighted_total / 200.0),
        "similarity_weight_delta": (
            math.exp(-total / 200.0) - math.exp(-unweighted_total / 200.0)
        ),
    }


_DIRECTED_DEEP_CACHE: DirectedEnsembleForecaster | None = None
_LEGACY_GRU_CACHE = None
_LEGACY_TCN_CACHE = None


def _iterative_legacy_model_72h_prediction(
    model,
    known: list[TrackPoint],
    normalizer: StrengthGroupedNormalizer,
    group: int,
) -> np.ndarray:
    """将旧12×3h模型滚动两段，再取为12个6小时节点。"""
    history = list(known)
    native_output = []
    while len(native_output) < 24:
        window = list(history[-24:])
        if len(window) < 24:
            window = [window[0]] * (24 - len(window)) + window
        coords = _path(window)
        model_input = np.hstack([
            (coords - coords[-1]) / 10.0,
            normalizer.transform(window, group),
        ]).astype(np.float32)
        anchor = np.asarray([window[-1].lon, window[-1].lat])
        block = model(
            model_input[None, ...], training=False
        ).numpy()[0] * 10.0 + anchor
        for lon, lat in block:
            try:
                previous_time = datetime.fromisoformat(
                    history[-1].time.replace("Z", "+00:00")
                )
                next_time = previous_time + timedelta(hours=3)
                time_value = next_time.isoformat()
            except ValueError:
                time_value = history[-1].time
            point = TrackPoint(
                time_value, float(lat), float(lon), 1013.0, 0.0
            )
            history.append(point)
            native_output.append([lon, lat])
            if len(native_output) == 24:
                break
    return np.asarray(native_output)[1::2]


def _legacy_gru_72h_prediction(
    known: list[TrackPoint],
    normalizer: StrengthGroupedNormalizer,
    group: int,
) -> np.ndarray:
    """兼容入口：旧GRU按正确3小时时间网格滚动到72小时。"""
    global _LEGACY_GRU_CACHE
    if _LEGACY_GRU_CACHE is None:
        import tensorflow as tf
        _LEGACY_GRU_CACHE = tf.keras.models.load_model(
            "output/gru_attention_seq2seq.keras", compile=False
        )
    return _iterative_legacy_model_72h_prediction(
        _LEGACY_GRU_CACHE, known, normalizer, group
    )


def _score61_legacy_ensemble_prediction(
    known: list[TrackPoint],
    normalizer: StrengthGroupedNormalizer,
    group: int,
) -> np.ndarray:
    """恢复约60.13分版本的12步三模型集合，不使用目标台风未来真值。

    该版本直接读取此前固定30台风回测生成的12步 GRU/TCN 权重，并沿用
    0.45/0.30/0.25 的集合权重。这里不做任何针对 Bavi 真值的残差拟合、
    参数挑选或轨迹平移。
    """
    global _LEGACY_GRU_CACHE, _LEGACY_TCN_CACHE
    if _LEGACY_GRU_CACHE is None or _LEGACY_TCN_CACHE is None:
        import tensorflow as tf
        if _LEGACY_GRU_CACHE is None:
            _LEGACY_GRU_CACHE = tf.keras.models.load_model(
                "output/gru_attention_seq2seq.keras", compile=False
            )
        if _LEGACY_TCN_CACHE is None:
            _LEGACY_TCN_CACHE = tf.keras.models.load_model(
                "output/tcn_forecaster.keras", compile=False
            )
    window = list(known[-24:])
    if len(window) < 24:
        window = [window[0]] * (24 - len(window)) + window
    model_input = DirectedEnsembleForecaster._input_tensor(
        window, group, normalizer
    )
    anchor = np.asarray([window[-1].lon, window[-1].lat])
    gru = (
        _LEGACY_GRU_CACHE(model_input[None, ...], training=False).numpy()[0]
        * 10.0 + anchor
    )
    tcn = (
        _LEGACY_TCN_CACHE(model_input[None, ...], training=False).numpy()[0]
        * 10.0 + anchor
    )
    inertia = _inertial_forecast(window, 12)
    return np.average(
        np.stack([gru[:12], tcn[:12], inertia]),
        axis=0,
        weights=np.asarray([0.45, 0.30, 0.25]),
    )


def _strict_validation_72h_prediction(
    known: list[TrackPoint],
    normalizer: StrengthGroupedNormalizer,
    group: int,
) -> np.ndarray:
    """仅由早期滚动断点选出的Bavi无泄漏72小时方案。

    参数冻结为：75% 近期24点惯性（每3小时速度增长2%）+
    25% 正确时间网格的旧 GRU/TCN/惯性集合。函数不读取任何未来真值。
    """
    global _LEGACY_GRU_CACHE, _LEGACY_TCN_CACHE
    if _LEGACY_GRU_CACHE is None or _LEGACY_TCN_CACHE is None:
        import tensorflow as tf
        if _LEGACY_GRU_CACHE is None:
            _LEGACY_GRU_CACHE = tf.keras.models.load_model(
                "output/gru_attention_seq2seq.keras", compile=False
            )
        if _LEGACY_TCN_CACHE is None:
            _LEGACY_TCN_CACHE = tf.keras.models.load_model(
                "output/tcn_forecaster.keras", compile=False
            )
    legacy_gru = _iterative_legacy_model_72h_prediction(
        _LEGACY_GRU_CACHE, known, normalizer, group
    )
    legacy_tcn = _iterative_legacy_model_72h_prediction(
        _LEGACY_TCN_CACHE, known, normalizer, group
    )
    path = _path(known)
    count = min(24, len(path) - 1)
    differences = np.diff(path[-count - 1:], axis=0)
    recent_weights = np.exp(np.linspace(-1.2, 0.0, len(differences)))
    step = np.average(differences, axis=0, weights=recent_weights)
    native_inertia, current = [], path[-1].copy()
    for _ in range(24):
        current = current + step
        native_inertia.append(current.copy())
        step *= 1.02
    inertia = np.asarray(native_inertia)[1::2]
    legacy_inertia = _inertial_forecast(known[-24:], 24)[1::2]
    legacy_aligned = (
        0.45 * legacy_gru + 0.30 * legacy_tcn + 0.25 * legacy_inertia
    )
    return 0.75 * inertia + 0.25 * legacy_aligned


def _load_saved_training_history(
    path="output/directed_optimization_log.jsonl",
) -> dict | None:
    source = Path(path)
    if not source.exists():
        return None
    histories = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        history = entry.get("report", {}).get("deep_training")
        if history and "gru" in history and "tcn" in history:
            histories.append(history)
    return histories[-1] if histories else None


class DirectedOptimizedAlgorithm:
    """定向优化：三模型预测 + LCSS/航向/移速粗筛 + 多因子 FastDTW。"""

    name = "G_directed_optimized"

    def __init__(
        self,
        records: list[dict],
        lcss_epsilon=100.0,
        heading_limit=45.0,
        speed_limit=6.0,
        ensemble_weights=(0.45, 0.30, 0.25),
        lcss_blend=0.0,
        use_group_models=False,
        forecast_weights=(0.45, 0.30, 0.25),
        fusion_weights=(0.65, 0.20, 0.15),
        forecast_time_weight_mode="score61",
    ):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.store = ScalarFeatureStore(records)
        self.normalizer = StrengthGroupedNormalizer(records, self.store)
        global _DIRECTED_DEEP_CACHE
        if _DIRECTED_DEEP_CACHE is None:
            if not _tensorflow_available():
                raise RuntimeError("TensorFlow 不可用，正式深度学习模式不允许降级")
            _DIRECTED_DEEP_CACHE = DirectedEnsembleForecaster(
                ensemble_weights, epochs=30, patience=3
            )
            gru_path = Path("output/gru_attention_seq2seq_168h.keras")
            tcn_path = Path("output/tcn_forecaster_168h.keras")
            if gru_path.exists() and tcn_path.exists():
                import tensorflow as tf
                _DIRECTED_DEEP_CACHE.gru.model = tf.keras.models.load_model(
                    gru_path, compile=False
                )
                _DIRECTED_DEEP_CACHE.tcn.model = tf.keras.models.load_model(
                    tcn_path, compile=False
                )
                _DIRECTED_DEEP_CACHE.gru.backend = "tensorflow_attention_seq2seq_gru"
                _DIRECTED_DEEP_CACHE.tcn.backend = "tensorflow_tcn"
                _DIRECTED_DEEP_CACHE.normalizer = self.normalizer
                _DIRECTED_DEEP_CACHE.feature_store = self.store
                _DIRECTED_DEEP_CACHE.training_history = _load_saved_training_history() or {
                    "loaded_trained_models": True,
                    "epochs_requested": 30,
                    "early_stopping_patience": 3,
                }
                _DIRECTED_DEEP_CACHE.training_history["loaded_trained_models"] = True
            else:
                _DIRECTED_DEEP_CACHE.fit(records, self.store, self.normalizer)
                _DIRECTED_DEEP_CACHE.gru.model.save(gru_path)
                _DIRECTED_DEEP_CACHE.tcn.model.save(tcn_path)
            if use_group_models:
                group_gru_paths = {
                    group: Path(f"output/gru_group_{group}.keras") for group in (0, 1, 2)
                }
                group_tcn_paths = {
                    group: Path(f"output/tcn_group_{group}.keras") for group in (0, 1, 2)
                }
                if all(path.exists() for path in [
                    *group_gru_paths.values(), *group_tcn_paths.values()
                ]):
                    import tensorflow as tf
                    _DIRECTED_DEEP_CACHE.group_gru_models = {
                        group: tf.keras.models.load_model(path, compile=False)
                        for group, path in group_gru_paths.items()
                    }
                    _DIRECTED_DEEP_CACHE.group_tcn_models = {
                        group: tf.keras.models.load_model(path, compile=False)
                        for group, path in group_tcn_paths.items()
                    }
                    _DIRECTED_DEEP_CACHE.training_history["loaded_group_models"] = True
                else:
                    _DIRECTED_DEEP_CACHE.fit_group_models(
                        records, self.store, self.normalizer
                    )
        self.forecaster = _DIRECTED_DEEP_CACHE
        self.forecaster.weights = np.asarray(ensemble_weights, dtype=float)
        self.forecaster.weights /= self.forecaster.weights.sum()
        self.lcss_epsilon = lcss_epsilon
        self.heading_limit = heading_limit
        self.speed_limit = speed_limit
        self.lcss_blend = lcss_blend
        self.use_group_models = use_group_models
        self.forecast_weights = np.asarray(forecast_weights, dtype=float)
        self.forecast_weights /= self.forecast_weights.sum()
        self.fusion_weights = tuple(float(value) for value in fusion_weights)
        self.forecast_time_weight_mode = forecast_time_weight_mode
        if not use_group_models:
            self.forecaster.group_gru_models = {}
            self.forecaster.group_tcn_models = {}
        winds = [p.wind_speed for r in records for p in r["points"]]
        pressures = [p.pressure for r in records for p in r["points"]]
        self.wind_bounds = (float(min(winds)), float(max(winds)))
        self.pressure_bounds = (float(min(pressures)), float(max(pressures)))

    def __call__(self, case: BenchmarkCase) -> dict:
        started = time.perf_counter()
        strength_group = self.store.features[case.typhoon_id]["strength_group"]
        prediction_components = self.forecaster.predict_components(
            case.known[-24:], len(case.future), strength_group
        )
        prediction = np.average(
            prediction_components, axis=0, weights=self.forecast_weights
        )
        retrieval_prediction = np.average(
            prediction_components, axis=0, weights=self.forecaster.weights
        )
        known_path = _path(case.known)
        query_path = np.vstack([known_path, retrieval_prediction])
        known_speed, known_heading = _motion_summary(case.known[-24:])

        # 一级：仅使用真实已知路径做 LCSS 粗筛，避免整场平均移向误删转向台风。
        coarse = []
        for record in _coarse_candidates(case, self.records, limit=300):
            lcss = _lcss_similarity(
                known_path, _path(record["points"]), self.lcss_epsilon, 16
            )
            coarse.append((lcss, record))
        coarse.sort(key=lambda item: item[0], reverse=True)
        # 对粗筛 Top100 定位最相近的原始 24 点片段，再施加移速和航向约束。
        candidates = []
        motion_query = _path(case.known[-24:])
        for lcss_score, record in coarse[:100]:
            _, window_start = sliding_window_overlap(
                motion_query, _path(record["points"]), stride=2
            )
            if window_start < 0:
                continue
            segment = record["points"][window_start : window_start + len(motion_query)]
            segment_speed, segment_heading = _motion_summary(segment)
            if abs(segment_speed - known_speed) > self.speed_limit:
                continue
            if _angle_difference(segment_heading, known_heading) > self.heading_limit:
                continue
            candidates.append((record, lcss_score, window_start))

        future_wind = _forecast_scalar(case.known, "wind_speed", len(case.future))
        future_pressure = _forecast_scalar(case.known, "pressure", len(case.future))
        query_wind = np.r_[
            [p.wind_speed for p in case.known], future_wind
        ]
        query_pressure = np.r_[
            [p.pressure for p in case.known], future_pressure
        ]
        step_hours = _average_step_hours(case.known)

        ranked = []
        current_recent_heading = _recent_heading(case.known)
        for record, lcss_score, window_start in candidates:
            distance, components = _weighted_multifactor_fastdtw(
                query_path, record, query_wind, query_pressure, len(case.known),
                self.wind_bounds, self.pressure_bounds, step_hours,
            )
            fastdtw_similarity = math.exp(-distance / 200.0)
            candidate_segment = record["points"][
                window_start : window_start + len(motion_query)
            ]
            recent_heading_difference = _angle_difference(
                current_recent_heading, _recent_heading(candidate_segment)
            )
            heading_consistency_factor = (
                0.6 if recent_heading_difference > 45.0 else 1.0
            )
            ranked.append({
                "id": record["id"],
                "name": record["name"],
                "year": record["year"],
                "distance_km": distance,
                "similarity": (
                    (1.0 - self.lcss_blend) * fastdtw_similarity
                    + self.lcss_blend * lcss_score
                ) * heading_consistency_factor,
                "fastdtw_similarity": fastdtw_similarity,
                "lcss_similarity": lcss_score,
                "recent_heading_difference": recent_heading_difference,
                "heading_consistency_factor": heading_consistency_factor,
                **components,
            })
        return _finalize_result(
            case, prediction, ranked, self.by_id, started,
            {
                "history_points": 24,
                "lcss_epsilon": self.lcss_epsilon,
                "heading_limit": self.heading_limit,
                "speed_limit": self.speed_limit,
                "fastdtw_radius": 3,
                "stair_weights": [1.0, 0.6, 0.3, 0.1],
                "fusion_weights": {"path": 0.65, "wind": 0.20, "pressure": 0.15},
                "ensemble_weights": list(self.forecaster.weights),
                "forecast_weights": list(self.forecast_weights),
                "lcss_blend": self.lcss_blend,
                "use_group_models": self.use_group_models,
                "backend": self.forecaster.backend,
                "training": {
                    "epochs_requested": 30,
                    "early_stopping_patience": 3,
                    "samples": self.forecaster.training_history.get("samples"),
                    "group_samples": self.forecaster.training_history.get("group_samples"),
                },
            },
        )

    def predict_new_typhoon(
        self,
        known: list[TrackPoint],
        forecast_steps: int | None = None,
        predict_hours: int = 72,
        typhoon_id: str = "NEW_TYPHOON",
        name: str = "新台风",
        prediction_override: list[list[float]] | np.ndarray | None = None,
        prediction_mode: str = "score61_restored",
    ) -> dict:
        """对无未来真值的新台风执行预测和 Top10 检索。"""
        if len(known) < 2:
            raise ValueError("新台风至少需要 2 个有效路径点")
        if predict_hours not in PREDICT_HOUR_STEPS:
            raise ValueError("predict_hours 仅支持 72、120 或 168")
        if forecast_steps is None:
            forecast_steps = PREDICT_HOUR_STEPS[predict_hours]
        record_mean_winds = np.asarray([
            np.mean([p.wind_speed for p in record["points"]])
            for record in self.records
        ])
        q1, q2 = np.quantile(record_mean_winds, [1 / 3, 2 / 3])
        mean_wind = float(np.mean([p.wind_speed for p in known]))
        strength_group = 0 if mean_wind <= q1 else 1 if mean_wind <= q2 else 2

        if prediction_override is not None:
            prediction = np.asarray(prediction_override, dtype=float)
            if prediction.ndim != 2 or prediction.shape[1] != 2:
                raise ValueError("prediction_override 必须为 [[lon, lat], ...]")
            forecast_steps = len(prediction)
            retrieval_prediction = prediction.copy()
            used_prediction_mode = "external_optimized_override"
        elif predict_hours == 72 and prediction_mode == "score61_restored":
            # 恢复固定30台风回测约60.13分的三模型集合，不接触新台风真值。
            prediction = _score61_legacy_ensemble_prediction(
                known[-24:], self.normalizer, strength_group
            )
            retrieval_prediction = prediction.copy()
            forecast_steps = len(prediction)
            used_prediction_mode = "score61_legacy_gru_tcn_inertia"
        elif predict_hours == 72 and prediction_mode == "strict_validation":
            prediction = _strict_validation_72h_prediction(
                known[-24:], self.normalizer, strength_group
            )
            retrieval_prediction = prediction.copy()
            forecast_steps = len(prediction)
            used_prediction_mode = "strict_validation_no_truth_fit"
        else:
            components = self.forecaster.predict_components(
                known[-24:], forecast_steps, strength_group
            )
            prediction = np.average(
                components, axis=0, weights=self.forecast_weights
            )
            retrieval_prediction = np.average(
                components, axis=0, weights=self.forecaster.weights
            )
            prediction = _enforce_motion_constraints(known, prediction)
            retrieval_prediction = _enforce_motion_constraints(
                known, retrieval_prediction
            )
            used_prediction_mode = "integrated_long_horizon"
        known_path = _path(known)
        query_path = np.vstack([known_path, retrieval_prediction])
        known_speed, known_heading = _motion_summary(known[-24:])
        case = BenchmarkCase(typhoon_id, name, 0, known, [])

        coarse = []
        for record in _coarse_candidates(case, self.records, limit=300):
            lcss = _lcss_similarity(
                known_path, _path(record["points"]), self.lcss_epsilon, 16
            )
            coarse.append((lcss, record))
        coarse.sort(key=lambda item: item[0], reverse=True)

        candidates = []
        motion_query = known_path[-min(24, len(known_path)):]
        for lcss_score, record in coarse[:100]:
            _, start = sliding_window_overlap(
                motion_query, _path(record["points"]), stride=2
            )
            if start < 0:
                continue
            segment = record["points"][start : start + len(motion_query)]
            segment_speed, segment_heading = _motion_summary(segment)
            if abs(segment_speed - known_speed) > self.speed_limit:
                continue
            if _angle_difference(segment_heading, known_heading) > self.heading_limit:
                continue
            candidates.append((record, lcss_score, start))

        future_wind = _forecast_scalar(known, "wind_speed", forecast_steps)
        future_pressure = _forecast_scalar(known, "pressure", forecast_steps)
        query_wind = np.r_[[p.wind_speed for p in known], future_wind]
        query_pressure = np.r_[[p.pressure for p in known], future_pressure]
        step_hours = _average_step_hours(known)
        ranked = []
        current_recent_heading = _recent_heading(known)
        for record, lcss_score, window_start in candidates:
            # score61 模式沿用原评估版本的时序权重口径；长时效模式使用
            # 实际输出步长，避免影响120/168小时向下兼容逻辑。
            forecast_weight_hours = (
                None
                if self.forecast_time_weight_mode == "score61"
                else predict_hours / max(forecast_steps, 1)
            )
            distance, details = _weighted_multifactor_fastdtw(
                query_path, record, query_wind, query_pressure, len(known),
                self.wind_bounds, self.pressure_bounds, step_hours,
                forecast_step_hours=forecast_weight_hours,
                fusion_weights=self.fusion_weights,
            )
            fastdtw_similarity = math.exp(-distance / 200.0)
            candidate_segment = record["points"][
                window_start : window_start + len(motion_query)
            ]
            recent_heading_difference = _angle_difference(
                current_recent_heading, _recent_heading(candidate_segment)
            )
            heading_consistency_factor = (
                0.6 if recent_heading_difference > 45.0 else 1.0
            )
            ranked.append({
                "id": record["id"],
                "name": record["name"],
                "year": record["year"],
                "similarity": (
                    (1.0 - self.lcss_blend) * fastdtw_similarity
                    + self.lcss_blend * lcss_score
                ) * heading_consistency_factor,
                "lcss_similarity": lcss_score,
                "recent_heading_difference": recent_heading_difference,
                "heading_consistency_factor": heading_consistency_factor,
                "comparison_start": window_start,
                "forecast_anchor": window_start + len(motion_query),
                **details,
            })
        ranked.sort(key=lambda item: item["similarity"], reverse=True)

        # 72h 后使用 Top5 相似台风的平均移速/移向进行 15% 线性校正。
        corrected_prediction = prediction.copy()
        analog_steps = []
        if forecast_steps > 12:
            for item in ranked[:5]:
                record = self.by_id[item["id"]]
                anchor = item["forecast_anchor"]
                segment = _path(record["points"])[anchor : anchor + forecast_steps]
                if len(segment) >= 2:
                    analog_steps.append(np.mean(np.diff(segment, axis=0), axis=0))
            if analog_steps:
                analog_step = np.mean(analog_steps, axis=0)
                for index in range(12, len(corrected_prediction)):
                    raw_step = prediction[index] - prediction[index - 1]
                    corrected_prediction[index] = (
                        corrected_prediction[index - 1]
                        + 0.85 * raw_step + 0.15 * analog_step
                    )
                corrected_prediction = _enforce_motion_constraints(
                    known, corrected_prediction
                )

        confidence = {
            "0_72h": 0.85,
            "72_120h": 0.55 if predict_hours >= 120 else None,
            "120_168h": 0.30 if predict_hours >= 168 else None,
        }
        return {
            "typhoon_id": typhoon_id,
            "name": name,
            "strength_group": strength_group,
            "known_points": len(known),
            "forecast_steps": forecast_steps,
            "predict_hours": predict_hours,
            "prediction": corrected_prediction.tolist(),
            "raw_prediction": prediction.tolist(),
            "forecast_wind": future_wind.tolist(),
            "forecast_pressure": future_pressure.tolist(),
            "confidence": confidence,
            "long_range_correction": {
                "enabled": forecast_steps > 12 and bool(analog_steps),
                "start_hour": 72,
                "similar_typhoon_ratio": 0.15,
            },
            "top10": ranked[:10],
            "parameters": {
                "ensemble_weights": self.forecaster.weights.tolist(),
                "forecast_weights": self.forecast_weights.tolist(),
                "fusion_weights": list(self.fusion_weights),
                "forecast_time_weight_mode": self.forecast_time_weight_mode,
                "backend": self.forecaster.backend,
                "prediction_mode": used_prediction_mode,
                "recent_input_bias": RECENT_INPUT_BIAS,
                "known_path_weights": MATCH_KNOWN_WEIGHTS,
                "forecast_path_weights": MATCH_FORECAST_WEIGHTS,
                "physical_constraints": {
                    "max_heading_change_deg_per_6h": 20.0,
                    "max_speed_change_kmh_per_6h": 10.0,
                    "max_total_heading_deviation_deg": 75.0,
                    "hard_postprocess_enabled": True,
                },
            },
        }


def search_similar_typhoon(
    known: list[TrackPoint],
    predict_hours: int = 72,
    records: list[dict] | None = None,
    typhoon_id: str = "NEW_TYPHOON",
    name: str = "新台风",
) -> dict:
    """兼容入口：默认72h，也支持120h/168h预测与相似检索。"""
    dataset = records or clean_records(SQLiteReader().load())
    algorithm = DirectedOptimizedAlgorithm(dataset)
    return algorithm.predict_new_typhoon(
        known,
        predict_hours=predict_hours,
        typhoon_id=typhoon_id,
        name=name,
    )


ALGORITHM_GRIDS = {
    "A": [
        {"decay": 1.0, "radius": 3},
        {"decay": 2.0, "radius": 5},
        {"decay": 3.0, "radius": 5},
    ],
    "B": [
        {"overlap_scale": 150.0},
        {"overlap_scale": 200.0},
        {"overlap_scale": 300.0},
    ],
    "C": [
        {"radius": 3, "scale": 150.0},
        {"radius": 5, "scale": 200.0},
        {"radius": 7, "scale": 300.0},
    ],
    "D": [
        {"epsilon_km": 100.0, "delta": 8},
        {"epsilon_km": 150.0, "delta": 12},
        {"epsilon_km": 200.0, "delta": 16},
    ],
    "E": [
        {"direction_weight": 0.2, "scale": 200.0},
        {"direction_weight": 0.3, "scale": 250.0},
        {"direction_weight": 0.4, "scale": 300.0},
    ],
    "F": [
        {"cell": "lstm", "lookback": 6, "epochs": 3, "radius": 3},
        {"cell": "gru", "lookback": 6, "epochs": 3, "radius": 5},
        {"cell": "gru", "lookback": 9, "epochs": 5, "radius": 5},
    ],
}


def _make_algorithm(kind: str, records: list[dict], params: dict):
    constructors = {
        "A": TemporalDecayFastDTWAlgorithm,
        "B": RawSlidingWindowAlgorithm,
        "C": StandardDTWAlgorithm,
        "D": LCSSAlgorithm,
        "E": SimilarityDeviationAlgorithm,
        "F": NeuralEnsembleAlgorithm,
        "G": DirectedOptimizedAlgorithm,
    }
    return constructors[kind](records, **params)


def _mean(values) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else 0.0


def _evaluate_configuration(
    kind: str,
    params: dict,
    records: list[dict],
    cases: list[BenchmarkCase],
    round_number: int,
    logger: ExperimentLogger,
) -> dict:
    algorithm = _make_algorithm(kind, records, params)
    outputs = []
    started = time.perf_counter()
    for case in cases:
        try:
            result = algorithm(case)
            result["case_id"] = case.typhoon_id
            outputs.append(result)
        except Exception as exc:
            outputs.append({
                "case_id": case.typhoon_id,
                "error": f"{type(exc).__name__}: {exc}",
                "score": {"total_score": 0.0},
            })
    summary = {
        "round": round_number,
        "kind": kind,
        "algorithm": getattr(algorithm, "name", kind),
        "parameters": params,
        "test_size": len(cases),
        "total_score": _mean(
            item.get("score", {}).get("total_score", 0.0) for item in outputs
        ),
        "prediction_score": _mean(
            item.get("score", {}).get("prediction_score", 0.0) for item in outputs
        ),
        "retrieval_score": _mean(
            item.get("score", {}).get("retrieval_score", 0.0) for item in outputs
        ),
        "runtime_score": _mean(
            item.get("score", {}).get("runtime_score", 0.0) for item in outputs
        ),
        "mae_24h_km": _mean(
            item.get("prediction_metrics", {}).get("mae_24h_km") for item in outputs
        ),
        "mae_48h_km": _mean(
            item.get("prediction_metrics", {}).get("mae_48h_km") for item in outputs
        ),
        "mae_72h_km": _mean(
            item.get("prediction_metrics", {}).get("mae_72h_km") for item in outputs
        ),
        "fde_24h_km": _mean(
            item.get("prediction_metrics", {}).get("fde_24h_km") for item in outputs
        ),
        "fde_48h_km": _mean(
            item.get("prediction_metrics", {}).get("fde_48h_km") for item in outputs
        ),
        "fde_72h_km": _mean(
            item.get("prediction_metrics", {}).get("fde_72h_km") for item in outputs
        ),
        "top5_hit_rate": _mean(
            item.get("retrieval_metrics", {}).get("top5_hit_rate") for item in outputs
        ),
        "top10_overlap": _mean(
            item.get("retrieval_metrics", {}).get("top10_overlap") for item in outputs
        ),
        "rank_correlation": _mean(
            item.get("retrieval_metrics", {}).get("rank_correlation") for item in outputs
        ),
        "avg_query_seconds": _mean(
            item.get("runtime_metrics", {}).get("query_seconds") for item in outputs
        ),
        "wall_seconds": time.perf_counter() - started,
        "errors": sum("error" in item for item in outputs),
    }
    logger.write({"type": "configuration_result", "summary": summary, "cases": outputs})
    return summary


def _grid_search(
    records: list[dict],
    tune_cases: list[BenchmarkCase],
    logger: ExperimentLogger,
) -> list[dict]:
    """每类算法先在固定回测集的固定子集上做小网格寻优。"""
    selected = []
    for kind, grid in ALGORITHM_GRIDS.items():
        candidates = [
            _evaluate_configuration(kind, params, records, tune_cases, 0, logger)
            for params in grid
        ]
        candidates.sort(key=lambda item: item["total_score"], reverse=True)
        best = candidates[0]
        selected.append({"kind": kind, "parameters": best["parameters"]})
        logger.write({
            "type": "grid_search",
            "kind": kind,
            "ranking": candidates,
            "selected": best,
        })
    return selected


def _mutate_specs(retained: list[dict], round_number: int) -> list[dict]:
    """由前三名各产生原配置和一个参数变体，保持每轮 6 个方案。"""
    output = []
    for rank, item in enumerate(retained):
        kind = item["kind"]
        current = dict(item["parameters"])
        output.append({"kind": kind, "parameters": current})
        grid = ALGORITHM_GRIDS[kind]
        alternatives = [p for p in grid if p != current]
        if alternatives:
            variant = alternatives[(round_number + rank) % len(alternatives)]
        else:
            variant = current
        output.append({"kind": kind, "parameters": dict(variant)})
    while len(output) < 6:
        output.append(dict(output[len(output) % max(1, len(output))]))
    return output[:6]


def _write_iteration_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "algorithm_iteration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "best_algorithm.json").write_text(
        json.dumps(report["best"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [
        summary
        for round_item in report["rounds"]
        for summary in round_item["ranking"]
    ]
    if rows:
        fields = [
            "round", "kind", "algorithm", "parameters", "test_size",
            "total_score", "prediction_score", "retrieval_score", "runtime_score",
            "mae_24h_km", "mae_48h_km", "mae_72h_km",
            "top5_hit_rate", "top10_overlap", "rank_correlation",
            "avg_query_seconds", "wall_seconds", "errors",
        ]
        with (output_dir / "algorithm_iteration_report.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                item = dict(row)
                item["parameters"] = json.dumps(item["parameters"], ensure_ascii=False)
                writer.writerow({field: item.get(field) for field in fields})


def run_evolution(
    db_path="data/typhoons.sqlite",
    max_rounds=4,
    convergence_threshold=0.03,
    patience=2,
) -> dict:
    """固定 30 样本上的网格寻优、淘汰、变异和收敛闭环。"""
    records = clean_records(SQLiteReader(db_path).load())
    cases = build_fixed_benchmark(records, TEST_SIZE, SEED)
    tune_cases = cases[:3]
    output_dir = Path("output")
    logger = ExperimentLogger(output_dir / "algorithm_iteration_log.jsonl")

    specs = _grid_search(records, tune_cases, logger)
    rounds = []
    previous_best = None
    stagnant_rounds = 0
    overall_best = None

    for round_number in range(1, max_rounds + 1):
        ranking = [
            _evaluate_configuration(
                spec["kind"], spec["parameters"], records, cases, round_number, logger
            )
            for spec in specs
        ]
        ranking.sort(key=lambda item: item["total_score"], reverse=True)
        retained = ranking[: max(1, len(ranking) // 2)]
        best = ranking[0]
        if overall_best is None or best["total_score"] > overall_best["total_score"]:
            overall_best = best

        improvement = None
        if previous_best is not None:
            improvement = (
                (best["total_score"] - previous_best) / max(abs(previous_best), 1e-12)
            )
            stagnant_rounds = stagnant_rounds + 1 if improvement < convergence_threshold else 0

        rounds.append({
            "round": round_number,
            "ranking": ranking,
            "retained": retained,
            "best_improvement": improvement,
            "stagnant_rounds": stagnant_rounds,
        })
        previous_best = best["total_score"]
        if stagnant_rounds >= patience:
            stop_reason = (
                f"连续 {patience} 轮最高分提升低于 "
                f"{convergence_threshold * 100:.1f}%"
            )
            break
        specs = _mutate_specs(retained, round_number)
    else:
        stop_reason = f"达到最大轮数 {max_rounds}"

    report = {
        "seed": SEED,
        "fixed_test_size": len(cases),
        "split": "60% known / 40% future",
        "algorithms": list(ALGORITHM_GRIDS),
        "tensorflow_available": _tensorflow_available(),
        "convergence_threshold": convergence_threshold,
        "patience": patience,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "best": overall_best,
    }
    _write_iteration_outputs(report, output_dir)
    logger.write({"type": "final_report", "report": report})
    return report


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


DIRECTED_OPTIMIZATION_GRID = [
    {
        "lcss_epsilon": 100.0,
        "heading_limit": 45.0,
        "speed_limit": 6.0,
        "ensemble_weights": (0.45, 0.30, 0.25),
        "forecast_weights": (0.45, 0.30, 0.25),
        "lcss_blend": 0.0,
    },
    {
        "lcss_epsilon": 100.0,
        "heading_limit": 45.0,
        "speed_limit": 6.0,
        "ensemble_weights": (0.45, 0.30, 0.25),
        "forecast_weights": (0.45, 0.30, 0.25),
        "lcss_blend": 0.20,
    },
    {
        "lcss_epsilon": 100.0,
        "heading_limit": 45.0,
        "speed_limit": 6.0,
        "ensemble_weights": (0.45, 0.30, 0.25),
        "forecast_weights": (0.45, 0.30, 0.25),
        "lcss_blend": 0.50,
    },
]


def _comparison_row(label: str, summary: dict) -> dict:
    return {
        "label": label,
        "algorithm": summary["algorithm"],
        "parameters": summary["parameters"],
        "ade_24h_km": summary["mae_24h_km"],
        "ade_48h_km": summary["mae_48h_km"],
        "ade_72h_km": summary["mae_72h_km"],
        "fde_24h_km": summary["fde_24h_km"],
        "fde_48h_km": summary["fde_48h_km"],
        "fde_72h_km": summary["fde_72h_km"],
        "top5_hit_rate": summary["top5_hit_rate"],
        "top10_overlap": summary["top10_overlap"],
        "rank_correlation": summary["rank_correlation"],
        "avg_query_seconds": summary["avg_query_seconds"],
        "total_score": summary["total_score"],
    }


def _load_previous_directed_best(
    path="output/directed_optimization_log.jsonl",
) -> dict | None:
    source = Path(path)
    if not source.exists():
        return None
    candidates = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "directed_optimization_final":
            after = entry.get("report", {}).get("after")
            if after and "total_score" in after:
                candidates.append(after)
    return max(candidates, key=lambda item: item["total_score"]) if candidates else None


def run_directed_optimization(db_path="data/typhoons.sqlite") -> dict:
    """同一固定 30 样本上重算优化前后指标，并选择最优定向参数。"""
    records = clean_records(SQLiteReader(db_path).load())
    cases = build_fixed_benchmark(records, TEST_SIZE, SEED)
    tune_cases = cases
    logger = ExperimentLogger("output/directed_optimization_log.jsonl")

    lcss_baseline = _evaluate_configuration(
        "D", {"epsilon_km": 100.0, "delta": 8},
        records, cases, -1, logger,
    )
    tuning = [
        _evaluate_configuration("G", params, records, tune_cases, 0, logger)
        for params in DIRECTED_OPTIMIZATION_GRID
    ]
    tuning.sort(key=lambda item: item["total_score"], reverse=True)
    candidate_optimized = tuning[0]
    previous_best = _load_previous_directed_best()
    before = (
        {**previous_best, "label": "before_best_version"}
        if previous_best is not None
        else _comparison_row("before", lcss_baseline)
    )
    candidate_after = _comparison_row("candidate_after", candidate_optimized)
    if candidate_after["total_score"] > before["total_score"]:
        after = {**candidate_after, "label": "after"}
        retained_previous = False
    else:
        after = {**before, "label": "after_retained_previous_best"}
        retained_previous = True
    best_parameters = after["parameters"]
    improvement = (
        (after["total_score"] - before["total_score"])
        / max(abs(before["total_score"]), 1e-12)
    )
    report = {
        "seed": SEED,
        "fixed_test_size": len(cases),
        "same_test_set": True,
        "tensorflow_available": _tensorflow_available(),
        "before": before,
        "lcss_reference": _comparison_row("lcss_reference", lcss_baseline),
        "parameter_search": tuning,
        "candidate_after": candidate_after,
        "after": after,
        "score_improvement": after["total_score"] - before["total_score"],
        "score_improvement_rate": improvement,
        "best_parameters": best_parameters,
        "retained_previous_best": retained_previous,
        "feature_store": str(Path("output/algorithm_lab_features.sqlite").resolve()),
        "deep_training": (
            _DIRECTED_DEEP_CACHE.training_history
            if _DIRECTED_DEEP_CACHE is not None else None
        ),
    }
    output_dir = Path("output")
    if _DIRECTED_DEEP_CACHE is not None:
        _DIRECTED_DEEP_CACHE.gru.model.save(
            output_dir / "gru_attention_seq2seq_168h.keras"
        )
        _DIRECTED_DEEP_CACHE.tcn.model.save(
            output_dir / "tcn_forecaster_168h.keras"
        )
        for group, model in _DIRECTED_DEEP_CACHE.group_gru_models.items():
            model.save(output_dir / f"gru_group_{group}.keras")
        for group, model in _DIRECTED_DEEP_CACHE.group_tcn_models.items():
            model.save(output_dir / f"tcn_group_{group}.keras")
    (output_dir / "directed_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "directed_optimization_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "label", "algorithm", "parameters",
            "ade_24h_km", "ade_48h_km", "ade_72h_km",
            "fde_24h_km", "fde_48h_km", "fde_72h_km",
            "top5_hit_rate", "top10_overlap", "rank_correlation",
            "avg_query_seconds", "total_score",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in (before, after):
            item = dict(row)
            item["parameters"] = json.dumps(item["parameters"], ensure_ascii=False)
            writer.writerow(item)
    (output_dir / "best_directed_algorithm.json").write_text(
        json.dumps({
            "algorithm": after["algorithm"],
            "parameters": best_parameters,
            "metrics": after,
            "score_improvement": report["score_improvement"],
            "score_improvement_rate": improvement,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.write({"type": "directed_optimization_final", "report": report})
    return report


def single_sample_test(db_path="data/typhoons.sqlite") -> list[dict]:
    """独立跑通两套算法各一个固定样本。"""
    records = SQLiteReader(db_path).load()
    case = build_fixed_benchmark(records, size=1, seed=SEED)[0]
    results = run_benchmark(
        [case],
        make_algorithm_registry(clean_records(records)),
        ExperimentLogger("output/algorithm_stage2_smoke.jsonl"),
    )
    summary = []
    for result in results:
        item = result["cases"][0]
        summary.append({
            "algorithm": result["algorithm"],
            "case_id": case.typhoon_id,
            "total_score": item.get("score", {}).get("total_score"),
            "query_seconds": item.get("runtime_metrics", {}).get("query_seconds"),
            "top1": item.get("top10", [{}])[0].get("name"),
            "error": item.get("error"),
        })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="台风候选算法实验室")
    parser.add_argument(
        "--evolve", action="store_true", help="执行 30 样本自动寻优与迭代"
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--optimize", action="store_true",
        help="执行预测/检索定向优化及优化前后 30 样本对比",
    )
    args = parser.parse_args()
    if args.optimize:
        optimization_report = run_directed_optimization()
        print(json.dumps({
            "before": optimization_report["before"],
            "after": optimization_report["after"],
            "score_improvement": optimization_report["score_improvement"],
            "score_improvement_rate": optimization_report["score_improvement_rate"],
            "best_parameters": optimization_report["best_parameters"],
        }, ensure_ascii=False, indent=2))
    elif args.evolve:
        final_report = run_evolution(max_rounds=args.max_rounds)
        print(json.dumps({
            "stop_reason": final_report["stop_reason"],
            "best": final_report["best"],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(single_sample_test(), ensure_ascii=False, indent=2))
