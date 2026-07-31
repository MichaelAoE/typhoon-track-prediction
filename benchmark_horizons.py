"""在同一模型实例中公平测试 72/120/168h 的单台风检索耗时。"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import algorithm_lab as lab
from query_import import parse_typhoon_from_csv


def benchmark(csv_path="bavii.csv", repeats=2) -> dict:
    source = parse_typhoon_from_csv(csv_path)
    known = [
        lab.TrackPoint(
            point.time.isoformat(),
            point.lat,
            point.lon,
            point.pressure,
            point.wind_speed,
        )
        for point in source.points
    ]
    records = lab.clean_records(lab.SQLiteReader().load())
    algorithm = lab.DirectedOptimizedAlgorithm(records)

    # 首次 TensorFlow 推理单独热身，不计入检索耗时。
    algorithm.predict_new_typhoon(known, predict_hours=72)
    timings = {}
    for hours in (72, 120, 168):
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            algorithm.predict_new_typhoon(known, predict_hours=hours)
            samples.append(time.perf_counter() - started)
        timings[str(hours)] = {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "mean_seconds": statistics.mean(samples),
        }
    baseline = timings["72"]["median_seconds"]
    for item in timings.values():
        item["increase_vs_72h_percent"] = (
            item["median_seconds"] / baseline - 1.0
        ) * 100.0
    report = {
        "sample": csv_path,
        "method": "single loaded model, one warmup, repeated median",
        "repeats": repeats,
        "timings": timings,
        "limit_percent": 30.0,
        "passed": all(
            item["increase_vs_72h_percent"] <= 30.0
            for item in timings.values()
        ),
    }
    output = Path("output/bavii_horizon_performance.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
