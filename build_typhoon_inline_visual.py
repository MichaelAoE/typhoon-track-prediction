"""Generate the in-conversation Bavi forecast and analog-path visualization."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import algorithm_lab as lab


WORKSPACE = Path(__file__).resolve().parent
OUTPUT = Path(
    r"C:\Users\Lenovo\.codex\visualizations\2026\07\28"
    r"\019fa66b-0fa1-7b43-b5c4-29eaad7c827a"
    r"\bavi-forecast-analogs.html"
)


def compact_path(points):
    return [[round(float(point.lon), 3), round(float(point.lat), 3)] for point in points]


def main():
    with (WORKSPACE / "bavi.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed = []
    for row in rows:
        try:
            lon, lat = float(row["lon"]), float(row["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        observed.append([round(lon, 3), round(lat, 3)])
    result = json.loads(
        (WORKSPACE / "output" / "bavi_prediction_result_72h.json").read_text(
            encoding="utf-8"
        )
    )
    records = lab.clean_records(
        lab.SQLiteReader(WORKSPACE / "data" / "typhoons.sqlite").load()
    )
    by_id = {record["id"]: record for record in records}
    analogs = []
    for rank, item in enumerate(result["top10"], start=1):
        record = by_id[item["id"]]
        start = int(item.get("comparison_start", item.get("match_start", 0)))
        anchor = int(item.get("forecast_anchor", start + 24))
        end = min(len(record["points"]), anchor + 24)
        analogs.append({
            "rank": rank,
            "id": item["id"],
            "name": item["name"],
            "year": item["year"],
            "similarity": round(float(item["similarity"]), 4),
            "path": compact_path(record["points"][start:end]),
        })
    payload = {
        "observed": observed,
        "prediction": [
            [round(float(lon), 3), round(float(lat), 3)]
            for lon, lat in result["prediction"]
        ],
        "analogs": analogs,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    fragment = f"""<div id="typhoon-path-visual">
  <div class="viz-controls">
    <label class="form-label" for="analog-select">高亮相似台风</label>
    <select class="form-select" id="analog-select" aria-label="选择要高亮的相似台风">
      <option value="-1">全部 Top10</option>
    </select>
  </div>
  <div id="analog-detail" class="text-small text-muted" aria-live="polite">灰色细线为 Top10 相似台风历史路径</div>
  <svg id="typhoon-path-svg" viewBox="0 0 920 570" role="img" aria-labelledby="typhoon-path-title typhoon-path-desc">
    <title id="typhoon-path-title">Bavi 已知路径、72小时预测和 Top10 相似台风路径</title>
    <desc id="typhoon-path-desc">经纬度轨迹图。已知路径为前景实线，预测路径为带圆点的高亮线，相似台风为较细历史轨迹。</desc>
  </svg>
</div>
<style>
  #typhoon-path-visual {{ width: 100%; color: var(--foreground); }}
  #typhoon-path-visual .viz-controls {{ margin-bottom: 0.5rem; }}
  #analog-detail {{ margin-bottom: 0.25rem; }}
  #typhoon-path-svg {{ width: 100%; height: auto; display: block; }}
  #typhoon-path-svg .grid {{ stroke: var(--border); stroke-width: 1; }}
  #typhoon-path-svg .axis {{ stroke: var(--muted-foreground); stroke-width: 1.25; }}
  #typhoon-path-svg .tick-label,
  #typhoon-path-svg .axis-label,
  #typhoon-path-svg .path-label {{ fill: var(--foreground); font-size: 13px; }}
  #typhoon-path-svg .analog {{ fill: none; stroke: var(--muted-foreground); stroke-width: 1.25; opacity: 0.34; }}
  #typhoon-path-svg .analog.is-selected {{ stroke: var(--viz-series-3); stroke-width: 3; opacity: 1; }}
  #typhoon-path-svg .observed {{ fill: none; stroke: var(--foreground); stroke-width: 3.5; }}
  #typhoon-path-svg .forecast {{ fill: none; stroke: var(--viz-series-1); stroke-width: 4; }}
  #typhoon-path-svg .forecast-point {{ fill: var(--viz-series-1); stroke: var(--background); stroke-width: 1.5; }}
  #typhoon-path-svg .legend-text {{ fill: var(--foreground); font-size: 13px; }}
  #typhoon-path-svg .legend-observed {{ stroke: var(--foreground); stroke-width: 3.5; }}
  #typhoon-path-svg .legend-forecast {{ stroke: var(--viz-series-1); stroke-width: 4; }}
  #typhoon-path-svg .legend-analog {{ stroke: var(--muted-foreground); stroke-width: 1.5; }}
</style>
<script>
(() => {{
  const root = document.getElementById("typhoon-path-visual");
  const svg = document.getElementById("typhoon-path-svg");
  const select = document.getElementById("analog-select");
  const detail = document.getElementById("analog-detail");
  const data = {data};
  const NS = "http://www.w3.org/2000/svg";
  const width = 920, height = 570;
  const margin = {{ left: 62, right: 32, top: 24, bottom: 58 }};
  const allPoints = [
    ...data.observed,
    ...data.prediction,
    ...data.analogs.flatMap(item => item.path)
  ];
  const lonValues = allPoints.map(point => point[0]);
  const latValues = allPoints.map(point => point[1]);
  const lonMin = Math.floor(Math.min(...lonValues) / 5) * 5 - 2;
  const lonMax = Math.ceil(Math.max(...lonValues) / 5) * 5 + 2;
  const latMin = Math.floor(Math.min(...latValues) / 5) * 5 - 2;
  const latMax = Math.ceil(Math.max(...latValues) / 5) * 5 + 2;
  const x = lon => margin.left + (lon - lonMin) / (lonMax - lonMin) * (width - margin.left - margin.right);
  const y = lat => height - margin.bottom - (lat - latMin) / (latMax - latMin) * (height - margin.top - margin.bottom);
  const make = (name, attrs = {{}}, text = "") => {{
    const element = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text) element.textContent = text;
    return element;
  }};
  const pathData = points => points.map((point, index) => `${{index ? "L" : "M"}}${{x(point[0]).toFixed(1)}},${{y(point[1]).toFixed(1)}}`).join(" ");
  const ticks = (start, end, count) => Array.from({{ length: count + 1 }}, (_, index) => start + (end - start) * index / count);

  const grid = make("g");
  ticks(lonMin, lonMax, 7).forEach(value => {{
    grid.appendChild(make("line", {{ x1: x(value), x2: x(value), y1: margin.top, y2: height - margin.bottom, class: "grid" }}));
    grid.appendChild(make("text", {{ x: x(value), y: height - margin.bottom + 22, "text-anchor": "middle", class: "tick-label" }}, `${{value.toFixed(0)}}°E`));
  }});
  ticks(latMin, latMax, 6).forEach(value => {{
    grid.appendChild(make("line", {{ x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: "grid" }}));
    grid.appendChild(make("text", {{ x: margin.left - 10, y: y(value) + 4, "text-anchor": "end", class: "tick-label" }}, `${{value.toFixed(0)}}°N`));
  }});
  svg.appendChild(grid);
  svg.appendChild(make("line", {{ x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "axis" }}));
  svg.appendChild(make("line", {{ x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis" }}));
  svg.appendChild(make("text", {{ x: (margin.left + width - margin.right) / 2, y: height - 14, "text-anchor": "middle", class: "axis-label" }}, "经度"));
  svg.appendChild(make("text", {{ x: 17, y: (margin.top + height - margin.bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 17 ${{(margin.top + height - margin.bottom) / 2}})`, class: "axis-label" }}, "纬度"));

  const analogLayer = make("g");
  data.analogs.forEach((item, index) => {{
    const path = make("path", {{
      d: pathData(item.path),
      class: "analog",
      "data-index": index,
      "aria-label": `${{item.rank}}. ${{item.name}} (${{item.year}})，相似度 ${{item.similarity.toFixed(4)}}`
    }});
    analogLayer.appendChild(path);
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${{item.rank}}. ${{item.name}} (${{item.year}}) · ${{item.similarity.toFixed(3)}}`;
    select.appendChild(option);
  }});
  svg.appendChild(analogLayer);
  svg.appendChild(make("path", {{ d: pathData(data.observed), class: "observed" }}));
  svg.appendChild(make("path", {{ d: pathData([data.observed[data.observed.length - 1], ...data.prediction]), class: "forecast" }}));
  data.prediction.forEach(point => svg.appendChild(make("circle", {{ cx: x(point[0]), cy: y(point[1]), r: 4, class: "forecast-point" }})));
  const forecastEnd = data.prediction[data.prediction.length - 1];
  svg.appendChild(make("text", {{ x: x(forecastEnd[0]) + 8, y: y(forecastEnd[1]) - 8, class: "path-label" }}, "72h预测终点"));

  const legend = make("g", {{ transform: `translate(${{margin.left + 12}},${{margin.top + 16}})` }});
  [["legend-observed", "已观测路径"], ["legend-forecast", "72h融合预测"], ["legend-analog", "Top10相似路径"]].forEach((item, index) => {{
    const offset = index * 145;
    legend.appendChild(make("line", {{ x1: offset, x2: offset + 28, y1: 0, y2: 0, class: item[0] }}));
    legend.appendChild(make("text", {{ x: offset + 35, y: 4, class: "legend-text" }}, item[1]));
  }});
  svg.appendChild(legend);

  const update = () => {{
    const selected = Number(select.value);
    root.querySelectorAll(".analog").forEach((path, index) => path.classList.toggle("is-selected", selected === index));
    if (selected < 0) {{
      detail.textContent = "灰色细线为 Top10 相似台风历史路径";
    }} else {{
      const item = data.analogs[selected];
      detail.textContent = `${{item.rank}}. ${{item.name}} (${{item.year}}) · 相似度 ${{item.similarity.toFixed(4)}}`;
    }}
  }};
  select.addEventListener("change", update);
  update();
}})();
</script>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
