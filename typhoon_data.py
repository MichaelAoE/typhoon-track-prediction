"""CMA-STI 最佳路径数据下载、解析与数据集管理。"""
from __future__ import annotations
import csv, json, logging, re, time, sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import requests

# CMA-STI 年度最佳路径文件命名为 CH{year}BST.txt，例如 CH2020BST.txt。
# 也可通过环境变量 CMA_STI_BASE_URL 覆盖此地址，模板需保留 {year}。
BASE_URL = "http://tcdata.typhoon.org.cn/data/CH{year}BST.txt"
RAW_DIR, CACHE_DIR = Path("./data/raw"), Path("./data/cache")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@dataclass
class TrackPoint:
    time: datetime; lat: float; lon: float; pressure: float; wind_speed: float; level: str
    def to_dict(self):
        d = asdict(self); d["time"] = self.time.isoformat(); return d

@dataclass
class Typhoon:
    id: str; name: str; year: int; points: list[TrackPoint]
    def to_dict(self): return {"id": self.id, "name": self.name, "year": self.year, "points": [p.to_dict() for p in self.points]}

def _url(year): return __import__("os").environ.get("CMA_STI_BASE_URL", BASE_URL).format(year=year)

def download_cma_sti_data(start_year, end_year, save_dir="./data/raw"):
    """按年下载；存在文件跳过，成功后解析并写入 JSON 缓存。"""
    folder = Path(save_dir); folder.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        path = folder / f"{year}.txt"
        if path.exists() and path.stat().st_size: logging.info("跳过已存在: %s", path); continue
        for attempt in range(1, 4):
            try:
                r = requests.get(_url(year), timeout=30, stream=True); r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                break
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 404: logging.warning("未找到 %s 数据 (404)", year); break
                logging.warning("下载 %s 失败 (%s/%s): %s", year, attempt, 3, e)
            except requests.RequestException as e: logging.warning("网络异常 (%s/%s): %s", attempt, 3, e)
            if attempt < 3: time.sleep(2 ** (attempt - 1))
    ds = TyphoonDataset.load_from_folder(folder); ds.save_cache(); return ds

def _nums(line): return re.split(r"[\s,]+", line.strip())

def parse_cma_file(path: str | Path) -> list[Typhoon]:
    """兼容常见 CMA 文本：头行含编号/名称，后续行 10 列；跳过无法识别行。"""
    out=[]; current=None
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        s=raw.strip()
        if not s: continue
        parts=_nums(s)
        try:
            if len(parts)>=10 and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", x) for x in parts[:10]):
                y, ident, mo, day, hour, level, lat, lon, pressure, wind = parts[:10]
                if current is None or current.id != ident:
                    current=Typhoon(str(ident), str(ident), int(y), []); out.append(current)
                dt=datetime(int(y), int(mo), int(day), int(hour), tzinfo=timezone.utc)
                current.points.append(TrackPoint(dt, float(lat)/10, float(lon)/10, float(pressure), float(wind), str(level)))
        except (ValueError, IndexError): continue
    return out

def parse_ibtracs_csv(path: str | Path) -> list[Typhoon]:
    """解析 NOAA IBTrACS CSV；风速 WMO_WIND 为 knots，统一转换为 m/s。"""
    groups = {}
    with Path(path).open(encoding="utf-8-sig", newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 第二行是单位说明，且缺少有效 SID/时间，自动跳过。
                sid, stamp = row.get("SID", "").strip(), row.get("ISO_TIME", "").strip()
                if not sid or not stamp or sid == "SID": continue
                lat, lon = float(row["LAT"]), float(row["LON"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180): continue
                dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                def first_number(*keys):
                    for key in keys:
                        value = row.get(key, "").strip()
                        if value and value not in {"NA", "-999", "-999.0"}:
                            try: return float(value)
                            except ValueError: pass
                    return float("nan")
                wind = first_number("WMO_WIND", "USA_WIND", "CMA_WIND", "HKO_WIND") * 0.514444
                pressure = first_number("WMO_PRES", "USA_PRES", "CMA_PRES", "HKO_PRES")
                # 缺失强度值不能参与相似度计算，使用有限的观测点。
                # IBTrACS 早期历史记录可能没有强度观测；保留路径点，使用中性缺省值，避免整条路径丢失。
                if wind != wind: wind = 0.0
                if pressure != pressure: pressure = 1013.0
                name = row.get("NAME", sid).strip() or sid
                if name.upper() in {"NOT_NAMED", "NOT NAMED", "UNNAMED"}: name = sid
                item = groups.setdefault(sid, {"name": name, "year": int(row["SEASON"]), "points": []})
                item["points"].append(TrackPoint(dt, lat, lon, pressure, wind, row.get("USA_SSHS", "").strip() or row.get("NATURE", "")))
            except (ValueError, KeyError, TypeError):
                continue
    return [Typhoon(sid, x["name"], x["year"], sorted(x["points"], key=lambda p: p.time)) for sid, x in groups.items() if x["points"]]

class TyphoonDataset:
    def __init__(self, typhoons=None): self.typhoons=typhoons or []; self._index={t.id:t for t in self.typhoons}
    @classmethod
    def load_from_folder(cls, folder_path):
        all_t=[]
        folder = Path(folder_path)
        for p in sorted(folder.glob("*.txt")): all_t.extend(parse_cma_file(p))
        for p in sorted(folder.glob("*.csv")):
            if "ibtracs" in p.name.lower(): all_t.extend(parse_ibtracs_csv(p))
        return cls(all_t)
    @classmethod
    def load_from_cache(cls, cache_path="./data/cache/typhoons.json"):
        p=Path(cache_path)
        if not p.exists(): return cls([])
        data=json.loads(p.read_text(encoding="utf-8")); ts=[]
        for x in data:
            ts.append(Typhoon(x["id"],x["name"],x["year"],[TrackPoint(datetime.fromisoformat(y["time"]),y["lat"],y["lon"],y["pressure"],y["wind_speed"],y["level"]) for y in x["points"]]))
        return cls(ts)
    def save_cache(self, cache_path="./data/cache/typhoons.json"):
        p=Path(cache_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([t.to_dict() for t in self.typhoons],ensure_ascii=False,indent=2),encoding="utf-8")
    def save_sqlite(self, db_path="./data/typhoons.sqlite"):
        """将台风和路径点导入 SQLite，便于后续查询和增量使用。"""
        p=Path(db_path); p.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(p) as con:
            con.executescript("DROP TABLE IF EXISTS track_points; DROP TABLE IF EXISTS typhoons; DROP TABLE IF EXISTS typhoon_features; CREATE TABLE typhoons (id TEXT PRIMARY KEY,name TEXT,year INTEGER); CREATE TABLE track_points (typhoon_id TEXT,time TEXT,lat REAL,lon REAL,pressure REAL,wind_speed REAL,level TEXT,FOREIGN KEY(typhoon_id) REFERENCES typhoons(id)); CREATE TABLE typhoon_features (typhoon_id TEXT PRIMARY KEY,min_lat REAL,max_lat REAL,min_lon REAL,max_lon REAL,start_lat REAL,start_lon REAL,point_count INTEGER); CREATE INDEX idx_points_id ON track_points(typhoon_id);")
            for t in self.typhoons:
                con.execute("INSERT INTO typhoons VALUES (?,?,?)",(t.id,t.name,t.year))
                con.executemany("INSERT INTO track_points VALUES (?,?,?,?,?,?,?)",[(t.id,p.time.isoformat(),p.lat,p.lon,p.pressure,p.wind_speed,p.level) for p in t.points])
                lats=[p.lat for p in t.points]; lons=[p.lon for p in t.points]; p=t.points[0]
                con.execute("INSERT INTO typhoon_features VALUES (?,?,?,?,?,?,?,?)",(t.id,min(lats),max(lats),min(lons),max(lons),p.lat,p.lon,len(t.points)))
            con.commit()
        return str(p)
    def get_by_id(self, typhoon_id): return self._index.get(str(typhoon_id))
    def filter_by_bbox(self,min_lat,max_lat,min_lon,max_lon): return [t for t in self.typhoons if any(min_lat<=p.lat<=max_lat and min_lon<=p.lon<=max_lon for p in t.points)]

def haversine(lat1,lon1,lat2,lon2):
    import math
    r=6371.0088; a=math.sin(math.radians(lat2-lat1)/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*r*math.asin(math.sqrt(a))
