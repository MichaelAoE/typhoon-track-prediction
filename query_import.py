"""导入新台风查询。

用法：python query_import.py                 # 自动选择 input 中最新 CSV
      python query_import.py path\to\bavii.csv
程序会把 7 月 10 日前的 Bavi 路径作为输入台风，并输出到 output/。
"""
import csv, sys
from pathlib import Path
from datetime import datetime, timezone
from typhoon_data import TrackPoint, Typhoon, TyphoonDataset
from similarity import find_similar_typhoons
from visualize import plot_similar_typhoons, plot_intensity_compare

def parse_typhoon_from_csv(csv_path):
    path=Path(csv_path); points=[]
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader=csv.reader(f); header=next(reader,None)
        if not header or len(header)<3 or header[0].strip()!="time":
            raise ValueError("CSV 前三列必须为 time, lat, lon；可选 pressure, wind_speed")
        for values in reader:
            try:
                if len(values)<3: continue
                dt=datetime.fromisoformat(values[0].strip().replace("Z","+00:00"))
                if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
                pressure=float(values[3]) if len(values)>3 and values[3].strip() else 1013
                wind=float(values[4]) if len(values)>4 and values[4].strip() else 0
                points.append(TrackPoint(dt,float(values[1]),float(values[2]),pressure,wind,""))
            except (ValueError,TypeError): continue
    if len(points)<2: raise ValueError("有效路径点不足 2 个")
    points.sort(key=lambda p:p.time)
    return Typhoon(path.stem,"导入台风",points[0].time.year,points)

def main(source=None):
    input_dir=Path("input"); output_dir=Path("output"); input_dir.mkdir(exist_ok=True); output_dir.mkdir(exist_ok=True)
    if source is None:
        files=list(input_dir.glob("*.csv"))
        if not files: raise FileNotFoundError("input 文件夹为空，请放入新台风 CSV")
        source=str(max(files,key=lambda p:p.stat().st_mtime))
    target=parse_typhoon_from_csv(source)
    ds=TyphoonDataset.load_from_cache()
    if not ds.typhoons: ds=TyphoonDataset.load_from_folder("data/raw")
    if not ds.typhoons: raise RuntimeError("本地历史台风数据集为空")
    results=find_similar_typhoons(target,ds,10)
    html=output_dir/"result.html"; image=output_dir/"intensity_compare.png"
    plot_similar_typhoons(target,results,str(html)); plot_intensity_compare(target,results,str(image))
    print(f"导入台风: {target.name} | 路径点: {len(target.points)} | 时间: {target.points[0].time} 至 {target.points[-1].time}")
    for i,(t,s) in enumerate(results,1): print(f"{i:2d}. {t.name} ({t.year}) 相似度={s:.4f}")
    print(f"地图: {html.resolve()}\n曲线: {image.resolve()}")

if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else None)
