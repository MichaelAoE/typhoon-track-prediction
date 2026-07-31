"""相似台风查询演示：历史台风前半段模拟一条新台风路径。"""
from pathlib import Path
from typhoon_data import TyphoonDataset, Typhoon
from similarity import find_similar_typhoons
from visualize import plot_similar_typhoons, plot_intensity_compare

def build_demo_target(history: Typhoon, fraction=0.5):
    n=max(2,int(len(history.points)*fraction))
    return Typhoon("NEW_QUERY", f"新台风查询（参考{history.name}）", history.year, history.points[:n])

def run(dataset, source_id=None, top_n=10):
    source=dataset.get_by_id(source_id) if source_id else max(dataset.typhoons,key=lambda t:len(t.points))
    target=build_demo_target(source)
    results=find_similar_typhoons(target,dataset,top_n)
    plot_similar_typhoons(target,results,"./query_result.html")
    plot_intensity_compare(target,results,"./query_intensity.png")
    print(f"查询路径: {target.name}，使用 {len(target.points)} 个点")
    for i,(t,s) in enumerate(results,1): print(f"{i:>2}. {t.name} ({t.year})  相似度={s:.4f}  ID={t.id}")
    return results

if __name__=="__main__":
    db=Path("./data/typhoons.sqlite")
    ds=TyphoonDataset.load_from_cache()
    if not ds.typhoons: ds=TyphoonDataset.load_from_folder("./data/raw")
    ds.save_sqlite(db)
    run(ds)
