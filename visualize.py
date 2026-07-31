"""Folium 地图及强度曲线。"""
import folium, matplotlib.pyplot as plt
from folium.plugins import BeautifyIcon
COLORS=["blue","green","purple","orange","darkred","cadetblue","darkgreen","pink"]
def plot_similar_typhoons(target_typhoon,similar_list,output_html="./result.html"):
    c=target_typhoon.points[len(target_typhoon.points)//2]; m=folium.Map([c.lat,c.lon],zoom_start=5,tiles="OpenStreetMap")
    def draw(t,color,score=None,width=3,radius=2):
        folium.PolyLine([(p.lat,p.lon) for p in t.points],color=color,weight=width,tooltip=f"{t.name} 相似度={score:.3f}" if score is not None else t.name).add_to(m)
        for p in t.points: folium.CircleMarker([p.lat,p.lon],radius=radius,color=color,weight=1,fill=True,fill_opacity=.75,tooltip=f"{p.time.isoformat()} | {p.level} | 风速 {p.wind_speed} | 气压 {p.pressure}").add_to(m)
        p=t.points[-1]; folium.Marker([p.lat,p.lon],tooltip=t.name).add_to(m)
    draw(target_typhoon,"black",width=5,radius=3)
    for i,(t,s) in enumerate(similar_list): draw(t,COLORS[i%len(COLORS)],s,2,radius=1.5)
    m.save(output_html); return output_html
def plot_intensity_compare(target_typhoon,similar_list,output_img="./intensity_compare.png"):
    fig,ax=plt.subplots(2,1,figsize=(12,8)); items=[(target_typhoon,1.0)]+list(similar_list[:3])
    for i,(t,s) in enumerate(items):
        x=[p.time for p in t.points]; ax[0].plot(x,[p.wind_speed for p in t.points],label=f"{t.name} ({s:.3f})"); ax[1].plot(x,[p.pressure for p in t.points],label=t.name)
    ax[0].set_ylabel("Wind speed (m/s)"); ax[1].set_ylabel("Pressure (hPa)"); ax[1].invert_yaxis(); [a.legend() for a in ax]; fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output_img,dpi=150); plt.close(fig); return output_img
if __name__=="__main__":
    from typhoon_data import TyphoonDataset
    from similarity import find_similar_typhoons
    ds=TyphoonDataset.load_from_cache()
    if not ds.typhoons: ds=TyphoonDataset.load_from_folder("./data/raw")
    if ds.typhoons:
        target=ds.typhoons[0]; result=find_similar_typhoons(target,ds); plot_similar_typhoons(target,result); plot_intensity_compare(target,result); print("已生成 result.html 与 intensity_compare.png")
    else: print("未找到数据，请先自动下载或将 TXT 放入 ./data/raw")
