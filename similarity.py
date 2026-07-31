"""DTW 与两级相似台风检索。"""
import numpy as np
from typhoon_data import haversine
try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

WEIGHTS = {"path": .90, "wind": .07, "pressure": .03}
DISTANCE_SCALE = 200.0
_NORMS = None
_ACTIVE_DATASET = None

def dtw_distance(seq1, seq2, dist_func):
    n,m=len(seq1),len(seq2); d=np.full((n+1,m+1),np.inf); d[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1): d[i,j]=dist_func(seq1[i-1],seq2[j-1])+min(d[i-1,j],d[i,j-1],d[i-1,j-1])
    return float(d[n,m])

def calc_dtw_distance(seq1, seq2, mode="path"):
    """FastDTW 平均距离；缺少 fastdtw 时回退到标准动态规划。"""
    if len(seq1)==0 or len(seq2)==0: return float("inf")
    if mode == "path":
        dist=lambda x,y: haversine(float(x[0]),float(x[1]),float(y[0]),float(y[1]))
    else:
        dist=lambda x,y: abs(float(np.asarray(x).ravel()[0])-float(np.asarray(y).ravel()[0]))
    if fastdtw is not None: raw,_=fastdtw(seq1,seq2,dist=dist,radius=5)
    else: raw=dtw_distance(seq1,seq2,dist)
    return float(raw/max(len(seq1),len(seq2)))

def _as_record(t):
    if isinstance(t,dict): return t
    return {"typhoon_id":t.id,"path":[[p.lon,p.lat] for p in t.points],"wind":[p.wind_speed for p in t.points],"pressure":[p.pressure for p in t.points]}

def set_global_norms(dataset):
    global _NORMS
    winds=[p.wind_speed for t in dataset.typhoons for p in t.points if np.isfinite(p.wind_speed)]
    press=[p.pressure for t in dataset.typhoons for p in t.points if np.isfinite(p.pressure)]
    _NORMS=((min(winds),max(winds)),(min(press),max(press))); return _NORMS

def calc_typhoon_similarity(target_ty,candidate_ty,wind_norm=None,pres_norm=None):
    """按指定 0.75/0.15/0.10 权重返回详细距离和综合相似度。"""
    global _NORMS
    if wind_norm is None or pres_norm is None:
        if _NORMS is None: raise ValueError("请先调用 set_global_norms(dataset)")
        wind_norm,pres_norm=_NORMS
    a,b=_as_record(target_ty),_as_record(candidate_ty)
    def norm(x,r):
        lo,hi=r; return (np.asarray(x,float)-lo)/(hi-lo) if hi>lo else np.zeros(len(x))
    d_path=calc_dtw_distance(a["path"],b["path"],"path")
    d_wind=calc_dtw_distance(norm(a["wind"],wind_norm).reshape(-1,1),norm(b["wind"],wind_norm).reshape(-1,1),"scalar")
    d_pres=calc_dtw_distance(norm(a["pressure"],pres_norm).reshape(-1,1),norm(b["pressure"],pres_norm).reshape(-1,1),"scalar")
    total=WEIGHTS["path"]*d_path+WEIGHTS["wind"]*d_wind*500+WEIGHTS["pressure"]*d_pres*500
    return {"similarity":float(np.exp(-total/DISTANCE_SCALE)),"path_dist_km":d_path,"wind_dist":d_wind,"pressure_dist":d_pres}
def _norm(xs):
    a=np.asarray(xs,float); lo,hi=np.nanmin(a),np.nanmax(a); return (a-lo)/(hi-lo) if hi>lo else np.zeros_like(a)

def _resample_points(points, count=48):
    """按路径累计距离重采样，使不同观测时次/点数可比较。"""
    if len(points) <= 1: return points * count
    xy=np.array([[p.lat,p.lon] for p in points],float)
    seg=np.array([haversine(*xy[i],*xy[i+1]) for i in range(len(xy)-1)])
    cum=np.r_[0,np.cumsum(seg)]
    if cum[-1] == 0: return [points[0]]*count
    grid=np.linspace(0,cum[-1],count); out=[]
    for x in grid:
        j=min(np.searchsorted(cum,x),len(points)-1)
        if j==0: out.append(points[0]); continue
        f=(x-cum[j-1])/max(cum[j]-cum[j-1],1e-12)
        a,b=points[j-1],points[j]
        from typhoon_data import TrackPoint
        out.append(TrackPoint(a.time+(b.time-a.time)*f,a.lat+(b.lat-a.lat)*f,a.lon+(b.lon-a.lon)*f,a.pressure+(b.pressure-a.pressure)*f,a.wind_speed+(b.wind_speed-a.wind_speed)*f,a.level))
    return out

def _bearing(a,b):
    import math
    dlon=math.radians(b.lon-a.lon); la1,la2=map(math.radians,(a.lat,b.lat))
    return (math.degrees(math.atan2(math.sin(dlon)*math.cos(la2),math.cos(la1)*math.sin(la2)-math.sin(la1)*math.cos(la2)*math.cos(dlon)))+360)%360

def path_similarity(a,b,count=48):
    """论文路径指标的可计算实现：位置偏差 + 方向偏差 + 转折形状偏差。"""
    pa,pb=_resample_points(a.points,count),_resample_points(b.points,count)
    pos=np.mean([haversine(x.lat,x.lon,y.lat,y.lon) for x,y in zip(pa,pb)])
    ba=[_bearing(pa[i],pa[i+1]) for i in range(count-1)]; bb=[_bearing(pb[i],pb[i+1]) for i in range(count-1)]
    ang=np.mean([abs((x-y+180)%360-180) for x,y in zip(ba,bb)])/180
    turn_a=np.diff(np.unwrap(np.radians(ba))); turn_b=np.diff(np.unwrap(np.radians(bb)))
    shape=np.mean(np.abs(turn_a-turn_b)) if len(turn_a) else 0
    # 指标均为距离型，指数映射到 [0,1]；位置使用 1000 km 作为稳健尺度。
    return float(np.exp(-(0.65*min(pos/1000,3)+0.2*ang+0.15*min(shape,3))))
def calculate_similarity(a,b):
    if not a.points or not b.points: return 0.0
    # 新版 FastDTW 多因子内核；兼容旧接口仍只返回分数。
    if _ACTIVE_DATASET is None:
        winds=[p.wind_speed for t in (a,b) for p in t.points]; press=[p.pressure for t in (a,b) for p in t.points]
        norms=((min(winds),max(winds)),(min(press),max(press)))
    else: norms=_NORMS
    return calc_typhoon_similarity(a,b,*norms)["similarity"]
    if not a.points or not b.points: return 0.0
    pa,pb=a.points,b.points
    pos=path_similarity(a,b)
    pressure=dtw_distance(_norm([x.pressure for x in pa]),_norm([x.pressure for x in pb]),lambda x,y:abs(x-y))/max(len(pa),len(pb),1)
    wind=dtw_distance(_norm([x.wind_speed for x in pa]),_norm([x.wind_speed for x in pb]),lambda x,y:abs(x-y))/max(len(pa),len(pb),1)
    month=abs(a.points[0].time.month-b.points[0].time.month)/11
    # 当前 IBTrACS 无环境场：路径占主权重，强度与月份作为辅助灾害特征。
    return max(0.,min(1.,.70*pos+.15*(1-min(pressure,1))+.10*(1-min(wind,1))+.05*(1-month)))
def find_similar_typhoons(target,dataset,top_n=10):
    global _ACTIVE_DATASET
    _ACTIVE_DATASET=dataset; norms=set_global_norms(dataset)
    lats=[p.lat for p in target.points]; lons=[p.lon for p in target.points]
    candidates=dataset.filter_by_bbox(min(lats)-2,max(lats)+2,min(lons)-2,max(lons)+2)
    t0=target.points[0]; tc_lat=(min(lats)+max(lats))/2; tc_lon=(min(lons)+max(lons))/2
    import math
    def rough(t):
        if t.id==target.id or not t.points: return False
        p=t.points; c=((min(x.lat for x in p)+max(x.lat for x in p))/2,(min(x.lon for x in p)+max(x.lon for x in p))/2)
        ratio=len(p)/max(len(target.points),1)
        return haversine(tc_lat,tc_lon,c[0],c[1])<=1000 and haversine(t0.lat,t0.lon,p[0].lat,p[0].lon)<=1500 and .5<=ratio<=2
    candidates=[t for t in candidates if rough(t)]
    scored=[]
    for t in candidates:
        detail=calc_typhoon_similarity(target,t,*norms); scored.append((t,detail["similarity"],detail))
    scored.sort(key=lambda x:x[1],reverse=True)
    return [(t,s) for t,s,_ in scored[:top_n]]
