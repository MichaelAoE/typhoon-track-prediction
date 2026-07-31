# 相似台风查询系统

完成 CMA-STI 最佳路径数据的下载、TXT 解析、JSON 缓存、DTW 多维相似度检索，以及 Folium 地图和强度曲线输出。

## 安装
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 数据
自动下载前，在 `typhoon_data.py` 修改 `BASE_URL` 为 CMA 官网实际年度直链（需包含 `{year}`），或设置环境变量 `CMA_STI_BASE_URL`。然后：
```bash
python -c "from typhoon_data import download_cma_sti_data; download_cma_sti_data(2000,2020)"
```
网络受限时，可将标准 TXT 文件手动放入 `./data/raw`，再运行：
```bash
python -c "from typhoon_data import TyphoonDataset; d=TyphoonDataset.load_from_folder('./data/raw'); d.save_cache()"
```

## 运行演示
```bash
python visualize.py
```
生成 `result.html` 交互地图和 `intensity_compare.png`。检索入口为 `find_similar_typhoons(target, dataset, top_n=10)`；粗筛扩大范围、权重和 Top-N 可在 `similarity.py` 调整。

## 数据库与新台风查询
运行 `python query_demo.py` 会把 CSV 解析结果导入 `data/typhoons.sqlite`，并选择历史记录最长的台风，将其前 50% 路径模拟为“新台风路径”，输出 Top-N 相似台风和带名称的查询结果：
```bash
python query_demo.py
```
结果文件为 `query_result.html` 和 `query_intensity.png`。实际业务中，将实时台风路径转换成 `TrackPoint` 列表，再构造 `Typhoon("NEW", "待命名台风", year, points)` 传给 `find_similar_typhoons` 即可。

TXT 每个路径点应为：年份、编号、月份、日期、小时、等级、纬度×10、经度×10、气压、风速。时间按 UTC 解析。若不同数据发布版本含名称头信息，解析器会保留编号作为默认名称，可在对象上修改 `name`。

## 长时效预测与加权检索

`algorithm_lab.search_similar_typhoon` 保持默认 72 小时行为，并通过
`predict_hours` 支持 72、120、168 小时三档预测：

```python
from algorithm_lab import search_similar_typhoon

result = search_similar_typhoon(points, predict_hours=120)
```

使用 `bavii.csv` 生成三档路径、Top10 结果和分段着色地图：

```bash
python predict_new_typhoon.py bavii.csv 72
python predict_new_typhoon.py bavii.csv 120
python predict_new_typhoon.py bavii.csv 168
python horizon_report.py
```

地图中 0–72h 为红色实线、72–120h 为橙色虚线、120–168h 为紫色点虚线。
结果 JSON 包含分时效置信度、加权/未加权相似度、近期移向一致性因子和
Top5 相似台风校正信息。参数集中记录于
`output/best_long_horizon_config.json`，三档实测对比见
`output/bavii_horizon_comparison.md`。

## Bavi 72 小时严格无泄漏优化

`bavi.csv` 同时含 3 小时和 1 小时间隔数据，而旧模型的原生时间步为
3 小时。严格流程会先统一到 3 小时网格，再自动选择“仍保留完整 72 小时
真值的最晚原始观测时刻”作为最终留出断点。预测模型和检索参数只使用
最终断点至少 72 小时以前的滚动样本选择。

```bash
python bavi_rollback_optimize.py
python predict_new_typhoon.py bavii.csv 72
```

第一条命令会重建断点前的 `bavii.csv`，比较约 60.13 分历史方案、
时间网格修正、统计/转向/加速度外推、历史相似路径集合及验证集融合，
并生成严格预测—真值对比图、交互地图、Top10 相似台风和 JSON 报告。
第二条命令验证平台标准入口已经恢复历史约 60.13 分方案。

默认 72 小时模式为历史 GRU + TCN + 惯性外推三模型集合，权重
0.45/0.30/0.25；120/168 小时模式保持原长时效实现。最终断点后的真实路径
只在预测和排名全部冻结后用于计算误差，禁止参与参数选择、残差校正或路径
平移，也不会生成任何“真值拟合路径”。严格报告和参数分别见
`output/bavi_strict_optimization_report.json` 与
`output/best_bavi_strict_config.json`。

## 预测—相似路径反馈融合

运行数据库固定样本的融合寻优：

```bash
python analog_fusion_optimize.py
```

算法先生成基础预测，再根据断点前路径和基础预测筛选历史相似台风，提取
Top-K 后续相对位移形成集合预报，并按时效与基础预测融合。历史目标台风
自身会从候选库排除；真实未来路径只在预测和排序冻结后用于评分。

使用通过独立留出集验证的融合配置查询完整 Bavi：

```bash
python predict_new_typhoon.py bavi.csv 72 analog_fusion
```

当前融合模式只支持 72 小时。72/120/168 小时原有模式仍可通过
`score61_restored`、`strict_validation` 或默认长时效入口调用。最优融合
参数见 `output/best_analog_fusion_config.json`，完整回测结果见
`output/analog_fusion_optimization_report.json`。

执行三轮“每轮一场新台风”的顺序参数优化：

```bash
python three_round_iterative_optimize.py
```

每轮从数据库选择一场未用于上一轮的台风，截取前60%路径预测未来72小时，
并搜索通用参数。单台风候选只有在固定10台风审计集同步改善时才会写入最优
配置，避免个例过拟合。三轮详情见
`output/three_round_iteration_report.json`。
