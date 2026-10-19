# 避障与可见性约束下的目标中心虚拟视角规划

状态：首版已实现，真实场景 CUDA 验证待完成。本文原设计依据本地提交 `00004a0`；以下保留完整设计目标，不能把其中的可选扩展一并当作已实现。实际接口见 [README](README.zh-CN.md#避障虚拟相机collision_aware)。

首版实现差异：自由空间用真实相机高 alpha 深度与 raw mesh 首交点交叉支持的**重叠深度片区凸视锥并集**，加经几何校验的真实相机原点邻域，不构建大体素阵列。全部片区像素须有效，不插值稀疏射线；查询保护球须完全包含于单个可信凸域，连续边须递归认证，因此宁可拒绝边界模糊的连接。路径绕障依赖认证候选图，保留每一个转折，**尚无连续空间 A***。候选含多半径方向、稠密轨道带和真实相机邻域；可见面积覆盖基于语义面片的确定性面积采样。同后端低分辨率预渲染已接通，完整场景和全分辨率画质验收仍待实测。

本次实现验证（2026-10-19）：

- Open3D 0.20 / Python 3.12：`test_view_planning.py` + `test_view_planning_integration.py`，18 passed、1 skipped；跳过项为该隔离环境未安装 Torch 的 CPU 相机重建测试，该项已在下一个环境通过。
- Python 3.10 / Torch 2.6：`test_virtual_camera_paths.py`、`test_virtual_render.py`、`test_paintmesh_normal.py` 以及新增精确相机重建项，68 passed、3 skipped、2 deselected。GPU smoke 未启用；2 项符号链接测试在最终针对性运行中排除。
- 首次包含 workspace/symlink 的较广运行出现 4 项 `WinError 1314`（系统未授予符号链接权限），不记为通过；未更改系统权限或用复制替代 symlink 掩盖失败。
- 合成房间测试使用真实三角网格查询和解析首交点图像替代 CUDA 预渲染，验证发布、下游读取、原样复用和网格变化后的拒绝复用；这不验证真实 Gaussian 后端画质。
- `run_remove.sh` 的 `bash -n`、修改 Python 的语法编译和 `git diff --check` 通过。当前 Windows CUDA 环境虽可用 Torch，但缺少两个项目自定义光栅化扩展，未进行真实场景全流程。

## 1. 目标与当前缺口

目标：从同源语义 3DGS、mesh 和真实训练相机出发，在完整场景约束下选择恰好 N 个虚拟观察位姿，保留有效目标视野和补全上下文，并验证相机及其连续路径与已知几何保持安全间隔。输出精确相机 manifest，继续驱动 full/removed 同相机渲染、掩码传播和 RGB/depth/normal 补全。

“最优”在本方案中指有限候选集和计算预算内的约束优化结果，不承诺全局最优，也不把虚拟渲染视角当作新增真实几何观测。

当前事实：

1. `utils/pose_utils.py::generate_virtual_path()` 仅支持 circle/hemisphere；位置依据训练相机 PCA、共同半径及焦点生成，没有场景碰撞检测。
2. `focus_point_fn()` 的输入是训练相机，生成焦点不等于选中实例的中心。`target_object_radius` 只影响半径；多个 selected 对象时还可能包含临时移除的 surrounding 对象。
3. `tools/virtual_pose.py::virtual()` 在 `--poses-only` 路径写出精确相机；`scripts/paintmesh/render_virtual_views.py` 随后分派 native/PGSR。
4. `render_virtual_worker.py::model_files()` 的补全输入是 target + surrounding 已移除的模型；完整场景和最终 target-only 移除模型具有不同用途。
5. schema 1/2 的校验器显式限制 circle/hemisphere。`paintmesh_normal.py::read_cameras()` 还重复计算相机身份，不能只修改生成入口。

## 2. 三类约束分别处理

### 2.1 相机位置与近裁剪面安全

相机中心处在空处，不代表近裁剪面没有切入墙壁。为虚拟相机构建保守的球形保护体，其半径涵盖配置的相机体积、近裁剪面四角和额外几何间隔。

令 `c` 为实际渲染坐标中的相机中心，`r_near` 为近裁剪面角点到 c 的最大距离：

```text
r_safe = max(r_body, r_near) + clearance
distance(c, full_collision_mesh) >= r_safe
```

near/far 不是障碍剔除工具。不能通过增大 near 裁掉墙面，把穿模相机判为安全。只有相机附近的保护体必须无碰撞；整个远端视锥本来就应与被观察表面相交，不能要求完整视锥无障碍。

### 2.2 连续路径安全

逐帧安全不代表两帧间安全。禁止对两个安全端点直接线性插值后默认通过。相邻位姿间的路径必须检查完整扫掠体，并且不得在最终重采样时抹掉绕障转折。

### 2.3 目标可见、构图与序列连续

相机没有碰撞时，目标仍可能被柜子遮挡、被画面裁掉，或者占满画面导致缺少补全上下文。单帧必须检查目标可见比例、画面占比、边缘留白和周边有效背景。序列还必须限制相邻位移、旋转和目标共视变化，为现有 tracking 提供连续输入。

几何安全是硬约束，目标覆盖与画质是选择目标。高覆盖分数不能抵消一次已知几何碰撞。

## 3. 输入、几何角色与坐标

### 3.1 需要绑定的输入

- 原始完整语义 Gaussian PLY、classifier、scene.json、bridge/semantic manifest。
- 真实训练相机及原始图像尺寸、内参、R/T、trans/scale。
- 原始完整 mesh，优先使用同源 EDGS `mesh/ours_<T_base>/tsdf_fusion.ply`。
- 同源后处理 mesh 及其 vertex/face instance sidecars，用于目标表面和邻接区域识别。
- target_ids、surrounding_ids、实际移除配置及其源文件身份。
- 已存在的逐物体 PLY：`work_model/point_cloud_object_removal/iteration_<T_sem>/point_cloud_<id>.ply`，用于核对实际删除包络及孤立目标渲染。

碰撞 mesh 使用完整场景所有几何，包括目标、surrounding、背景和 unknown 语义区域。unknown 标签不等于空闲空间。不能用 removed mesh 作为原场景避障依据。

`tsdf_fusion_post.ply` 的小组件可能在后处理中被删除；碰撞建模优先保留 raw mesh。若只存在 post mesh，记录来源不足；严格模式要求重建同源 raw mesh 后再规划，不静默把丢失组件当成空地。

raw 和 post mesh 的面索引不能混用。语义 sidecar 只绑定其原始 mesh；raw 碰撞查询不依赖语义索引。为了性能创建的代理 mesh 不得无检查地简化掉薄墙或桌腿，最终安全检查使用完整碰撞几何。

### 3.2 目标中心与实际编辑区域

以 target_ids 对应的高可信 Gaussian/语义面片计算目标中心和边界。surrounding 只参与完整场景障碍与其既有移除用途，不并入主目标中心。

语义目标包络与实际删除包络分开：前者用于构图和实例可见性；后者取已保存的目标 PLY，用于补全区域范围。两者差异须报告，因为实际移除包含概率筛选和凸包扩张。

少量离群语义点可通过确定性的连通性/支持分析识别，但不得默默只保留最大组件而丢弃有效物体部分。组件歧义或包络异常应报错并输出诊断，允许显式指定目标组件或 ROI。

首版聚焦单个 target ID；原模式的多目标功能保持可用。多目标规划后续按每个实例分别定义覆盖与最低可见性，不能简单取 union 中心后声称每个对象都有效。

### 3.3 明确一个碰撞坐标系

全部 mesh、Gaussian 中心、路径和距离在同一渲染世界坐标中计算。PCA 只用于候选方向与场景 up 的估计，不作为不声明的距离归一化。

当前 manifest 的 R 允许带统一相似尺度；必须用精确世界—相机变换和 trans/scale 适配获得实际相机中心、近裁剪面角点和世界射线。不能随意将其投影到 SO(3) 后丢失尺度。平面 z-depth 也必须经完整反投影变为世界坐标，不能直接与 raycast 的射线距离相减。

默认 clearance 等空间参数按目标半径或 mesh 分辨率归一化。没有场景尺度标定时，参数与报告使用场景单位，不写成厘米或米。

## 4. 规划流程

```text
完整 mesh + 原始 3DGS + 真实相机 + target ID
       ↓
目标 ROI / 完整障碍场 / 已观测自由空间
       ↓
目标中心、多距离、真实视角邻域候选
       ↓
相机保护体 + 构图 + 遮挡筛选
       ↓
具有已验证边的视角图
       ↓
覆盖收益与 tracking 连续性约束下选择 N 帧
       ↓
低分辨率高斯预渲染、必要的有限重规划
       ↓
冻结精确相机与安全诊断
       ↓
原 full / removed 渲染 → tracking → RGB/D/N completion
```

### A. 完整场景查询与已观测自由空间

使用 Open3D `RaycastingScene` 的 unsigned distance 和 first-hit raycast。TSDF 提取网格不一定水密，不能默认 `compute_occupancy()` 或 signed distance 能正确判断墙内/墙外、物体内部。

在目标及候选周围建立有内存上限的局部三维网格，区分 `occupied / observed_free / unknown`。已知表面及其保护带为 occupied。由真实训练相机到可信首个表面之前的视线区间形成 observed_free；穿过首个表面后的空间和无有效深度区域保留 unknown。

可信终点由训练视角下完整模型的 PGSR 深度、alpha 与原始碰撞几何交叉核对得到。对深度断层、低 alpha、mesh/深度明显不一致的射线不做自由空间外推；不通过插值填补稀疏射线之间未经证实的空域。帧采样、网格离散和冲突处理属于算法配置，记录实际证据覆盖。重建伪影仍会影响该估计，不能称为物理空间真值。

首版严格模式把 unknown 当作不可规划，并只在通过安全校验的真实相机邻域所连接的自由空间组件中选择路径。这样避免在外墙另一侧、封闭柜体内部或未观测区域找到表面距离很大的“伪安全点”。真实相机中心也是待检查的锚点，并非天然通过。

可选 Gaussian 支撑包络可以补充薄结构障碍，但透明度并非占据概率，不能据此清除 mesh 障碍或把遮挡后区域标为空闲。首版不依赖该可选项获得通过。

### B. 目标中心的多距离候选

1. 从目标中心生成等面积方向候选，限制过高仰角与 scene-up 退化方向。
2. 每个方向采样多个距离，允许在障碍附近改变半径；加入真实相机位置及局部扰动候选。
3. 所有候选朝向目标 ROI，roll 固定锚定 scene up。第一帧从安全且清楚观察目标的候选中选取，不沿用旧圆环首帧作为新模式约束。
4. 初始距离可使用目标尺度和 FoV 得到先验，最终以实际投影包络和留白判定，不能把球半径除以 tan(FoV/2) 当作严格构图证明。
5. 既有 FoV、分辨率和投影约定保持一致，不用临时扩大视角或改 near 让不合格候选通过。

候选数、表面探针数、局部网格上限和有限扩展轮数显式配置。先做位置/构图等便宜过滤，再进行批量射线查询，最后只对少量候选做 GPU 预渲染。首版不引入需要训练的视角评分网络。

### C. 遮挡与补全效用

对目标 mesh 的面积加权表面样本发射单位方向射线。已知样本只有在落入画面、法向夹角合适、且完整场景的首个交点与其吻合时才算可见。自遮挡和其他对象遮挡均由 first-hit 处理；单独测试“中心到中心没有相交”不够。

mesh/语义不足以表达的目标区域，用已保存的孤立目标 Gaussian 和完整 Gaussian 做低分辨率预渲染复核。完整场景的可见目标贡献与孤立目标轮廓比较，估计外部遮挡；不能把完整轮廓、预测标签或 alpha 任何一项单独当作真实可见性证明。native/PGSR 的预渲染必须通过同一后端适配入口执行，不将两个渲染器的深度定义混用。

选择效用包含：

- 可见目标面积与未覆盖方向带来的新增收益；
- 实际删除区域的投影范围和边界上下文；
- 已知周边背景的可见支持，以及 RGB 匹配所需的视差与适度共同视野；
- 相对真实相机观测范围的外推代价、重复视角惩罚和路径长度。

补全洞内 removed alpha 低是预期现象，不能因此否决候选。上下文有效性只在洞外的邻接区域评估。完全不可见的物体背面/背景没有真值，覆盖指标只针对已重建表面或明确的几何代理，不能把扩大虚拟视角数量算作真实信息增益。

可将覆盖写为饱和形式：

```text
Coverage(S) = sum_k area_k * max_{v in S} quality(v, k)
J(S, route) = Coverage_target
              + lambda_context * Coverage_context
              + lambda_pair * Useful_pair_support
              - lambda_length * Route_length
              - lambda_extrapolation * Extrapolation_cost
```

硬约束为相机/路径安全、已观测自由空间、目标最低可见性、构图留白、相邻姿态变化以及恰好 N 个不同位姿。多项组合后的序列目标不直接继承简单最大覆盖贪心算法的近似保证。

### D. 可达视角图与 N 帧序列

对通过单帧筛选的候选建立近邻图，只连接几何安全且相邻画面变化可接受的边。视角图不是按方位角排序的列表。

直线边采用 unsigned distance 的 1-Lipschitz 下界做自适应细分。对长度为 L 的线段，中点 m 满足：

```text
distance(m, mesh) - L/2 >= r_safe
```

时，该子段的球形相机扫掠体可通过已知 mesh 的距离检查；未满足时继续细分，发现碰撞立即拒绝，达到最小分辨率仍无法确认时标为 unresolved 并拒绝。该保守界需要计入数值误差余量。自由空间网格同时检查 supercover cells；中心格点为空不能代表整条边可通行。

相邻姿态之间采用明确的分段路径模型。近裁剪面保护采用姿态无关的保守球包络，因而上面的检查覆盖旋转过程；若以后改成紧致定向保护体，必须增加其姿态扫掠检查。

当局部直线图被障碍阻断，可在局部 observed_free 网格中对少量高价值连接执行 A*。A* 的折线节点必须重新计算目标朝向、构图、可见性和邻接共视；离开目标有效视野的绕行不能直接交给单序列 tracker。平滑只允许替换已重新验证的局部路径。

采用确定性的多起点 beam search 或有限视野搜索，在图上同时决定覆盖与路线，避免先挑分散的最高分视角、再发现无法连接。选择局部可用连通组件；墙背区域不可达时，不强求 360° 环绕。

最终帧数仍由 `VIRTUAL_CAMERA_COUNT=N` 决定。转折、过渡帧计入 N；恰好 N 帧，名称仍是 `00000..N-1`。输出连续路径及必要转折的单独证据。不能直接按等弧长重采样丢掉拐点，也不能静默插入第 N+1 帧或用重复相机补足预算。

如果帧数、可见性、最低覆盖和安全条件不能同时满足，输出不可行原因，不生成 `complete=true`。可以显式增大 N 或调整非安全质量目标后新建 run；不自动减小 clearance、放开 unknown 或回退未检查的圆环。

### E. 渲染复核与冻结

在最终发布相机之前，以选定后端做低分辨率复核，检查目标可见性、洞外上下文、有效几何和近处遮挡。mesh 与 Gaussian 表现不一致时丢弃候选并进行有上限的重规划，不在相机发布后修改个别帧。

最终完整分辨率若发现质量门槛失败，应保留诊断且不提交 tracking/completion 的完成状态；重新规划须写新身份并作废下游相机绑定。旧 circle/hemisphere 的帧集合与模型来源语义保持原样。

## 5. 代码落点与进程边界

新增文件：

| 文件 | 职责 |
|---|---|
| `scripts/paintmesh/plan_virtual_views.py` | 独立规划 CLI，按冻结输入生成相机与诊断 |
| `scripts/paintmesh/view_planning_io.py` | 配置、输入身份、上下文和规划结果契约 |
| `scripts/paintmesh/view_planning_geometry.py` | 完整 mesh、UDF/raycast、自由空间、保护体与路径检查 |
| `scripts/paintmesh/view_planning_candidates.py` | 目标定位、多距离候选、look-at 和精确投影 |
| `scripts/paintmesh/view_planning_selection.py` | 覆盖、可达图、序列选择及有限重规划 |
| `scripts/paintmesh/configs/virtual_view_planning.yaml` | 资源预算、距离归一化、安全及质量参数 |
| `scripts/paintmesh/tests/test_view_planning_*.py` | 合成几何、序列、契约和端到端适配测试 |

修改入口：

| 当前文件 | 具体改动 |
|---|---|
| `submodules/Inpaint360GS/tools/virtual_pose.py` | 增加只导出精确训练相机/模型引用的 planning context 模式，复用现有模型加载与参数解析 |
| `scripts/paintmesh/run_remove.sh` | 接入 `collision_aware` 分支和配置；传入 full raw mesh、目标模型和同源 manifest；先规划再走公共渲染 |
| `submodules/Inpaint360GS/utils/virtual_camera_manifest.py` | 新 schema、模式校验、相机身份、公用读取和非球形轨迹诊断；旧 schema 身份保持不变 |
| `submodules/Inpaint360GS/tools/paintmesh_normal.py` | 使用公用相机身份与 schema 校验，移除固定列举旧身份字段的重复计算 |
| `scripts/paintmesh/render_virtual_views.py` | 最终相机与规划证据预检；继续两个渲染后端共用同一组相机 |
| `scripts/paintmesh/render_virtual_worker.py` | 复用相机加载与渲染逻辑，为小批量低分辨率评估提供独立输出模式 |
| `submodules/Inpaint360GS/tools/prepare_inpaint_workspace.py` | 接受和绑定新相机协议，保持有序帧集合及源身份检查 |
| `scripts/paintmesh/run_inpaint.sh` | 如需显式模式校验，则对照上游 manifest；不重新生成相机 |
| `submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py` 等相机消费入口 | 审计所有 circle_radius/schema 假设，通过统一读取器获取精确相机 |
| `README.zh-CN.md`、`PRINCIPLES.zh-CN.md`、`plan.md` | 记录模式、边界、命令和真实验证状态 |

`pose_utils.py` 的旧 circle/hemisphere 数值生成器无需改写成复杂规划器；新模式在 shell/规划 CLI 层同级分派，避免使纯数值函数隐式依赖 Open3D 或 GPU。

进程划分：planning context 在现有 Inpaint360GS 环境导出；几何规划在已有 Open3D 的 EDGS 环境执行；GPU 评估通过现有 worker 独立进程运行。禁止把不同项目的 scene/utils 包同时塞入同一个进程。

设计层的逻辑接口如下，具体实现入口以实际文件为准：

```python
context = load_planning_context(context_path)
roi = resolve_target_roi(context, target_id)
scene = PlanningScene.from_full_scene(context, config)
candidates = generate_candidates(roi, context.training_cameras, config)
valid, rejected = evaluate_candidates(scene, roi, candidates, config)
graph = build_certified_view_graph(scene, roi, valid, config)
plan = select_feasible_sequence(graph, count=N, config=config)
plan = verify_with_gaussian_previews(plan, context, config)
validate_complete_plan(scene, roi, plan, config)
publish_camera_manifest_and_evidence(plan, output)
```

## 6. 新相机协议与缓存

建议新增 schema 3 专用于 collision_aware。不要用假的 `circle_radius` 或旧半球字段描述不等半径、非球面的轨迹。

schema 3 的公共字段为 kind、schema_version、frame_count、iteration、精确 cameras 和 artifact_id。新增 trajectory/planning 描述包含：

- `type=collision_aware`、算法版本、配置哈希和随机种子；
- target ID、目标包络、focus、scene up 和精确坐标适配策略；
- 完整碰撞 mesh、原始 Gaussian、目标 PLY、语义 sidecar、真实相机的输入哈希；
- 自由空间构建策略、分辨率、clearance、保护体定义；
- 选中的候选 ID、连续路线及必要的折线转折、边验证结果；
- 候选数、有效数、拒绝原因统计、覆盖与 tracking 指标。

schema 1/2 的字段和 artifact_id 算法保持原样。增加统一的 `camera_identity_payload()`，让 normal、renderer、workspace、publisher 使用同一函数计算新旧身份，禁止各模块自行拼接字段。

纯结构读取器维持 CPU/NumPy 可用；完整几何证据的验证放在规划/预检层，不能让 normal 数据读取隐式启动 Open3D 或 CUDA。

输入、阈值、算法、mesh 或实际相机变化必须新建 removal run，旧 tracking mask 不复用。下游从 manifest 读取 N 与有序帧名，训练迭代数不随 N 隐式变化。

## 7. 配置示例

以下均为建议起始参数，尚未通过真实场景标定；实现时所有参数进入身份哈希。最终命名以实际接口为准。

```yaml
algorithm: collision_aware_v1
seed: 0
candidate_directions: 256
radius_multipliers: [0.9, 1.2, 1.6]
surface_probes: 2048
max_refinement_rounds: 2
collision:
  unknown_space: reject
  body_radius_over_target_radius: 0.02
  clearance_over_target_radius: 0.03
  include_near_plane: true
free_space:
  voxel_over_target_radius: 0.02
  max_voxels: 2000000
view_quality:
  min_target_visible_ratio: 0.60
  target_area_fraction: [0.03, 0.45]
  image_border_fraction: 0.05
selection:
  beam_width: 16
  max_neighbor_rotation_deg: 15
  max_neighbor_translation_over_target_radius: 0.15
  minimum_common_target_samples: 32
```

网格粗细和内存预算不满足时应报告资源上限，不静默用更粗格子获得安全结论。clearance 可以附加由重建分辨率决定的下限，最终使用的实际值须报告。相邻共视样本数只是初始工程门槛，真实 tracking 稳定性需要另行评估。

预期 shell 使用方式：

```bash
VIRTUAL_CAMERA_PATH=collision_aware \
VIRTUAL_CAMERA_COUNT=60 \
VIRTUAL_VIEW_CONFIG=scripts/paintmesh/configs/virtual_view_planning.yaml \
VIRTUAL_RENDERER=edgs-pgsr \
RUN_NAME=target14_collision_aware_v1 \
bash scripts/paintmesh/run_remove.sh <dataset> <scene> <resolution> 14 none 1
```

该 shell 接口现已接入；配置字段以实际 YAML 为准，上面的原设计参数仅用于说明。显式选择 PGSR 是为了同时获得 RGB、plane-depth 和直接 normal；规划模式与渲染后端、后续 inpaint pipeline 仍相互独立。

## 8. 输出与失败诊断

输出建议放在当前 removal run 的 tracker 下：

```text
tracker/
  virtual_cameras.json
  camera_trajectory.json
  camera_trajectory.svg
  view_planning/
    input_manifest.json
    resolved_config.json
    candidates.npz
    candidate_metrics.json
    path_edges.json
    safety_report.json
    coverage_report.json
    preview_contact_sheet.png
```

`safety_report` 分开给出相机保护体、近裁剪面、路径扫掠体和已观测自由空间的验证结果。报告最小距离及其定义；未解析的边保持 unresolved。`coverage_report` 给出已知目标表面覆盖、不可达方向、相邻共视和上下文有效性，而不是只有“成功/失败”。

不可行原因至少区分：目标几何不足、目标包络歧义、collision mesh 缺失、无安全锚点、自由空间不连通、目标遮挡过大、上下文不足、N 不足、数值退化、资源上限及预渲染不一致。

保证边界：可以验证相机与已知几何在明确误差余量和路径模型下不相交；TSDF 漏掉的真实障碍、错误几何和未观测空间无法获得无条件物理保证。严格 unknown 策略用于降低此风险，不能将它表述为完整真实世界碰撞证明。

## 9. 验收与实施顺序

### 合成几何测试

1. 空场景、靠墙椅子、桌前椅子：中心距离、构图和遮挡筛选正确。
2. 相机中心安全但 near plane 切入墙体：必须拒绝。
3. 两端安全、连线穿薄墙：必须拒绝；墙厚小于离散步长也不能漏检。
4. 非水密墙面、孔洞、柜体内空腔：不以 signed distance/occupancy 假装可靠判断。
5. 弯折通道：路径转折保留，重采样不得直接抄近路穿墙。
6. 未知区、完全遮挡与无法绕行：正确报告不可行，不拼接重复帧。
7. target/surrounding 分离：临时移除遮挡物不改变完整障碍场。
8. raw/post mesh 面索引不同：sidecar 不能附错 mesh；小障碍不因 post 丢弃而消失。
9. PCA 尺度、trans/scale、z-depth 与射线距离：独立构造可验证的相机与几何，检查距离单位和投影一致。

### 数据链与回归

1. 旧 circle/30 输出矩阵和 artifact_id 完全不变；旧 hemisphere 合法产物继续可读。
2. schema 3 经 render、tracker、workspace、normal、fusion、两训练分支和 publisher 读取，全部接受同一个 N、顺序和相机身份。
3. 更换碰撞 mesh、target ID、规划配置或任一相机后，旧 masks/下游缓存被拒绝。
4. full 与 removed 使用逐元素一致的相机；final target-only 与 completion all-selected 的角色不混淆。
5. 后端预检与有限小型 GPU smoke test 通过后，再做真实场景质量对比。

### 真实场景消融

固定帧数、分辨率、渲染后端、LaMa 权重及下游训练预算，对比 circle、hemisphere、仅避障、避障+可见性、完整序列规划。

报告：已知几何碰撞数（目标为 0）、最小相机/路径间隔、目标可见率、构图裁切率、洞外上下文有效率、相邻共视、tracking 完整率、规划时间和峰值内存。再评估补全视图一致性及最终 mesh 质量；没有隐藏背景 GT 时，不将渲染损失下降当作隐藏几何更准确的证明。若某方法不可行，报告失败率，不只比较成功场景。

实施顺序：

1. 几何与坐标核心、完整障碍输入、位置和路径合成测试。
2. 候选/可见性/构图、图上序列选择、独立 planning CLI。
3. 新 manifest 与全链消费者兼容，保持旧模式数值回归。
4. 小批量 Gaussian 预渲染与有限重规划。
5. 真实场景消融，确定参数后再考虑作为默认模式。

## 10. 参考依据

- [Plan3D: Viewpoint and trajectory optimization for aerial multi-view stereo reconstruction](https://doi.org/10.1145/3233794)：视角收益、轨迹长度、遮挡和已知自由空间的联合规划思想。本文的具体候选、质量目标和工程接入属于本项目设计。
- [Open3D Ray Casting](https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html)：批量 first-hit、primitive ID 及射线参数约定。
- [Open3D Distance Queries](https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)：unsigned distance 可用性；signed distance/occupancy 对水密且内外明确的前提。
- 本仓库 `PRINCIPLES.zh-CN.md` 的相机、移除和补全链，`plan.md` 的 A/A0 与后续阶段边界，以及相机 manifest 和虚拟渲染的实际代码。
