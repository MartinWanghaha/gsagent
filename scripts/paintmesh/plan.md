# PaintMesh normal 全流程、局部几何优化与同级 EDGS-PGSR 路径方案

实现状态：阶段 A 的双后端渲染及阶段 C 的自动 `removed normal → LaMa → completed normal` 已落地。渲染入口为 `scripts/paintmesh/render_virtual_views.py`，同级适配器集中在 `render_virtual_worker.py` 的 `native()` / `pgsr()`。normal 输入准备/总校验位于 `prepare_paintmesh_lama_data.py`；独立 dataset、向量处理位于 `tools/paintmesh_normal.py`；推理位于 `LaMa/bin/predict_normal.py`；`run_inpaint.sh` Stage 2/3 按上游 normal 自动执行。

阶段 A0 已实现：在当前圆环生成器旁增加同心同径的半球螺旋生成器，由独立的 `VIRTUAL_CAMERA_PATH` 选择；两种方式均以 `VIRTUAL_CAMERA_COUNT` 设置总帧数 N，默认 30。半球采用等球带面积间隔与随 N 自动变化的螺距，近似均匀覆盖球面，序列从赤道连续绕行上升至接近顶点。姿态逐帧固定锚定原圆环的场景 up，不累计滚转。全部下游共享同一个 N 帧精确相机 manifest；不改变 RGB/depth/normal completion、反投影和训练算法，其固定帧数校验与枚举已改为 manifest 驱动。已验证非 30 帧数据链与小型 CUDA 局部优化；真实长序列 tracking 和完整训练质量仍待评估。

阶段 F 已接入：保持已有 RGB-D 反投影、单 seed Gaussian 初始化和 RGB finetune，在其后运行**独立、默认关闭的 EDGS-PGSR 局部几何优化**，新增 LaMa normal 监督及局部权重渐增调度。阶段 D/E 的 RGB-D-N 融合和 normal-aware 初始化延期，不作为阶段 F 的依赖。局部入口、开关、配置、编辑范围、checkpoint 与发布链已实现；通过小型合成 30 帧 CUDA 链路测试，不代表真实 kitchen 完整训练的质量已验证。

正式实现：**D0 无手调密度阈值的自适应配额采样**。`SUPPORT_DENSITY_MODE=mass_adaptive` 在 Stage 4 后、Stage 5a 初始化前，根据周边表面密度和洞内表面积自动计算点数。密度匹配只有这一种实现；`legacy` 仅表示不启用密度匹配，保持原生流程。默认开关行为不变，不改变 Stage 5b 的固定点数契约和输入/资源安全检查。真实 kitchen 预处理和小型 CUDA 链路已验证，完整训练视觉质量仍待评估。

新增实现：**同级 EDGS-PGSR 初始化与联合训练路径（已接通，完整场景质量待评估）**，详见 [路径 J](#edgs-pgsr-direct-plan)。公共 RGB/depth/normal completion 后，始终执行 EDGS RGB 匹配，可选 depth 辅助与 normal 定向，直接进入 PGSR 外观/几何联合训练。它不是 Stage 5b 的扩展，也不依赖原有单 seed、RGB finetune 或 D0/D1/E 的完成。原路径保持默认；已增加独立入口、配置、发布分流、debug 与恢复测试，完整真实场景质量仍未验收。

根据 [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md) 当前的 RGB/depth 流程，虚拟视角渲染增加两个同级、可配置的后端：`inpaint360gs` 和 `edgs-pgsr`。默认使用 `inpaint360gs`，显式选择 `edgs-pgsr` 时同步输出：

```text
RGB + PGSR plane-depth + PGSR normal + alpha
```

关键原则是：

1. remove 阶段只删除 3DGS Gaussian，不单独“删除法线图”；
2. full 和 removed 3DGS 使用同一组虚拟相机、同一个选定后端；PGSR 分支同步渲染 RGB、depth、normal、alpha；
3. normal completion 固定为 `removed normal → LaMa → completed normal`，与 depth completion 平级；只要上游提供完整的 removed normal 就自动执行，不增加 normal completion 开关或模式选择；
4. 现有 `inpaint360gs` 路径不使用 normal 初始化，仅在显式开启 Stage 5b 后使用 completed normal/depth 监督；新 `edgs-pgsr` 路径可选 RGB-only、RGB+depth、RGB+depth+normal 初始化，并独立选择训练监督；
5. 最终 mesh 仍然由 PGSR depth + RGB 做 TSDF，法线用于监督和质量检查。

局部优化开关独立于自动 normal completion，也独立于 `VIRTUAL_RENDERER`。关闭时完整保留原流程；开启时只影响当前 inpaint run。禁止更改 `run_seg.sh` 的全局训练、`configs/gs/pgsr.yaml` 的默认几何启用时间，或给全局 loss 隐式注入 LaMa 目标。

已新增 `INPAINT_PIPELINE=inpaint360gs|edgs-pgsr`，默认 `inpaint360gs`。它选择 completion 之后的重建路径，与选择虚拟渲染后端的 `VIRTUAL_RENDERER`、选择相机轨迹的 `VIRTUAL_CAMERA_PATH` 相互独立；不能看到 `VIRTUAL_RENDERER=edgs-pgsr` 就自动切换训练路线。

`VIRTUAL_RENDERER` 只选择虚拟视角渲染后端，不改变基础训练、对象删除、3DGS finetune 或最终 TSDF 的后端配置。下面的 normal 全流程图描述 `edgs-pgsr` 分支；`inpaint360gs` 分支保留现有 RGB/depth 补全能力。normal 是否执行由经过验证的上游模态决定，不根据后端名字硬编码，也不由用户额外启用。completed depth 派生的法线只可用于后续质量诊断，不替换 LaMa completed normal。

---

## 一、当前代码中的缺口

当前实现状态及仍然延期的缺口：

- 虚拟视角双后端与 PGSR raw normal 导出已经实现；`virtual_pose.py --poses-only` 负责相机生成，`render_virtual_worker.py::pgsr` 负责四模态同步渲染；
- A0 同级 circle/hemisphere、动态 N 帧、连续姿态与相机 manifest 兼容性改造已接入；默认 circle/30 相机不变；
- [`PGSRRenderer.render()`]( /home/martin/code/gsagent/submodules/EDGS/source/renderers/pgsr.py:146) 已经返回：
  - `render`
  - `plane_depth`
  - `rendered_normal`
  - `rendered_alpha`
  - 可选的 `depth_normal`
- [`EDGS/render.py`]( /home/martin/code/gsagent/submodules/EDGS/render.py:378) 的最终真实视角输出仍需扩展 raw normal；这与已完成的虚拟视角 raw normal 导出是不同入口；
- `prepare_paintmesh_lama_data.py`、`run_inpaint.sh` Stage 3 和 `predict_normal.py` 已实现自动 normal 输入准备、LaMa 推理和完成校验；completed normal 已接入独立 Stage 5b loss，不改变融合/初始化；
- [`edit_object_inpaint.py`]( /home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_inpaint.py:251) 当前 finetune 只有 RGB loss，没有 depth/normal loss；
- [`compose_utils.py`]( /home/martin/code/gsagent/submodules/Inpaint360GS/utils/compose_utils.py:90) 新 Gaussian 默认单位旋转、三轴相同尺度，PLY 中的 `nx,ny,nz` 目前也不会被实际用于初始化。
- 新路径 J 已新增 EDGS 局部初始化器、可选模态数据加载器、PGSR 外观/几何联合训练入口与分支发布契约；其完整场景质量消融仍未完成，不能用旧 Stage 5b 的通过记录代替验收。

---

## 二、现有路径流程与新路径边界

下图专指现有 `inpaint360gs` 路径。以下 30 帧表示默认配置。A0 已落地，各处帧数统一为 camera manifest 声明的 N，包括 N 个支持 PLY；仍只选择一个 seed 初始化，不因增帧而合并所有 PLY。新路径共享图中 completion 之前的步骤，此后按 [路径 J](#edgs-pgsr-direct-plan) 分流，不要求单 seed 支持 PLY。

```text
原始 EDGS-PGSR 3DGS
        │
        ├── PGSR render
        │     ├── full RGB
        │     ├── full depth
        │     └── full normal       （仅用于诊断/参考）
        │
        └── semantic Gaussian removal
                    │
                    ▼
             removed EDGS-PGSR
                    │
       同一组 30 个 virtual cameras
                    │
                    ├── removed RGB
                    ├── removed depth
                    ├── removed normal
                    └── removed alpha
                    │
                    ▼
                 tracker mask
                    │
        ┌───────────┼───────────────────────┐
        │           │                       │
   removed RGB  removed depth         removed normal
        │           │                       │
      LaMa        LaMa                    LaMa
        │           │                       │
   completed RGB completed depth      completed normal
        │           │                       │
        └─────┬─────┘                       │
              ▼                             │
    原有 RGB-D 反投影（30 个 PLY）           │
              │                             │
              ▼                             │
    原有单 seed Gaussian 初始化             │
              │                             │
              ▼                             │
    Stage 5a：原有 RGB finetune              │
              │                             │
              ├── 局部优化关闭 ──────────────┼──> 直接发布 5a 结果
              │                             │
              ▼                             ▼
    Stage 5b：独立 PGSR 局部几何优化（已实现、显式开启）
              ↑                             │
    completed RGB/depth + cameras/masks ─────┘
              │
              ▼
    Stage 6：发布所选结果 -> PGSR render -> RGB-D TSDF mesh
```

completed normal 在 Stage 5b 直接作为相机坐标系二维监督，不需要反投影成点，也不需要先变换到世界坐标。其余阶段不会因为局部优化开关而更换深度定义、seed 或初始化方式。

---

## 三、统一数据格式

不要把 normal PNG 当作数值法线使用。本方案的 normal PNG 只作为预览；LaMa 输入直接从 float32 raw normal 编码，不经过 PNG 量化。

两个后端复用相同目录和帧命名规则，并在 manifest 中声明实际提供的模态。下面是 `edgs-pgsr` 分支每个视角的输出：

```text
virtual/ours_object_removal/iteration_<N>/
├── renders/
│   └── 00000.png                  # RGB
├── depth/
│   └── 00000.npy                  # PGSR plane z-depth, float32, HxW
├── normal/
│   └── 00000.npy                  # raw normal, float32, HxWx3
├── alpha/
│   └── 00000.npy                  # PGSR accumulated alpha, float32, HxW
├── normal_valid/
│   └── 00000.png                  # 独立的法线有效性 mask
├── depth_normal/
│   └── 00000.npy                  # 可选，depth 派生法线
├── normal_vis/
│   └── 00000.png                  # 仅可视化
└── render_manifest.json
```

法线规范：

```text
坐标系：PGSR camera space
轴方向：+x 向右，+y 向下，+z 向前
方向：朝向相机
有效法线：单位向量
无效法线：[0, 0, 0]
```

`plane_depth` 是场景坐标尺度下的 z-depth。只有场景经过尺度标定时，才能称为“米”。

---

## 四、阶段 A：建立同级、可配置的虚拟视角渲染后端

### A0. 同级圆环/半球虚拟相机生成方式（已实现）

#### A0.1 范围与选择接口

增加与原圆环平级的半球生成器，不替换旧算法，也不把轨迹绑定到某个 renderer。完整几何定义见 [PRINCIPLES.zh-CN.md 第 3.3 节](PRINCIPLES.zh-CN.md#33-模块-b生成严格一致的虚拟相机默认-30-个)。以下接口已接入：

```bash
VIRTUAL_CAMERA_PATH=circle                     # 默认；circle | hemisphere
VIRTUAL_CAMERA_COUNT=30                        # 两种轨迹共用；总数 N，不是每圈数量
VIRTUAL_HEMISPHERE_MAX_ELEVATION_DEG=85         # 仅 hemisphere 使用，0 < 值 < 90
```

`VIRTUAL_RENDERER` 继续独立选择 `inpaint360gs|edgs-pgsr`，两种轨迹与两种渲染后端均可组合。N 必须为至少 2 的整数；仰角校验有限性与合法范围；未知轨迹名提前报错。circle 模式不使用半球专属参数，其缓存身份也不应随未使用的参数变化。小 N 仅数值合法，不意味着足够绕行、覆盖和 tracking；后续 seed/debug 索引必须另行校验。

N 是整条序列的精确总数，不额外叠加一整圈 30 帧。circle 模式保持单圈算法、使用 N 个位置；默认 `circle + N=30` 必须复现原相机矩阵。hemisphere 的圈数从 N 和上限仰角自动推导，以免只加密螺旋曲线而不填补圈间空隙；只有首帧在赤道，随后立即上升，不要求与 circle 后续同名帧重合。例如可配置 N=60/90/120，不承诺某一数量适合所有场景。

#### A0.2 共同基准与半球位置

先从同一批训练相机得到 PCA 变换、观察焦点 $F=(f_x,f_y,f_z)$、原圆环几何圆心 $C=(f_x,f_y,0)$ 和实际半径 $R$。$R$ 沿用原 `generate_virtual_radius` 与原圆环半径的完整精度结果；`circle_radius` 仍为比例参数，不能直接拿它当球半径。

必须明确 $C$ 与 $F$ 不一定相等：半球球心为 $C$、直径为 $2R$，注视点继续为 $F$。不在此改成目标物体中心，不改变原赤道平面，也不对半球重新拟合一个球心。半球的上轴为经过训练相机 up 符号校正后的 PCA 正 z 轴，沿原 PCA 变换返回世界坐标；不假定它是世界坐标的 z 轴或真实重力方向。

以仰角 $\phi$、展开方位角 $\theta$、归一化高度 $h=\sin\phi$ 参数化球面。面积元和覆盖区域面积为：

$$
dA=R^2\cos\phi\,d\phi\,d\theta=R^2\,dh\,d\theta,
\qquad A=2\pi R^2h_{max},\quad h_{max}=\sin\phi_{max}.
$$

面积关系可参考 [PBRT 均匀半球采样推导](https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres)。本项目采用确定性连续螺旋，并非随机取点后任意排序。不能沿仰角等间隔、每圈固定点数，也不能把曲线等弧长声称为球面等面积。

先在 h 上均匀布点，再按自动螺距确定方位：

$$
h_i=\frac{i}{N-1}h_{max},\quad \phi_i=\arcsin h_i,
\qquad i=0,\ldots,N-1,
$$

$$
d=\sqrt{\frac{A}{N-1}},\quad \delta=\frac dR,
\qquad \theta_i=\theta_0+\frac{2\pi}{\delta}\phi_i,
\qquad K=\frac{\phi_{max}}{\delta},
$$

$$
P_i=C+R\bigl(\sqrt{1-h_i^2}\cos\theta_i,
             \sqrt{1-h_i^2}\sin\theta_i,h_i\bigr).
$$

全部角度以弧度计算。$\theta_0$ 沿用原圆环首帧方位，$\delta$ 为绕行一圈的仰角增量，不强制 K 为整数。相邻完整球带面积严格相等；远离极点、细采样时，沿圈和跨圈的间距都约为 d。K 随 $\sqrt{N-1}$ 增长，N 增大时同时加密两个方向。在 85° 上限下，N=30/60/90 对应 K≈3.19/4.56/5.59。

这保证确定性与高度面积分配，并使二维间距近似均衡，不声称有限点集严格等面积 Voronoi 或全局最优。端点强制位于赤道和上限仰角，存在边界效应；85° 以上顶帽不属于采样区域。均匀性必须同时检查高度/方位覆盖、最近邻距离和球面空隙，不能仅由公式或高度直方图宣称通过。

这里“一圈一圈向上”采用**连续螺旋**：其连续参数形式为 $\theta=\theta_0+2\pi\phi/\delta$、$0\le\phi\le\phi_{max}$。序列直接按 i 编号，不按方位取模后重排，不在圈边界反向或返回低处。近顶点方位角差可以变大，但判断空间跳变应看球面距离和相机姿态，而不是单独看方位角。禁止固定 xy 半径的圆柱螺旋、重复极点、回连赤道和二次等弧长重采样。

简化伪代码（只有相机中心；随后必须按 A0.3 构建姿态并返回原场景坐标）：

```python
h_max = sin(radians(max_elevation_deg))
h = linspace(0.0, h_max, N)
phi = arcsin(h)
pitch = sqrt(2 * pi * h_max / (N - 1))
theta = theta0 + 2 * pi * phi / pitch
radial = sqrt(clip(1 - h * h, 0, 1))
positions = C + R * stack([radial * cos(theta), radial * sin(theta), h], axis=-1)
```

#### A0.3 姿态、尺度与 tracking 连续性

- 沿用第一台训练相机的 FoV、分辨率、near/far、scene translation/scale。新首帧使用原圆环首帧姿态；旧 circle 分支的相机矩阵必须保持不变。
- 所有新相机继续看向 $F$，姿态策略只有 `scene_up`。参考 $U$ 与原圆环一致，为 PCA 中训练相机平均 up 的最大绝对分量轴及其符号；不把世界 z 轴硬编码成相机 up。每帧独立计算 $b=(P-F)/\|P-F\|$、$r=(U\times b)/\|U\times b\|$、$u=b\times r$，然后执行现有 PCA 逆变换与轴翻转。相机 up 对齐场景 up 的像平面投影，绝对 roll 为零，不引用上一帧基、不累计滚转，不提供另一套姿态开关。
- 顶点附近只走到默认 85°，不生成重复极点。若视线长度退化或 $\|U\times b\|\le10^{-12}$，明确报错，要求检查输入/降低末端仰角，不静默切换构图参考。仍需检测姿态跳变：无累计滚转不代表近顶点的方位运动消失，也不保证 tracking 成功。
- 保持现有 camera-to-world/world-to-camera、轴翻转及 PCA 相似变换约定；归一化姿态基不等于删除原相机矩阵中的统一尺度。对精确相机中心和实际渲染投影作往返验证，防止 depth 尺度悄然变化。
- 记录相邻位置距离、视轴夹角、相对旋转角、每帧仰角、展开方位角及 `scene_up_roll_deg`；汇总 `scene_up_roll.max_abs_deg/undefined_frames`，输出带编号的轨迹 SVG 及球面覆盖指标。展开方位直接使用生成参数，不能对稀疏相机用 unwrap 错认绕行圈数。测试必须断言整段绝对 roll 接近零，不能只测相邻旋转小。GPU smoke test 验证真实相机与 RGB/depth/normal/alpha 输出；目标覆盖、遮挡及 tracker 传播仍需实际序列验收。

默认 30 帧覆盖多圈后，低处方位采样比原 30 帧单圈更稀。若实际 tracking 丢失，应检查覆盖、遮挡和姿态，可由用户显式增大 N 后新建 removal run；不隐式增帧、插入额外过渡帧、改变对象中心、跳过不可见帧或复用旧 masks。相机中心的球面均匀不代表场景表面观测均匀；高处视角可能缺乏真实观测支持，不能通过增加虚拟帧数凭空补充真实几何证据。

#### A0.4 manifest 与缓存兼容

`virtual_cameras.json` 仍是后续消费相机的唯一依据；精确逐帧 `R/T/FoV/尺寸` 为权威数据，任何 worker 或训练入口都不能仅凭轨迹参数重新生成。新增半球记录需要一个明确的新相机 schema，reader 显式同时支持现有格式与新格式；这只是数据契约演进，不是另一套 normal 算法。

新格式在相机身份中增加 `trajectory`：至少包含 `type`、生成算法身份、N、PCA 坐标基/尺度、实际 $C/R/F$、起始方位、上升方向、自动计算的螺距/圈数、末端仰角、`sampling=equal_height_auto_pitch`、`orientation=scene_up` 和 `scene_up_pca`；顶层 `frame_count=N` 必须等于 `len(cameras)`，有序帧名严格为 `[f"{i:05d}" for i in range(N)]`。上游训练相机与半径来源也要可追溯。参数和实际相机共同参与 hash；算法身份不符、未知 schema、轨迹类型、缺字段或篡改均拒绝，不能让当前流程读取不符合当前姿态策略的半球缓存。

兼容策略：

1. 默认 `circle + N=30` 继续原相机计算及原 manifest 写出规则，避免仅因新增模式就使旧圆环产物失效；读取合法旧格式时保留原 artifact ID，不原地补字段或重算身份。对已知 PaintMesh 旧产物按原 30 帧圆环上下文兼容，不能把任意外来旧 schema 自动认作已验证的轨迹。
2. hemisphere 以及 `circle + N!=30` 写新格式；统一相机 schema 校验后，由 render、tracker、workspace、LaMa、fusion、密度匹配、Stage 5a/5b 和发布链共同接受。重点清理 `paintmesh_normal.read_cameras()` 的固定 schema/身份字段列表，以及 `prepare_inpaint_workspace.py` 的 schema 假设，不能只改写入端。
3. 共享的纯 JSON/NumPy 身份校验应可在 LaMa 环境运行，不为读取相机引入 EDGS/CUDA 依赖。仅扩展相机 manifest 的接受范围，不放宽其他 manifest 的类型、hash 和完成状态验证。
4. `run_remove.sh` 在生成或复用 Stage 4 产物前验证请求轨迹、N 与已存身份；断点从 Stage 5 开始也必须检查，不能由于跳过生成就误用另一轨迹或另一数量。
5. 切换轨迹、N、姿态策略或其他生效参数必须使用新的 removal run；重做 full/removed 渲染、archive、tracking、LaMa 与下游优化。只换 `INPAINT_RUN_NAME` 不能改变上游相机，不覆盖已有 run，不修改既有 manifest 冒充新产物。

normal 始终使用各自精确相机坐标系，射线翻转从该帧 FoV 和像素坐标计算。姿态改变后 normal 数值可以变化，但 RGB/depth/normal/alpha 必须同帧同投影；不能把相同帧名下的旧圆环 normal 或 mask 混入半球结果。

#### A0.5 动态帧数必须贯穿全链

N 仅在 removal 相机生成入口配置；其余阶段从验证后的相机 manifest 读取 N 和完整有序帧表，不设置独立的 LaMa/训练帧数开关。`run_inpaint.sh` 不重新生成相机；若显式传入 `VIRTUAL_CAMERA_COUNT`，只能与上游比对，不可覆盖。具体改造包括：

- `render_virtual_views.py`、`run_remove.sh` 的 archive/session/mask 检查和提示，不能再使用 `range(30)`。
- `tracker_web.py` 的 `NAMES`、索引边界、完整性与 API `total`，以及 `tracker_web.html` 的按钮、进度条和提示，都来自会话帧表；模型传播必须输出恰好 N 帧。
- `prepare_inpaint_workspace.py` 的 `expected_frames` 与 tracking 绑定；`run_inpaint.sh` 对 LaMa 准备/验证传 `--frames N`。保持有 normal 就自动完成 N 帧 normal 的规则，不能只取前 30 帧。
- `virtual_camera_manifest.py` reader 不再隐式要求 30，`paintmesh_normal.py` 使用共享 CPU schema/身份校验并接受任意合法 N。
- `edit_object_removal_plyfusion.py`、`edit_object_inpaint.py` 的 `FRAME_COUNT` 及序列校验改为 manifest 驱动。RGB-D 反投影输出 N 个 PLY，但保留单 seed 初始化；非默认轨迹缺 manifest 时直接报错，不得落入旧圆环 fallback。
- `prepare_density_support.py` 的 `seed_frame < 30` 改为 `< N`；`local_geometry_io.py` 的目标帧集合及 `debug.view_index` 上界取 N。涉及相机数量的关联、去重或软证据逻辑遍历全表，不把增帧直接当作点数倍率。
- `publish_inpainted_edgs_model.py` 等发布/验收工具移除固定 `FRAME_COUNT=30`，保持 camera、tracking、LaMa、fusion、RGB/local manifest 对同一有序集合逐项一致，不能退化成仅比较目录文件数。

所有 seed/debug 索引要求 `0 <= index < N`，越界提前报错而不是截断或自动选帧。缺帧、重帧、乱序、额外帧仍失败；沿用既有文件命名格式和来源 hash。训练迭代数、损失权重与图像分辨率不随 N 自动改变：固定 5000 步在 N 增大时每个视角平均被采样次数会下降，应单独评估，不隐式改训练预算。

#### A0.6 代码落点与实施顺序

下表为已接入的代码落点；算法与工程验收边界见 A0.7。

| 位置 | 已实现职责 |
|---|---|
| `utils/pose_utils.py` | 保留 circle 路径；复用原轨道基准，通过 `generate_virtual_path` 同级分派，等球带面积/自动螺距位置采样与逐帧 scene-up 姿态构建 |
| `tools/virtual_pose.py` | 新增 `--camera-path`、`--camera-count`、`--hemisphere-max-elevation-deg`；显式选择同级生成器与总数，仍支持 `--poses-only` |
| `utils/virtual_camera_manifest.py` 与共享 CPU 校验工具 | 新旧格式读取、半球轨迹身份、精确相机写出与校验；不改变旧圆环身份 |
| `scripts/paintmesh/run_remove.sh` / `run_inpaint.sh` | 轨迹与总数解析、预检、Stage 4 参数传递、动态帧数传递、恢复时身份检查；不改 stage 编号 |
| workspace / `paintmesh_normal.py` / 渲染、跟踪、融合、密度匹配、训练及发布消费者 | 按 A0.5 消除固定 30 帧假设，更新相机 schema 和 hash 规则，仍只读 manifest，不生成自己的轨迹 |
| `README.zh-CN.md` 与相机相关 tests | 可运行命令、几何/兼容性测试和新的 removal run 示例 |

- [x] 圆环回归及多 N 的纯 CPU 球面位置、最近邻分布和绝对滚转测试，含倾斜世界坐标和偏离赤道的注视点。
- [x] 同级生成器、manifest 读写兼容及 A0.5 固定数量消费者适配。
- [x] shell 轨迹/总数选择、续跑身份检查及 `camera_trajectory.json/.svg` 逐帧/最近邻诊断。
- [x] 非 30 帧合成 fixture 的 tracker、normal 编解码/LaMa 产物链、workspace/fusion 发布和 CUDA 局部优化/密度 sidecar 测试；不代表执行了真实 LaMa 网络或完整 Stage 5a 训练。
- [x] 临时隔离工作区使用真实 kitchen 输入，验证 90 帧 scene-up 半球的两后端 full/removed 渲染、产物哈希和 PGSR 四模态输出。
- [ ] 真实半球长序列的人工 tracking、完整 LaMa 网络、Stage 5a/5b 长训练与球面覆盖质量对比。

#### A0.7 验收边界

数值测试要求：`circle + N=30` 输出与基线一致；半球与同输入圆环严格同心同径（允许浮点误差）；全体中心在上半球、仰角单调、首帧吻合、最后一帧达到配置仰角；恰好 N 个不重复且有限的相机，顺序为连续上升绕行；视线与 $F$ 对齐、姿态基正交且不改变手性，全程相对于 $U$ 的滚转误差接近零。覆盖 N=30/60/90/120、低帧数边界、$f_z\ne0$、接近顶视、不同 PCA 方向、非法 N/仰角及退化输入。

均匀性与连续性分别验收，不混为一个指标：

1. 用等 h 球带统计点数；联合 h/方位扇区检查覆盖，不能只测高度均匀。分区尺度随 N 调整，小样本允许离散计数误差，不要求每个细网格都非空。
2. 记录球面最近邻角距离的 min/median/max 和变异系数；在覆盖球面带的确定性密集等面积探针上，记录到最近相机的距离分位数及最大空隙，并由最近相机归属估计单元面积离散度。增加探针密度检验统计收敛；不能拿完整球面的 Voronoi 面积直接衡量截断半球带。
3. 检查增大 N 时典型间距和覆盖空隙约按 $N^{-1/2}$ 下降，若覆盖明显偏斜，先修正采样设计，不把统计输出当作已通过。
4. 按帧顺序另记相邻位置/视轴/相对旋转的变化，再用实际 RGB 重叠与 tracker 结果验收；球面均匀不保证可跟踪，也不保证缺失法线几何正确。

已用实际生成器测试 N=2/5/7/30/60/90/120/180 的球面半径、首尾、方向、姿态与默认圆环精确兼容；相机专项 48 项测试通过。新 schema 可在不导入 torch 的 CPU 环境校验，实际 LaMa 环境也通过相机读取与缓存拒绝检查。工具测试集 150 项通过（1 项真实 LaMa 网络测试未启用），PaintMesh 测试集 57 项通过（2 项真实场景渲染测试单独执行）；包含合成 tracker、normal 产物链、workspace/fusion 发布及 CUDA 短局部训练，不代表真实 tracking 或完整训练。

真实 kitchen 的 90 帧 scene-up 半球已完成两后端 full/removed 渲染，并独立通过全部帧的产物哈希、全部 PGSR 有效 normal 单位长度和 `--validate-only` 校验。合并 GPU smoke 进程被 SIGTERM 终止，未计为整组 pytest 通过；已对其完整生成的产物单独重跑上述校验。相对场景 up 的最大绝对滚转约 $6.7\times10^{-14}$ 度，无未定义帧；首帧 RGB/raw RGB/depth/normal/alpha 与原圆环逐文件相同。人工查看第 10、23、89 帧确认构图不再出现累计滚转。测试仅写临时隔离目录，不覆盖已有场景结果。

使用 64×256 等面积探针检查默认 85° 球面带：N=30/60/90/120 的最大覆盖空隙约为 26.63°/19.27°/15.27°/13.60°，估计最近点单元面积变异系数约为 0.196/0.165/0.150/0.143。这些离散探针结果支持自动螺距随 N 加密覆盖，不等于严格等面积证明；真实 tracking/LaMa 网络及长训练质量仍需评估。当前诊断文件直接输出探针最大空隙、95% 分位数与单元面积变异系数，并单独报告相机绝对滚转。

契约测试要求：旧 manifest 原样复用；新格式篡改或未知类型拒绝；两后端对同一相机 manifest 消费一致；变化的轨迹或 N 不能复用旧 tracker/workspace/LaMa/fusion；所有 N 帧产物完整且排序一致，含正常/缺失 normal 分支；seed/debug 越界提前失败；normal 三通道与像素射线约定仍正确；Stage 5b 与全局 `run_seg` 开关行为不变。

真实场景验收分别报告 tracker 完整性、补全区域覆盖、normal 有效率和相邻视角稳定性。`FUSION_SEED_FRAME=4` 保持“第 5 帧”索引语义，但 N 和位姿改变后需人工检查其有效覆盖；本次不修改 seed 选择算法。输出支持 PLY 的文件数量随 N 变化，不等于初始化 Gaussian 数量自动增长。半球不保证多视角 LaMa 一致性，更不能用公式检查或 smoke test 代替完整训练质量对比。

### A1. 统一入口与职责

通过以下配置显式选择后端，不根据文件名或 PLY 属性自动切换：

```bash
VIRTUAL_RENDERER=inpaint360gs  # 默认，现有原生渲染行为
VIRTUAL_RENDERER=edgs-pgsr    # RGB + plane-depth + Gaussian normal + alpha
```

统一调度层位于 PaintMesh 层。两个 worker 是同级实现，在各自环境中独立运行：

```text
run_remove.sh Stage 4
    -> 生成或验证 virtual_cameras.json
    -> render_virtual_views.py --backend <VIRTUAL_RENDERER>
         ├── inpaint360gs -> Inpaint360GS virtual render worker
         └── edgs-pgsr   -> EDGS PGSR virtual render worker
    -> 校验 full/removed 输出与 manifest
    -> 打包所选后端的 removed RGB
    -> 建立 tracking session
```

已实现中立调度入口 `scripts/paintmesh/render_virtual_views.py`，统一接收 backend、camera manifest、removal work_model、迭代号、EDGS 配置目录及各环境 Python 路径。它负责参数检查、子进程调度和产物校验，不加载任一项目的 CUDA renderer。两个后端均从同一 work_model 读取 full semantic PLY 和 all-selected removed PLY，manifest 记录实际文件 hash 和 removal variant。

两个同级适配器在同一 worker 文件中惰性加载所需项目模块，每次只执行选定的后端：

```text
scripts/paintmesh/render_virtual_worker.py::native
scripts/paintmesh/render_virtual_worker.py::pgsr
scripts/paintmesh/virtual_render_io.py           # 共同的原始数据保存和校验
```

`virtual_pose.py` 增加仅生成相机 manifest 的模式，保留现有直接调用行为兼容旧命令。原生 worker 从现有渲染逻辑抽取，两个 worker 都读取同一份精确相机参数，不重新生成轨迹。独立进程避免 `scene/utils/gaussian_renderer` 同名包冲突。

### A2. 统一数据契约，显式表达能力差异

| 模态 | `inpaint360gs` | `edgs-pgsr` |
|---|---|---|
| RGB | `render` | `render` |
| depth | `depth_3dgs`，保留原生数值定义 | `plane_depth`，场景尺度下的平面 z-depth |
| alpha | 已在 Python render 返回值中暴露 `alpha` | `rendered_alpha` |
| 直接 Gaussian normal | 当前不支持 | `rendered_normal`，归一化并保存有效性 mask |
| 语义 feature | `render_object`，保留现有分类输出 | 当前 wrapper 不提供 |

统一的 frame result 包含 `rgb`、`depth`、`alpha`、可选 `normal` / `normal_valid`，以及 `depth_kind`、`normal_source`、`capabilities`。各模态在所选后端的一次渲染中取得。原生 depth 保持原值，禁止为了接口一致直接冒充 `plane_depth`。

原生分支声明 `normal_source=null`，不生成伪造的零 normal 文件，后续自动保持 RGB/depth-only completion。上游声明有 normal 时，其文件和有效性 mask 必须完整，否则报错而不是当作“没有 normal”跳过。不得通过 depth-derived normal 冒充直接渲染法线，也不可静默切换到 PGSR。

PGSR 虚拟渲染不要求自动追加一次原生语义渲染。tracker 使用所选后端的 RGB；如有其他消费者需要虚拟语义 feature，应作为独立、显式任务配置。

### A3. PGSR 分支的原始数据导出

复用现有 `PGSRRenderer`，无需新增 rasterizer。full 和 removed 的 RGB、depth、normal、alpha 必须来自各自模型在同一相机下的单次 PGSR 调用。

核心解码逻辑：

```python
def decode_pgsr_geometry(package, alpha_min=0.01):
    rgb = package["render"].detach().clamp(0.0, 1.0)

    depth = package["plane_depth"].detach().squeeze(0)
    alpha = package["rendered_alpha"].detach().squeeze(0)

    # rendered_normal 是 alpha-weighted normal
    normal_premultiplied = package["rendered_normal"].detach()
    normal = normal_premultiplied / alpha.clamp_min(1e-6)[None]
    normal_length = torch.linalg.vector_norm(normal, dim=0)
    normal = torch.nn.functional.normalize(normal, dim=0, eps=1e-6)

    valid = (
        torch.isfinite(depth)
        & (depth > 0)
        & torch.isfinite(alpha)
        & (alpha >= alpha_min)
        & torch.isfinite(normal_length)
        & (normal_length > 1e-6)
        & torch.isfinite(normal).all(dim=0)
    )

    normal = torch.where(valid[None], normal, torch.zeros_like(normal))

    return {
        "rgb": rgb,
        "depth": depth,
        "normal": normal.permute(1, 2, 0),  # HWC
        "alpha": alpha,
        "valid": valid,
    }
```

渲染调用如下；`return_depth_normal` 只控制可选的深度派生法线诊断，不是直接 normal 输出的前提：

```python
package = renderer.render(
    view,
    gaussians,
    pipeline,
    background,
    return_plane=True,
    return_depth_normal=save_depth_normal,
)
```

使用 `torch.no_grad()`，不要使用 `torch.inference_mode()`，因为 PGSR 会创建需要梯度属性的 screen-space tensor。

### A4. 后端选择与断点续跑

`render_manifest.json` 记录统一 backend 名称（`inpaint360gs` 或 `edgs-pgsr`）、模态能力、depth 定义、normal 来源、相机与模型 hash、removal variant 和全部帧输出 hash。两个 worker 都采用“开始标为未完成、全部验证后提交完成”的规则。

tracker archive 必须由本次选定后端的 removed RGB 生成。workspace、tracking、LaMa 输入和最终产物通过 render artifact ID 追溯后端。`run_inpaint.sh` 从上游 manifest 读取实际后端；若用户指定期望后端，则必须匹配。

切换后端应使用新的 removal run；已有 run 的 backend 或数据契约不匹配时拒绝复用。禁止只覆盖 normal 或 depth 而继续沿用另一后端的 RGB、tracker archive 或旧 mask 会话。

### A5. 第一批实现与验收

先实现统一入口、原生 adapter、PGSR adapter 与单视角输出校验，再扩展到 30 个虚拟视角并接入 Stage 4。验收包括：默认原生行为兼容；PGSR 四模态同源；未知后端和不支持的模态请求能提前报错；两后端复用完全相同的相机；切换后端后旧产物不被误复用。阶段 A 完成后再实现 normal completion。

---

## 五、阶段 B：remove 后重新渲染 normal

不需要再编写一个“normal removal”算法。

正确方式是：

```text
semantic 3DGS
    -> edit_object_removal.py
    -> removed 3DGS
    -> 所选后端重新渲染（PGSR 分支输出 RGB/depth/normal/alpha）
```

不能直接对 full normal 图做 mask，因为删除 Gaussian 后：

- 遮挡关系发生变化；
- alpha 发生变化；
- 可见背景可能暴露出来；
- normal 的可见性也发生变化。

如果当前有：

```text
iteration_N
iteration_N_removal_target
```

这类多个 removed 版本，RGB、depth、normal 必须使用同一个版本，并在 manifest 中记录：

```json
{
  "removal_variant": "target_only",
  "removed_model_artifact_id": "...",
  "camera_manifest_id": "..."
}
```

full normal 只能用于诊断和坐标检查，不能直接作为实际洞区域的 normal ground truth。因为 full normal 描述的是被删除物体，而不是希望补出的背景。

---

## 六、阶段 C：normal completion

此阶段已实现。固定采用与 depth 平级的 LaMa 分支，不提供 depth-derived / learned / hybrid 模式切换。

### C1. 自动启用契约

以 workspace 验证过的 removed render manifest 与文件为准：

| 上游状态 | 行为 |
|---|---|
| 声明并完整提供 30 帧 normal、normal_valid、alpha | Stage 2 自动准备 normal；Stage 3 必须完成 normal LaMa 才能提交完成标记 |
| 明确不提供 normal，且不存在残留 normal 文件 | 保持原有 RGB/depth 路径与 manifest 身份，不生成 normal 产物 |
| 声明有 normal 但缺帧、缺 sidecar、hash 不匹配；或目录与声明矛盾 | 提前报错，不静默跳过，不复用旧 RGB/depth-only completion |

旧 run 没有模态声明却存在 normal 时，要求先通过虚拟渲染入口生成有效 manifest；不能仅凭目录存在就信任来源。不新增 normal 开关，也不要求单独指定 normal checkpoint，当前实现复用现有 `LAMA_MODEL_PATH`。

### C2. 输入与 mask

扩展 `tools/prepare_paintmesh_lama_data.py::prepare_lama_inputs()`：

- 读取 float32、H×W×3 的 removed normal，以及 normal_valid、alpha、相机 manifest；只读 raw `.npy`，不从 `normal_vis` 反解；
- RGB、depth、normal 共用同一张经过连通域处理和膨胀的 hole mask `M`，不重复清洗或独立膨胀；
- 校验 normal 有限、有效向量单位长度、无效向量为零，以及帧名、尺寸、相机、removal variant 与 RGB/depth 一致；
- normal 使用固定范围编码，不读取 full normal，也不需要 `normal_original/` 做逐帧 min/max 归一化。

建议目录（均位于当前 inpaint run）：

```text
lama/input/normal/<frame>.npy             # removed raw normal 原样保存
lama/input/normal/<frame>_mask.png        # 与 color/depth 完全相同的 M
lama/input/normal/valid/<frame>.png       # removed normal_valid
lama/input/normal/inference_mask/<frame>.png  # M ∪ 法线无效区域
lama/output/normal/<frame>.npy            # completed normal，float32 HWC
lama/output/normal/valid/<frame>.png      # completed normal validity
lama/output/normal/vis/<frame>.png        # 仅用于预览
```

推理 mask 固定为 `M_infer = M | ~normal_valid`，避免把 mask 外无效的零向量当作真实表面约束；该 mask 仅供 LaMa 内部使用，最终写回范围仍严格为 `M`。正常像素的 removed alpha 用于上游有效性检查，不能用洞内接近零的 removed alpha 否决新预测。若整帧没有 `normal_valid & ~M` 的上下文则报错。

### C3. 同级 predictor 与向量编解码

新增与 `predict_depth.py` 同级的入口：

```text
submodules/Inpaint360GS/LaMa/bin/predict_normal.py
```

独立 `NormalInpaintingDataset` 实际位于 `tools/paintmesh_normal.py`，由 predictor 直接使用，与 CPU 编解码及校验共享契约，采用相同的 symmetric padding/unpadding。当前 `InpaintingDataset` 将所有 `.npy` 当作 depth，并读取 `depth_original/`，所以不能只把 `img_suffix` 改成 `.npy`。保持现有 RGB/depth loader 行为不变。

固定通道对应 `R=nx, G=ny, B=nz`，数组直接以 float32 送入模型，不经过 uint8 PNG 或 BGR 转换：

```text
encoded = (normal_removed + 1.0) / 2.0   # [-1,1] -> [0,1]
prediction = LaMa(encoded, M_infer)      # 使用全部三个输出通道
vector = 2 * clip(prediction, 0, 1) - 1 # [0,1] -> [-1,1]
normal_pred = vector / ||vector||
```

固定处理规则：

- 先检查 prediction 有限性，再裁剪与解码；非有限预测或解码后范数小于等于 `1e-6` 的向量不归一化，保存为零且 valid=false；
- 用同一相机内参得到像素射线 `r = K^-1 [u,v,1]^T`，对洞内 `dot(normal_pred, r) > 0` 的预测翻转，使其朝相机；不是简单令所有 `nz < 0`；
- 只对洞内预测单位化/翻转，mask 外从原始数组直接复制，禁止整图重新归一化或平滑；
- 同步合成 valid：mask 外沿用 removed validity，mask 内取预测有效性。valid 只表示有限、非退化的单位向量，不代表正确性或模型置信度；
- 记录每帧洞内退化比例；洞内全无有效预测时失败，不发布成功的 completion。不使用 depth-derived normal 隐式填补失败结果。

最终合成定义为：

```python
normal_completed = normal_removed.copy()
normal_completed[M] = normal_pred[M]
valid_completed = normal_valid_removed.copy()
valid_completed[M] = valid_pred[M]
```

当前实现复用 RGB Big-LaMa 权重，是“编码法线作为三通道图像补全”的基线，不把它标成专门训练的法线模型。单位化与朝向校正并不保证 depth-normal 一致性或多视角一致性。completed depth 派生法线只能用于角度误差诊断，不能覆盖此分支输出；后续融合/训练使用的置信度另行评估，不把 valid 或 removed alpha 当作洞内预测置信度。

### C4. 自动调度、缓存与完成校验

`run_inpaint.sh` Stage 3 仍为一个 completion stage，按现有环境依次运行 RGB、depth、normal predictor；“平级”指同级输入/输出和依赖，不要求三个 GPU 推理进程同时启动。normal 不依赖 completed depth，也不受 RGB 的 `RECURSIVE_GUIDE` 控制。

扩展现有 `lama_input_manifest.json`、`lama_completion_manifest.json`，而不是另建不受总完成标记约束的 normal 完成链。记录模态集合、normal 来源与坐标、编码版本、共同 mask 和推理 mask hash、相机/渲染 artifact IDs、checkpoint/config hash、所有 raw normal/valid/vis 输出 hash。

复用前根据上游能力推导 required modalities：有 normal 就要求三路全部通过；任何 normal 帧缺失或来源变化均拒绝旧缓存。明确提示重建 Stage 2/3 或使用新的 inpaint run，不能自动把旧 RGB/depth-only 完成标记升级成成功。只有全部预期模态验证通过，才原子提交 complete=true。

### C5. 已实现代码落点与验收

| 文件 | 实现职责 |
|---|---|
| `tools/prepare_paintmesh_lama_data.py` | 自动识别模态、准备 normal/mask/valid、扩展输出校验与两个既有 manifests |
| `tools/paintmesh_normal.py`（新增） | 独立 normal dataset，固定向量编码、精确帧枚举、padding/unpadding、单位化和相机射线朝向 |
| `LaMa/bin/predict_normal.py`（新增） | 复用 LaMa checkpoint 推理，三通道解码、洞内定向/单位化、raw/valid/vis 原子保存 |
| `scripts/paintmesh/run_inpaint.sh` | Stage 2 准备 normal；Stage 3 从输入 manifest 自动调度 predictor 和完整性校验；不新增 enable/mode 环境变量 |
| `tools/publish_inpainted_edgs_model.py` | 发布时校验 normal 来源、输入、预测记录和逐帧输出 hash，避免绕过 Stage 3 后接受损坏的 normal 产物 |
| `tools/tests/` 与 predictor 测试 | 原生兼容、PGSR 自动启用、数据/缓存失败、法线编解码与真实推理 smoke test |

CPU 测试覆盖：三路 hole mask 相同、mask 外 normal 数值与 valid 完全不变、单位长度/零向量、通道顺序、射线朝向、非整除尺寸恢复、NaN/Inf 与零范数、缺帧/残留模态/hash 不匹配、无有效上下文、旧缓存缺 normal 被拒绝，以及 workspace symlink 和发布链校验。真实 Big-LaMa GPU 单帧/30 帧合成输入 smoke test 检查 predictor 与 completion manifest，并记录洞内有效率；真实场景的边界角度突变和 depth-normal 角度差仍需完成 tracker 后评估，不能因格式验收通过就宣称几何正确。

---

## 七、阶段 D：支持点密度与后续 RGB-D-N 融合

### D0. 无手调密度阈值的自适应配额采样（已实现）

本节描述正式的自适应密度匹配：算法位于 `support_mass.py`，相机/反投影工具位于 `support_geometry.py`，配置为 `configs/support_mass.yaml`。

#### D0.1 问题与“无阈值”的准确边界

原 RGB-D 反投影已经是每像素一个点，但只用 `FUSION_SEED_FRAME` 的单张 seed 初始化，没有把点数与周边 Gaussian 密度关联。Stage 5a 还会 densify/prune 和 gate 提交，Stage 5b 则固定点数；因此初始化密度改善不能保证最终密度不再变化。

这里承诺的是**不依赖用户手调角度、深度残差、覆盖率、密度比或采样倍率阈值来决定哪里补点/补多少点**，不宣称算法没有模型假设、离散选择、数值容差和资源限制：

| 计算环节 | 正式实现 |
|---|---|
| 表面建模 | 独立像素面元，局部与多视角连续模型证据 |
| 周边参考 | 有限多尺度邻域模型平均，连续观测可靠度 |
| 点数计算 | 表面积 × 目标密度 − 已有中心贡献 |
| 点位布局 | 面元内分层采样，整数配额守恒 |
| 质量诊断 | 连续误差、配额残差和不确定性，不以质量门槛筛点 |

保留的硬约束只有输入可解释性和任务边界：有限正 z-depth、相机/坐标一致、mask、原空间 gate、产物身份、可分配内存和数值有效性。原 gate 中的距离界限属于用户已有的编辑许可范围，**不作为新密度控制器，也不在此次扩大或移除**。自动 normal completion、Stage 5b 和全局 run_seg 的既有阈值不属于本次改动。

#### D0.2 管线与数据域

```text
30 帧 completed RGB-D、相机、mask + 实际保留背景 Gaussians
       -> 多尺度表面密度参考 + 观测不确定性
seed 每个有效像素 -> 独立连续表面元 / 面积
       -> 传播目标密度 -> 扣除现存背景贡献 -> 每个表面元的缺点配额
       -> 等质量分层采样 -> init_support.ply
       -> Stage 5a RGB finetune（原 seed gate）
       -> 可选 Stage 5b PGSR depth / LaMa normal loss
```

仍是 Stage 4b，不合并 30 张补全深度成 30 层初始化点云；其他视角提供参考和软证据。LaMa normal 不反投影，不覆盖 completed depth，也不决定点数；法线分支继续 `removed normal → LaMa → completed normal → Stage 5b loss`。

最终采样域 Ω 是 seed 有效像素表面与 hole mask、原 gate 的交集。全 hole、有效 Ω、因非法深度/原 gate 限制不可处理的域分别记账，不能把分母缩成“成功放点的面积”来报告覆盖率。单 seed 不可见区域仍不在此版的可恢复范围。

#### D0.3 自动估计周边密度，不把全场景混在一起

1. 复用 Stage 5a 的实际保留行选择，跨视角按原 Gaussian 行号去重；不能拿 full 模型中删除物体的中心作参考。每个背景中心的观测只融合一次，30 帧不能把密度放大 30 倍。
2. 对所有不同 XYZ 的保留中心建立空间索引，使用可用的 `k=2,4,8,16,32` 邻域；数据极少时使用 k=1 的可辨识内外环。以外环留出计数的 Poisson 预测分数对不同尺度的 log-density 做模型平均，不依赖固定环带宽度或残差停止阈值。这是版本化有限模型族，不是无限范围的自动搜索。
3. 当前强度估计为 `k/(πr_k²)`，以局部欧氏半径近似切面面积。使用完整保留中心而非薄环可减少参考截断，但**尚未严格校正相邻双层、尖锐折叠和体积 Gaussian 分布**；这些情况下会有密度偏差。估计尺度间的 log-density 方差降低该锚点精度，不把全场景密度平均成一个目标。
4. 已知区 alpha、opacity、深度残差的 Cauchy 权重及可见性决定观测可靠度，跨相机取每个中心的最大可靠度；seed 已知区域中的投影作为密度锚点。噪声尺度由 MAD 估计，退化时仅使用数据尺度相关的机器精度正则，不新增米制/角度阈值。当前不读取 normal 参与密度估计；可靠度影响锚点可信程度，不把一个中心数成多个 splat。
5. 多表面存在时保留条件于 seed 中心深度的表面假设，不能把前后两层 XYZ 或深度直接加权平均成中间假面。可采用稳健局部逆深度回归与多假设的预测似然比较；若模型选择不能区分，则记录后验歧义，而不是断言自动恢复了正确表面。
6. 将已知区域的 `log(rho)` 在 seed 有效深度的四邻接图上延拓到 hole；边权为 `1/(1+(Δ逆深度/σ)²)`，使用带锚点精度的稀疏线性系统。没有阈值切边或用户可调传播权重，但这不是严格的跨层隔离，复杂遮挡仍需检查。

少于三个不同背景中心（内外环计数不可辨识）、没有已知区域参考观测、或某个有效深度连通域完全没有观测锚点时，返回 `no_reference`，不静默使用全场景均值。这个算法有效域不是可调的 `min_reference` 门槛；本版未实现任意跨遮挡的参考搜索。

#### D0.4 不依赖三角化覆盖率的表面元

每个有限正 completed depth 像素自带一个中心三维点和一个图像像素单元。以该点为锚，在局部/单侧邻域中拟合逆深度表面 `X_u(s,t)`；邻域、表面假设及拟合尺度按 D0.3 的预测模型自动决定。拟合只用于子像素采样和面积计算，不改写 completed depth 文件，中心深度保持原值。

具体实现比较半径 1/2/4 像素下的全邻域、左/右/上/下单侧模型及前平行模型。局部 PRESS 预测误差和无量纲方差惩罚，再结合多视角软残差选取一个模型；不混合跨层深度。`model_choice` 与候选歧义写入 field。面积使用每像素 4×4 子单元的世界坐标雅可比求积，该有限求积精度属于算法版本，不是质量拒绝阈值。

无法从邻域辨认倾斜方向时，退回该像素中心深度的前平行表面元，并记录 `geometry_uncertainty`，而不是删除像素。这是明确的局部几何先验，不代表新恢复的真实几何。在能选择单侧假设时使用同侧颜色/深度支持；不同像素单元之间不构造跨层连接三角形，也不让新点落在前后表面的均值深度上。离散模型选择仍可能选错，必须保留诊断。

表面元只拥有本像素单元裁剪后的 mask/gate 域，不与邻居重复计面积。面元内用面积雅可比做数值积分：

```text
dA = || ∂X_u/∂s × ∂X_u/∂t || ds dt
A_u = ∫_(pixel cell ∩ mask ∩ gate) dA
```

前平行情况下 `A_u ≈ z_u²/(fx*fy)`（刚性相机）；有倾斜或相机均匀尺度时必须用世界坐标雅可比计算。不要用跨深度跳变三角形的巨大面积制造异常点预算。非有限面积/发散模型属于数值失败，不能把无限面积解释为无限补点。

#### D0.5 用缺点配额自动决定数量

设 `rho*(x)` 为延拓来的目标表面中心密度，不额外乘手调 `density_ratio`；设 `b_u` 为现存保留背景在表面元 u 的有效中心贡献：

```text
t_u = ∫_(cell u) rho*(x) dA        # 此表面元应有的中心数，可为小数
m_u = max(t_u - b_u, 0)           # 还需新增的中心质量
M = Σ_u m_u
N_new = round(M)                  # 自动总点数，记录不超过半个点的取整误差
```

`b_u` 由现存背景向同一表面假设的软归属分配得到；同一 Gaussian 的分配质量总和不超过 1，不能跨相机/像素重复抵扣。已存点过密只记 `surplus`，不删除或移动背景。没有依据的低置信点不能硬算作 1 个完整背景中心抵扣。

**不将 `m_u` 再乘 alpha、normal valid 或几何置信度。** 否则越不确定的位置越少点，会重现 hole 稀疏。置信度决定位置估计的可靠程度和报告，不替代密度目标。新点也不能通过缩小 opacity 来假装具有相同的有效覆盖；密度与渲染质量分别评估。

邻域估计与拟合按 `batch_size` 分块；代码在参考/表面计算前及配额计算后估算工作集，超出 `memory_budget_mb` 时返回 `resource_limited`，后者同时报告所需点数。不因资源预算缩减配额。当前实现尚非全流程磁盘流式算法，预算是估计值，不是 OS 级内存上限。`N_new=0` 时允许无新增点，initializer 跳过追加，不用强制造点通过非空检查。

#### D0.6 等质量采样与连续几何证据

1. 按空间局部顺序遍历像素表面元，将 `m_u` 归一化到 `N_new`，用固定随机种子的系统分层重采样分配整数 `n_u`。总数严格为 `N_new`，每个表面元得到其归一化配额的向下或向上取整；这是整数计数要求，不是“低于阈值不采样”。
2. 在获得 `n_u` 个点的面元内进行分层/低差异面积采样，按雅可比修正倾斜表面的采样分布；多个点占据不同子像素位置。位置由 `X_u(s,t)` 给出，不在三维空间随机抖动，不复制相同 XYZ，也不通过放大 scale 代替补点。
3. 新点与现存点的间距可通过保持每个面元配额的受限位置优化改善，现存点固定；只能在同一表面元/表面假设内移动。默认配额采样不需要 Poisson 最小间距阈值；精确重复或不可表示的重合属于数值问题，在同面元重新布局，不能阈值删点破坏数量。
4. 30 视角的深度重投影残差以数据估计的噪声尺度归一化，联合评估 seed 候选。已知区用 removed depth，hole 内用 completed depth；比观测表面更远的样本通过单侧 Cauchy 权重降低可见性贡献，这是遮挡近似，不是严格的后验边缘化。相机相关性核给近重复视角连续降权，同位置重复相机不提供独立证据，不用 2° 二值计票。
5. 这些证据参与面元的表面模型拟合/位置选择，**不事后按置信阈值删除已分配的点**；若位置模型改变了面积，应重新计算配额，而不是维持旧面积上的点数。低置信但有限的结果可以输出并标注不确定性，不保证“多帧一致即真实”。
6. 自适应采样不执行统计离群点或半径过滤；`legacy` 不变。初始子像素布局若越出原 gate，在同像素内向可行求积节点收缩，不减少配额；有限精度内仍无法布局或产生 float32 重复点时返回 `geometry_unresolved`，不发布完成 manifest。全 hole 与 gate 内面积分别记录；完全没有 gate 内面元也报错，不能以空域宣称完成。

#### D0.7 初始化、阶段审计与无门槛质量报告

- 继续分离 `init_support` 和原始 `gate_support`，重采样不改变原 gate 范围；两者、实际保留行、参考估计和配额均绑定 hash。Stage 5a 最终行顺序确定后重新生成 editable sidecar，Stage 5b 固定点数/行序，禁止用初始化索引猜最终行。
- 暂保留 SH/语义、单位 rotation 与 opacity 初始化；scale 从最终支持点间距计算，沿用 squared distance `1e-7` 的数值 floor。1～3 个新点时 scale 邻域加入保留背景，避免三近邻内核因邻居不足产生无限 scale；零点跳过追加。该数值 floor 仍会限制极密点的最小 scale，不作为数量调节器。normal-aware 各向异性初始化仍延期。
- 记录初始化、5a gate 前后及 5b 后的相同局部密度/间距分布；5a 再次剪枝造成的密度下降属于训练期问题，不以重复增加 seed 掩盖，后续密度保持训练需单独设计。
- 当前报告全 hole/有效像素数、完整/原 gate 内面积、gate 外目标质量、参考密度分布、目标/背景/新增/surplus 质量、配额残差和几何不确定性。`density_ratio` 仅为配额兑现比，**不是独立测量的几何或训练后密度精度**。训练阶段另保存支持邻域占用/最近间距审计；其距离带仅用于诊断，不筛点、不控制配额。RGB/alpha/depth/normal 的图像比较继续使用既有 PGSR/debug 输出，不在 Stage 4b 自动宣称视觉验收通过。
- `complete=true` 只表示契约、配额求解及写出完成，不表示几何质量已被证实。高不确定性以警告/数值记录，不以质量阈值拦截；缺输入、无参考、无可行几何、数值失败、资源不足分别有明确状态，不能静默回退 legacy。

#### D0.8 代码落点与迁移（已接通）

`SUPPORT_DENSITY_MODE=mass_adaptive` 启用正式密度匹配，`legacy` 表示不启用该步骤（默认）。算法标识固定为 `mass_adaptive`，不带展示版本后缀。源码、配置和所有输入/输出仍绑定内容 hash；与当前标识或源码不匹配的缓存不自动迁移，请使用新 `INPAINT_RUN_NAME`，不覆盖已有结果。

| 文件 | 责任 |
|---|---|
| `support_mass.py` / `support_geometry.py` | 表面元、参考场、质量配额、分层采样 / 共用相机几何工具 |
| `prepare_density_support.py` | 参考/相机/数据验证、无三角化前置的 Stage 4b 调度、明确失败状态 |
| `configs/support_mass.yaml` / 配置 loader | 仅允许算法标识、seed、debug、内存预算/批量；不支持的质量阈值参数显式报错 |
| `support_density_io.py` | 容器 schema 保持 1，以 `parameters.mode` 和 `config.algorithm=mass_adaptive` 区分；验证输入 hash、配额守恒、field/样本/PLY 一致性 |
| `export_density_reference.py` / `edit_object_inpaint.py` | 保持相同的保留行选择与原 gate；消费配额输出、不二次过滤；支持零追加点 |
| `run_inpaint.sh` / `local_geometry_io.py` / 发布与最终校验 | 新模式身份、init/gate、5a/5b 恢复、产物版本隔离；全局训练不改 |
| `tests/test_support_mass.py` | 配额守恒、尺度/批量一致性、深度跳变、窄 mask、密度变化、零配额、无参考/资源失败、身份篡改测试 |
| `tools/tests/test_density_initialization.py` / `tests/test_local_geometry.py` | CUDA 初始化、背景保持与 30 帧局部优化/中断恢复/缓存复用集成测试 |

实际新增 `density_field.npz`（pixel_id、uv、中心深度、逆深度 slope、rho、完整/gate 内面积、模型选择/歧义、gate_fraction）、`quota.npz`（target、existing、mass、count、surplus、pixel_id）。`support.npz` 保存 XYZ/RGB、子像素坐标、间距、配额 ID 与多视角残差不确定性。field 的候选歧义在 [0,1]，sample 的软残差不确定性可大于 1，均不是校准的正确率。

manifest 绑定源码/输入/输出与运行配置；在发布完成标记前检查质量配额与 PLY 一致性。debug 保存目标密度、配额、候选不确定性、seed 支持预览与原始数组。相同文件名不能跨算法复用，输入文件修改也会拒绝恢复。

#### D0.9 验证范围与后续质量评估

1. 已通过 CPU 合成平面/斜面、深度跳变、窄 mask、左右不同密度、尺度/批量一致性、配额守恒、原 gate 限制、零配额及失败状态测试。
2. 已通过真实 CUDA 初始化（0、1、2、144 个支持点，背景逐点保留），以及带/不带密度匹配 receipt 的小型 30 帧、3 步局部 PGSR 优化、中断恢复、复用及 stale target 拒绝测试。
3. 已用真实 kitchen seed 4 的现有输入运行完整 Stage 4b 入口和 `--validate-only`，结果见 D0.10；只写临时目录，没有覆盖已有训练。
4. 待评估：密度估计的分辨率收敛、紧邻双层、噪声梯度与大遮挡偏差；当前多视角权重是近似模型，不能把测试通过解释为真实表面恢复保证。
5. 待评估：真实 kitchen 开关 Stage 5b 的完整训练与视觉对照，区分初始化收益和 normal loss 收益；训练后剪枝减少点数应通过阶段审计观察。

程序性验收采用可证明/可测试的约束：整数配额之和等于 `N_new`，背景身份与数据不变，所有点在原 mask/gate 和可行表面元内、有限且无重复写入，固定输入可重现，源数据变更拒绝恢复。整体世界尺度变化 s 后，面积按 s²、密度按 1/s² 变换，推导点数应不变；同一连续表面改变像素分辨率时点数仅有离散积分/取整误差。图像分辨率不应再直接决定 3D 密度。

几何/视觉验收报告连续误差、分布及与基线的变化，不恢复固定密度比/覆盖率门槛。包含正面/斜面、同表面不同密度、相邻前后双层、噪声梯度、窄 mask、背景已足够密、全部缺参考、退化面元和资源不足。数值断言的容差只对应浮点/积分误差；这不等于声称算法“零参数”或实际场景已达到真实几何。

#### D0.10 运行命令与验证记录

2026-09-18 正式入口复测：`algorithm=mass_adaptive`，kitchen 的 13,031 个 seed 像素生成 76,701 个补点，配额取整误差 −0.168；`--validate-only` 缓存复用通过。测试仅写 `/tmp/paintmesh-mass-release-36jhinfe`，未修改已有训练结果。算法标识与源码身份已更新，已有缓存不得手工改 manifest 后续跑，应使用新运行目录。

从仓库根目录启用自适应密度匹配和已有独立局部优化，使用新的 run：

```bash
REMOVAL_ROOT="$PWD/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14_pgsr" \
INPAINT_RUN_NAME=normal_mass_local_geometry \
SUPPORT_DENSITY_MODE=mass_adaptive \
LOCAL_GEOMETRY_REFINE=true \
LOCAL_GEOMETRY_ITERATIONS=5000 \
LOCAL_GEOMETRY_FROM_ITER=100 \
LOCAL_GEOMETRY_RAMP_ITERS=2000 \
END_STAGE=8 \
bash scripts/paintmesh/run_inpaint.sh mip-nerf/360_v2 kitchen 8 14 none 1
```

不必设置角度、覆盖率、密度倍率或最大点数。仅想先检查补点时，把上面 `END_STAGE=8` 改成 `4`；确认 `fused/density/debug/` 与 `diagnostics.json` 后，保持所有环境变量一致，将最后的起始阶段 `1` 改成 `5`、`END_STAGE` 改回 `8`。Stage 5 续跑会先校验密度匹配完成 manifest，不接受身份不匹配或未完成的预处理。

2026-09-17 算法数值验证：上游产物只读，Stage 4b 输出到独立临时目录。922,137 个不同保留中心，13,031 个洞内有效像素；目标质量 77,240.802、现存贡献 539.633、新增质量 76,701.168，写出 **76,701** 个不同 float32 XYZ（约为 seed 点数的 5.89 倍），取整误差 −0.168。所有 seed 像素有 gate 内求积面积，入口及缓存复用通过；这不代表训练后最终点数或法线质量已验证。

可复现回归测试（当前 paintmesh 环境）：

```bash
PAINTMESH_LOCAL_GPU_TEST=1 conda run --no-capture-output -n paintmesh \
  python -m pytest -q scripts/paintmesh/tests

PAINTMESH_LOCAL_GPU_TEST=1 \
PYTHONPATH="$PWD/submodules/Inpaint360GS:$PWD/submodules/Inpaint360GS/seg/detectron2" \
conda run --no-capture-output -n paintmesh \
  python -m pytest -q submodules/Inpaint360GS/tools/tests/test_density_initialization.py
```

两组分别运行以避免两个项目的同名 `render` 模块互相污染。2026-09-18 清理后回归：脚本测试 50 passed、2 skipped，Inpaint360GS 工具测试 90 passed、1 skipped、16 subtests passed；共 140 passed。覆盖几何守恒、CUDA 初始化、局部优化和身份拒绝；跳过项是未启用的重型虚拟渲染/LaMa 测试。完整 kitchen 5000 步训练未运行。

### D1. 现有路径扩展 RGB-D-N 点云融合（仍延期，不是 D0 前置条件）

现有实现保持 `edit_object_removal_plyfusion.py`、`point_utils.py` 的 RGB-D 反投影算法、30 个支持 PLY、`FUSION_SEED_FRAME` 和既有 fusion manifest 不变。下面 D1 仅保留未来 RGB-D-N 联合融合设计，不实施这些参数或本节的 normal support 契约，也不以完成 D1 作为局部优化或 D0 的前置条件。D0 的 `fused/density/support.npz` 是独立的密度采样 sidecar，不等于此处的 RGB-D-N support。

本节是对现有 Inpaint360GS 融合器的候选扩展，不是新路径 J 的实施要求。新路径在自己的 EDGS 初始化器内生成、融合候选，不先改造或调用这里的 all-view fusion。

修改：

```text
submodules/Inpaint360GS/edit_object_removal_plyfusion.py
submodules/Inpaint360GS/utils/point_utils.py
```

新增参数：

```text
--completed-normal-dir
--normal-confidence-dir
--removed-alpha-dir
--camera-manifest
--all-view-fusion
```

每个像素需要保存：

```text
xyz_world
rgb
normal_camera
normal_world
confidence
frame_id
pixel_uv
```

建议输出：

```text
fused/
├── mask/
│   ├── 00000.ply
│   └── ...
├── support.npz
└── fusion_manifest.json
```

未来若需要附着三维法线，刚性 camera-to-world 变换下只使用旋转，不能加平移：

```text
n_world = R_camera_to_world · n_camera
```

一般线性变换 `A` 下应使用 `normalize(A^{-T} · n_camera)`；若相机矩阵带统一尺度，应提取真实旋转或正确归一化，不能把含缩放的矩阵直接当正交旋转。

由于 EDGS 的 `world_view_transform` 使用了转置存储，建议从精确的 camera-to-world 矩阵或逆 `world_view_transform` 得到旋转，并写一个 fronto-parallel plane 单元测试验证方向。

不要只把法线写进 PLY 的 `nx,ny,nz` 就结束。当前 Gaussian loader 不会自动使用这些字段，而且现有 PLY 写出的 normal 字段实际上是零值。推荐：

```text
support.npz 作为训练初始化的权威输入
PLY normal 只作为可视化兼容字段
```

另外，当前流程只使用一个 `FUSION_SEED_FRAME`。为了改善补全区域稀疏问题，建议：

1. 融合全部 30 个虚拟视角；
2. 按 voxel 或局部距离合并重复点；
3. 保留 mask 边缘的点；
4. 依据局部邻域距离设置 Gaussian scale；
5. 使用已知区域 alpha、经评估的 normal confidence 和视角夹角过滤异常点；不能把洞内 removed alpha 当成 completed 几何置信度，也不能把 normal valid 当成预测正确性。

---

## 八、阶段 E：现有路径使用 normal 初始化新 Gaussian（延期）

现有路径保留单位旋转、等向尺度初始化，不修改 `compose_utils.py`。本节是现有初始化器的未来候选优化；Stage 5b 从 RGB finetune 后的已有参数开始，不能为了接入 normal loss 重做 seed 初始化。新路径 J 的可选 normal 定向采用独立初始化器，可借用下面的数学约定，但不以修改这些文件为前置条件。

修改：

```text
submodules/Inpaint360GS/utils/compose_utils.py
submodules/Inpaint360GS/scene/gaussian_model.py
```

当前初始化方式是：

```text
rotation = identity
scale_x = scale_y = scale_z
```

建议改为：

```text
最小尺度轴 ≈ normal_world
scale_normal < scale_tangent
```

例如：

```python
normal_axis = 2
s_normal = 0.5 * s_tangent

scales = [
    s_tangent,
    s_tangent,
    s_normal,
]
rotation = quat_from_two_vectors(
    source_axis=torch.tensor([0.0, 0.0, 1.0]),
    target_axis=normal_world,
)
```

注意 quaternion 顺序要遵循仓库当前的 `[w, x, y, z]` 约定。

无效法线点才回退到：

```text
identity rotation + isotropic scale
```

这样初始化后，PGSR 的 `smallest_axis_normal` 才会得到接近目标法线的 Gaussian normal。

---

## 九、阶段 F：独立 EDGS-PGSR 局部几何优化（已实现）

本节只属于现有 `inpaint360gs` 路径的 Stage 5b。新路径 J 不经过此入口，不继承它对 Stage 5a、固定 SH/opacity 或 seed gate 的依赖。

### F1. 入口与全局训练隔离

Stage 5a 继续执行现有 `edit_object_inpaint.py` 的 RGB finetune；Stage 5b 在单独子进程中加载其输出，运行 EDGS Gaussian 参数接口与 PGSR 可微 renderer。不能调用全局 `train.py` 重新 RoMa 初始化，也不能把两个项目的同名 `scene/utils` 包加载进同一进程。

实现入口：

```text
submodules/EDGS/tools/finetune_pgsr_geometry.py    # 独立局部训练入口
submodules/EDGS/source/paintmesh_local_data.py    # 已有 PLY、相机、LaMa 目标、编辑范围
submodules/EDGS/source/paintmesh_local_losses.py  # 局部损失与调度，独立于全局 composer
scripts/paintmesh/configs/local_geometry.yaml    # 仅局部入口读取的配置
scripts/paintmesh/local_geometry_io.py           # 跨项目 CPU 校验、编辑范围与发布契约
```

不修改 `run_seg.sh`、EDGS `train.py` / `source/trainer.py` / `configs/gs/pgsr.yaml` 的训练行为；`source/pgsr_losses.py` 的全局 `PGSRLossComposer` 及其 `step > 7000` 语义保持不变。局部入口只复用 renderer、相机与几何辅助函数，不实例化全局 loss composer，不把局部参数合并回基础模型 `config.yaml`。

### F2. 唯一启用开关与局部配置

局部 shell 参数：

| 环境变量 | 默认值 | 职责 |
|---|---:|---|
| `LOCAL_GEOMETRY_REFINE` | `false` | 仅控制 Stage 5b；与 normal completion 自动执行无关 |
| `LOCAL_GEOMETRY_CONFIG` | `scripts/paintmesh/configs/local_geometry.yaml` | 独立配置文件，不使用全局 PGSR 配置作为 loss 默认值 |
| `LOCAL_GEOMETRY_ITERATIONS` | `1000` | Stage 5b 总步数 `T`，不改 `FINETUNE_ITERATION` |
| `LOCAL_GEOMETRY_FROM_ITER` | `100` | 局部几何项从哪一步开始增加，记为 `s` |
| `LOCAL_GEOMETRY_RAMP_ITERS` | `400` | 从零升到目标权重的步数，记为 `r` |

上述数值是首轮实验起点，不是已验证的最佳超参数。配置优先级为显式环境变量覆盖局部 YAML；最终解析值全部写入 run-local 配置快照。开关关闭时不要求该配置、normal 目标或局部训练依赖存在，也不创建几何产物。开关开启且执行/复用 Stage 5b 时，缺少合法 LaMa normal、匹配的 plane z-depth 或编辑范围应报错，不静默退回 RGB-only。

局部 YAML 示例：

```yaml
iterations: 1000
geometry_from_iter: 100
geometry_ramp_iters: 400
loss:
  rgb_hole: 1.0
  rgb_known_preserve: 1.0
  alpha_preserve: 0.1
  depth: 0.10
  lama_normal: 0.05
  depth_normal_consistency: 0.015
validity:
  render_alpha_min: 0.01
  coverage_alpha_floor: 0.10
optimizer:
  position_lr: 0.000001
  scaling_lr: 0.0001
  rotation_lr: 0.0001
```

当前实现不启用局部 densify/prune、opacity reset、SH degree 递增、全局 scale loss 或多视角 LNCC/reprojection loss。这些均不是新增 LaMa normal loss 的前置条件；未来若加入，另行定义局部调度和验收。

### F3. 输入与局部编辑范围

#### Stage 5a 独立增删点开关（已实现）

- `RGB_FINETUNE_DENSIFY=true` 为默认值，维持 Stage 5a 原有 clone/split/prune 调度；设为 `false` 时停止三种操作和所有对应统计。
- 只控制 Stage 5a 训练循环，不改变 Stage 4b 的自适应密度初始化、RGB loss/optimizer、最终 gate、Stage 5b 固定点数优化或全局 `run_seg`。
- Python 入口关闭参数为 `--disable_rgb_densify`。不通过抬高梯度阈值模拟关闭，也不使用全局 `NO_DENSIFY`。
- 每个 run 的 Stage 5a PLY 同目录保存 `point_cloud.density_policy.json`；不依赖 normal 模态。RGB context/receipt 同时绑定 `parameters.rgb_densify`，局部优化沿既有身份链继承此设置。切换开关拒绝原目录续跑；从 Stage 6 等后续阶段恢复也必须通过 policy 校验。已有无 policy 的产物需使用新运行目录重建，不猜测其训练设置。
- 关闭不等于最终点数完全固定：初始化筛选、gate 和对象恢复仍生效；点位、scale、opacity 也仍会更新，需分别检查 gate 前后密度。

例如，在现有完整运行命令前增加 `RGB_FINETUNE_DENSIFY=false`，并更换 `INPAINT_RUN_NAME`。其余 depth/normal completion 与局部优化参数保持不变。

回归验证：关闭时不访问任何增密统计或增删点操作；开启时保留原 500/100/5000 调度边界；policy/context 拒绝设置切换和无身份产物复用。2026-09-18 两组测试分别 54 passed、2 skipped 和 101 passed、1 skipped（另 16 subtests passed），包含局部 CUDA 恢复测试；未执行完整 Stage 5a 5000 步视觉对照训练。

输入严格绑定当前 inpaint run：

- Stage 5a 最终提交的含语义 PLY、其 hash、行数与 SH degree；不是最初的 full/removed PLY；
- 30 个精确虚拟相机，以及 RGB/depth/normal 共用的 hole masks；不能重新生成轨迹或以原始真实 RGB 监督洞区域；
- 已验证的 completed RGB、completed plane z-depth、float32 HWC completed normal、normal validity 和 LaMa completion manifest；depth 必须与当前模型同场景尺度，normal 为相机坐标、朝向相机；
- 与 Stage 5a 最终 PLY 行顺序绑定的 `editable_mask.npy` 和 `rgb_finetune_manifest.json`。

Stage 5a 的训练、反投影和 gate 算法不变，仅在输出处记录既有空间 gate 确认的可编辑行，包含 gate 内新增点与允许局部提交的原有点。sidecar 必须在最终行筛选/顺序确定后生成，并绑定 seed、mask、相机和 PLY hash；不能只保存训练前的索引或靠 `obj_dc` 猜测新增点。`surrounding_ids` 等最终处理必须同步映射 sidecar。

局部 optimizer 只持有可编辑行的 XYZ、rotation、scale。SH、opacity、语义 embedding、classifier 及其他所有行均冻结。将可训练局部张量与常量背景按原行顺序组合后做**完整场景渲染**，保留正确遮挡；不能只渲染局部子模型。使用新的 optimizer，不继承全局 Adam 动量；仅置零背景梯度不作为冻结保证。

当前实现保持点数、行序和语义字段不变；写出时在输入 PLY 的副本上只更新许可字段，避免 EDGS 普通 PLY writer 丢掉 `obj_dc_*` 或额外字段。最终检查编辑行仍处于已记录的 seed 空间 gate；越界行回退到 Stage 5a 参数并重新计算验收指标。参数级冻结不能阻止局部 splat 对 mask 外像素产生影响，因此仍需已知区域外观约束与渲染检查。

### F4. 四类监督与新增 LaMa normal loss

用同一相机一次可微 PGSR render 得到 `render`、`plane_depth`、`rendered_normal`、`rendered_alpha`，需要一致性项时请求 `depth_normal`。训练不能复用阶段 A 的 `.detach()`/NumPy 导出路径。

```python
N_pred = F.normalize(
    pkg["rendered_normal"] / pkg["rendered_alpha"].clamp_min(1e-6),
    dim=0, eps=1e-6,
)
N_target = completed_normal.permute(2, 0, 1)  # HWC -> CHW
```

两者都是当前相机坐标系的单位向量，直接逐像素比较；不反投影 normal，不旋转到世界坐标，也不读取可视化 PNG。

```text
L_lama_normal = weighted_mean(1 - clamp(dot(N_pred, N_target), -1, 1))
L_depth = masked_mean(SmoothL1(log(D_pred), log(D_completed)))
L_consistency = masked_mean(1 - clamp(dot(N_pred, N_depth(D_pred)), -1, 1))
```

`L_lama_normal` 是外部伪目标监督；`L_consistency` 是模型内部 depth-normal 一致性，二者分别记录，不能用现有 `pgsr_normal` 的名称将其混为一项。统一朝相机后使用有符号点积，不用绝对值掩盖坐标错误。`N_depth(D_pred)` 保留对预测深度的梯度；目标和权重停止梯度。

固定目标域为 hole mask 内 finite、positive 的 completed depth；法线项再要求 completed normal valid、有限且非退化。各项分别判定预测深度、法线与 alpha 的有效性，一致性还须排除无效差分邻域、洞边缘及深度跳变。先选取有效元素再取 log/归一化，不能用 `0 * NaN` 处理非法值。

当前实现在有效目标内使用统一权重，并以较小的 `lama_normal` 权重表达伪监督的不确定性；`normal_valid` 不是置信度，不要求不存在的 confidence 文件。记录 completed depth 派生法线与 LaMa normal 的角度差供诊断，不能覆盖 LaMa 输出；后续引入置信度时再显式版本化其计算方法。

**覆盖率防退化**：不得用洞内接近零的 removed alpha 否决 normal 目标。预测低 alpha 像素暂不计角度/深度损失，但必须记录有效覆盖率，不能靠失去覆盖降低 loss。局部优化开始时缓存同一 PGSR renderer 的 RGB/alpha 基线 `I0/A0`，定义：

```text
L_rgb = λ_rgb_hole * mean_M |I_pred - I_completed|
      + λ_rgb_known_preserve * mean_outside_M |I_pred - I0|
L_alpha = mean_valid_target relu(max(A0, coverage_alpha_floor) - A_pred)
L_total(t) = L_rgb + λ_alpha L_alpha
           + a(t) * (λ_depth L_depth + λ_lama_normal L_lama_normal
                     + λ_consistency L_consistency)
```

`I0/A0` 固定且停止梯度；权重按局部配置取值。没有有效像素的单项返回可反传的零并计数，持续无有效几何监督或覆盖率崩溃时中止且不提交完成。法线目标角度下降不代表重建真值误差下降，应结合 mask 外变化、深度/法线一致性、覆盖率及新视角检查。

### F5. 启用时间与渐增语义

局部 `t=0..T-1` 独立计数，不使用基础训练的 30000 或 RGB finetune 的 5000；恢复 checkpoint 时恢复同一个局部计数、optimizer 和随机状态，不重新 warmup。

```text
a(t) = clip((t - s) / r, 0, 1)    # r > 0
```

当 `t <= s` 时几何项权重为零；在 `s < t < s+r` 线性增加；`t >= s+r` 为完整目标权重。示例 `s=100,r=400`：`t=100/300/500` 对应 `0/0.5/1`。RGB 与 alpha 保持项从 `t=0` 生效。当前实现要求 `T>0, 0<=s<T-1, r>0, s+r<=T-1`，各权重/学习率有限非负；错误配置在 GPU 优化前拒绝。希望 5000 步局部优化可独立设置 `T=5000`，不会改变原 RGB 的 5000 步或全局调度。

局部 renderer 的 plane 输出请求由当前实际损失需要决定，不能复用全局 `required_outputs(step)` 的 7000 步门槛。仅在一致性生效或诊断时计算 `depth_normal`；不得用设置全局 `virtual_camera=true` 来代替读取虚拟相机 manifest。

### F6. 独立输出、发布与断点复用

独立目录，不覆盖 Stage 5a PLY，也不写入最初的 EDGS/semantic/removal 模型：

```text
<INPAINT_RUN_ROOT>/
├── manifests/rgb_finetune_manifest.json
├── local_geometry/
│   ├── config.resolved.yaml
│   ├── rgb_context.json             # Stage 5a 参数与来源绑定
│   ├── request.json                 # Stage 5b 配置与实现身份
│   ├── editable_mask.npy
│   ├── gate.npz                     # seed 投影、膨胀 mask、距离阈值
│   ├── checkpoints/                 # 模型、optimizer、local_step、RNG
│   ├── point_cloud/iteration_1000/point_cloud.ply
│   └── diagnostics/                 # RGB/depth/normal/alpha、覆盖率、权重曲线
└── manifests/local_geometry_manifest.json
```

目录中的 `iteration_1000` 表示局部 `T`；发布目录继续用原 `FINETUNE_ITERATION`，例如 `inpainted_3dgs/.../iteration_5000`，仅作为兼容的输出标签。发布 manifest 必须分别写明 `rgb_iterations=5000`、`local_geometry_iterations=1000`、所选源 PLY 及局部 artifact ID，不能把目录标签解释成总优化步数。

关闭时 Stage 6 沿用旧 PLY 和旧 manifest 身份，不要求局部 sidecar。开启时 Stage 6 只能发布通过验证的 Stage 5b 输出，几何失败不得悄悄发布 Stage 5a。Stage 7/8 的 render、mesh、semantic 和 final manifests 全部依赖新的 model artifact，不能复用旧模型的 mesh。

启用开关、局部配置、训练实现版本、随机种子、5a PLY/编辑范围、相机、LaMa/mask 及输出 hash 都参与几何身份。输出完整验证后原子提交 complete；改开关/权重/调度或重做 tracking/LaMa 时拒绝旧结果，推荐新 `INPAINT_RUN_NAME`。检查点恢复必须同时验证输入与配置身份。

保留整数 Stage 1..8，5a/5b 属于 Stage 5。`END_STAGE<=4` 不运行几何；`END_STAGE=5` 包含启用后的 5b；从 Stage 6 续跑时仅验证所选 5a/5b 已完成，缺少 5b 则提示从 Stage 5 运行。Stage 5 重入时先验证/复用 5a，再运行或恢复 5b，不能为了重跑几何再次覆盖 5a。旧 5a 没有可信编辑 sidecar 时要求重建 Stage 5a，不猜测行对应关系。

### F7. 代码落点与验收

局部过程可视化已接入 `source/paintmesh_local_debug.py`，复用全局 debug 的无状态 RGB/OpenCV 工具，但不修改全局 visualizer 或配置。独立 `debug` 配置默认 `enabled=true, interval=100, from_step=0, view_index=4, jpeg_quality=95`。固定视角额外渲染在 `no_grad` 下执行，不改变随机采样顺序；保存初始/周期更新后/最终 gate 后的 RGB、depth、normal、角度误差、alpha 2×4 拼图及 loss/权重 JSON。深度色阶固定为该目标视角的有效深度范围；输出位于 `local_geometry/debug/`，默认开启只在局部优化运行时生效。配置变更参与局部缓存身份，旧 run 不静默复用。`debug.from_step` 按已完成更新数计数，与零起始 loss 调度独立。

| 文件/环节 | 已实现职责 |
|---|---|
| `run_inpaint.sh` | 解析局部开关/配置；5a/5b 调度与复用；Stage 6 选取来源 |
| `edit_object_inpaint.py` | 仅按需导出最终编辑 mask 与 5a manifest；不改 RGB loss、反投影、初始化或 gate |
| 新增 `finetune_pgsr_geometry.py` / `paintmesh_local_data.py` | 直接加载已有模型、局部参数 optimizer、相机/监督契约、独立保存与恢复 |
| 新增 `paintmesh_local_losses.py` / `local_geometry.yaml` | LaMa normal、depth、内部一致性、外观/覆盖保护及局部渐增调度 |
| `publish_inpainted_edgs_model.py` / `finalize_inpaint_result.py` | 可选局部 manifest 校验、所选 PLY 绑定及完整依赖链验证 |
| 全局 `pgsr_losses.py` / `pgsr.yaml` / `run_seg.sh` | 不改默认配置、公式、调度或训练入口 |

最低验收：

1. 关闭开关，旧 RGB/depth-only 和 normal completion run 均保持原 Stage 4/5a/6 行为；无局部训练调用、依赖和新增完成标记。
2. 固定输入下 Stage 4 支持 PLY、seed、5a 初始化/训练参数不受开关影响；sidecar 对应最终点数、顺序和 PLY hash。
3. 调度单测覆盖 `s-1/s/s+1/s+r/T-1`、非法值和 checkpoint 恢复；全局 7000/7001 步边界行为不变。
4. 平面/倾斜平面测试检验相机轴、朝向、HWC/CHW；零向量、NaN、低 alpha、空 mask 不产生非法 loss；normal、depth、一致性均有预期梯度。
5. 多步优化后非编辑行、SH/opacity、语义与 classifier 数值不变；局部点数和行序不变；落在 gate 外的更新被回退。
6. 缺 normal、错误 depth 定义、相机/PLY/sidecar/hash 不匹配、未完成几何与过期 mesh 均阻止发布；失败保留 5a 和原模型。
7. 小规模 GPU smoke test 后再运行 30 视角；对比“5a 基线 / 局部 depth+一致性 / 再加 LaMa normal”，报告角度、depth-normal 一致性、覆盖率和 mask 外 RGB 变化，不以训练 loss 单独证明几何改善。

阶段 F 本身不承诺修复点云稀疏：normal loss 不会凭空增加点。已实现的 D0 可独立改进初始化支持点密度，不自动启用延期的 D1/E，也不改变 F 的固定点数契约。

已执行局部 loss/config/gate/冻结字段测试，并用真实 PGSR CUDA 执行小型合成 30 帧的基线和最终渲染、3 步优化、中途 checkpoint 中断恢复、完成后复用与最后 checkpoint 恢复；验证输入变化会拒绝复用。PaintMesh/Inpaint360GS 回归 109 项通过、3 项跳过，另有全局 PGSR 相关 28 项通过。真实场景完整训练、质量消融与恢复的长期数值表现仍需进一步评估，不能把上述链路测试当作几何质量验收。

---

<a id="edgs-pgsr-direct-plan"></a>

## 九补充、路径 J：同级 EDGS-PGSR 初始化与联合训练（已接通）

### J1. 路径选择与非目标

算法说明见 [PRINCIPLES 第 3.9b 节](PRINCIPLES.zh-CN.md#edgs-pgsr-direct)。以下契约已接入代码；具体测试与仍未验证的范围见 J7/J8。

```text
公共 Stage 1..3：removal/tracker → RGB/depth/normal LaMa completion
    ├── INPAINT_PIPELINE=inpaint360gs（默认，现有实现）
    │     Stage 4 RGB-D support / 可选 4b 密度匹配
    │       → Stage 5a RGB finetune → 可选 Stage 5b 局部几何优化
    └── INPAINT_PIPELINE=edgs-pgsr（独立实现）
          Stage 4 EDGS RGB 匹配 + 可选 depth/normal 初始化
            → Stage 5 PGSR 外观/几何联合训练
                         ↓
        Stage 6 发布所选路径 → Stage 7 PGSR/TSDF → Stage 8 语义/完成提交
```

新路径不运行 `edit_object_inpaint.py`，不要求 `fusion_manifest.json`、`FUSION_SEED_FRAME`、D0 support、5a editable sidecar 或 5b receipt。它不修改全局 `run_seg.sh`、全局 RoMa 初始化行为、全局 PGSR 权重/启用时间，也不重新估计相机。已有独立局部优化仍服务原路径，不能为了复用它而放宽其输入校验。

共用 completion 按原契约完整执行：上游有 removed normal 时必须补全并验证 normal，与下游是否使用它无关。这里的 RGB-only 指初始化/外部监督不消费 depth、normal，不代表新增一个跳过公共 completion 的开关。

### J2. 可选模态与独立配置

已新增以下接口，默认值及依赖写入 resolved config：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `INPAINT_PIPELINE` | `inpaint360gs` | completion 后的同级路径选择 |
| `EDGS_INPAINT_CONFIG` | `scripts/paintmesh/configs/edgs_inpaint.yaml` | 新路径独立配置，不复用 `local_geometry.yaml` |
| `EDGS_INIT_USE_DEPTH` | `false` | 初始化是否消费 completed depth |
| `EDGS_INIT_USE_NORMAL` | `false` | 初始化是否消费 completed normal；本批要求 depth 同时开启 |
| `EDGS_MATCH_CONFIDENCE_MIN` | `0.5` | `init.confidence_min`：正反向原始 RoMa 置信度都必须达到此门槛，越大越严格 |
| `EDGS_MATCH_CYCLE_PIXELS` | `3.0` | `init.cycle_pixels`：原图像素单位的双向回环误差上限，越小越严格 |
| `EDGS_MATCH_REPROJECTION_PIXELS` | `2.0` | `init.reprojection_pixels`：原图像素单位的三角化重投影误差上限，越小越严格 |
| `EDGS_TRAIN_USE_DEPTH` | 未指定时继承解析后的 `EDGS_INIT_USE_DEPTH` | 是否启用外部 completed depth loss |
| `EDGS_TRAIN_USE_NORMAL` | 未指定时继承解析后的 `EDGS_INIT_USE_NORMAL` | 是否启用外部 LaMa normal loss |
| `EDGS_INPAINT_ITERATIONS` | `5000` | 独立联合训练步数，不代表已验证的最优预算 |
| `EDGS_GEOMETRY_FROM_ITER` | `100` | 新路径几何项开始渐增的零起始局部步数 |
| `EDGS_GEOMETRY_RAMP_ITERS` | `2000` | 新路径几何项渐增区间 |

配置优先级为显式环境变量 > 独立 YAML > 默认值；先解析初始化开关，再解析训练开关的继承。YAML 中训练模态可以用 `null` 表示继承，最终 `config.resolved.json` 与 request 中保存明确布尔值。`matcher.weights` / `matcher.dinov2_weights` 可指定本地权重（相对 YAML 所在目录）；默认使用 PyTorch 缓存，首次缺失时下载官方权重，文件 hash 纳入初始化身份。

初始化必须覆盖以下三种模式：

| depth / normal 开关 | RGB 匹配 | depth 作用 | normal 作用 |
|---|---|---|---|
| `false / false` | 始终执行 | 不读取、不门控、不回退 | 不读取、不定向、不提供置信度 |
| `true / false` | 始终执行 | 辅助点位精修、补充候选 | 不读取；也不使用 depth-derived normal 代替定向 |
| `true / true` | 始终执行 | 同上 | 三维表面定向与各向异性初始化 |

初始化 `false / true` 在本批不支持，预检明确报错，不能自动打开 depth。训练开关则相互独立：例如 RGB-only 初始化之后可显式启用 depth+normal loss，也可仅开启 normal 外部监督；后者不能强制要求 completed depth。PGSR 内部预测的 depth/normal 与自一致性项不等于 LaMa 外部监督。

局部联合训练另有自己的 loss 权重、学习率、随机种子、调度及默认开启的 debug 配置；不继承 `LOCAL_GEOMETRY_*` 或 Stage 5a 的增删点调度。选择新路径同时设置 `LOCAL_GEOMETRY_REFINE=true` 时应报配置冲突，不能在联合训练后再偷偷运行 5b。原有 `SUPPORT_DENSITY_MODE=mass_adaptive` 属于旧 Stage 4b，不能用于新路径，非默认值应提示改用新初始化器的密度预算；`RGB_FINETUNE_DENSIFY` 在新路径不生效，并在启动摘要中注明。

### J3. EDGS RGB 匹配与混合初始化

1. **精确相机与配对。** N、顺序、K、camera-to-world、图像尺寸及 PCA/场景尺度都来自已验证 manifest，不重新 COLMAP，也不按帧名推导姿态。基于相机重叠与基线选对，半球支持跨圈邻接；保留洞外 RGB 上下文，逐批 RoMa 匹配，避免 N² 个全分辨率图对同时驻留 GPU。RoMa 权重冻结，仅用于初始化；PGSR 联合训练不训练匹配网络。
2. **RGB 主干候选。** 复用 EDGS 对应与三角化函数，通过正深度、匹配可靠性、条件数和重投影检查生成局部候选。新增点应投影到支持视图的待补全区域，保留多视图观测轨迹；不把洞外匹配点重复加入已存在的背景。RGB-only 只依据 RGB、相机、mask 与保留场景作此判断，不读取 completed depth/normal 数值。全局 `Trainer.init_with_corr` 会删初始点及调整尺度，禁止直接对整个 removed 模型调用。
3. **可选 depth 点位辅助。** 以鲁棒重投影残差为基础，加入同尺度 completed z-depth 软约束，精修匹配点；匹配不足的区域由有效 depth 反投影产生补充候选。不要求三角化点与 LaMa 深度硬一致，也不把全部点强行吸附到一张补全深度图。跨视角相互矛盾的层保留高可信假设或放弃低可信候选，不能简单平均成不存在的中间表面。
4. **融合与密度预算。** 对重复轨迹/表面候选去重，避免 N 张深度图重复堆叠。RGB-only 从匹配点与可见边界邻域估计间距，不承诺低纹理区必然补齐；启用 depth 后，可从洞内表面积和周边同一表面 Gaussian 间距计算局部预算，再进行配额采样。可复用 D0 的纯几何/配额函数，但不依赖 seed gate、旧支持 PLY 或旧 receipt。密度预算不以手调密度比例为质量门槛，仍需数值有效性、几何可靠性检查和资源上限；不能混入远处背景密度。
5. **可选 normal 定向。** 仅使用 raw completed normal 与 validity，不读取可视化 PNG。按 `normalize(A_camera_to_world^{-T} n_camera)` 转到世界坐标，统一符号、鲁棒聚合；最短尺度轴沿该方向，切向尺度由局部间距决定，法向厚度为严格正值。关闭 normal 时使用单位 quaternion、等向初始尺度。法线不产生三维点，也不是给 PLY 写 `nx,ny,nz` 就完成初始化。
6. **追加并记录来源。** 原有背景参数逐行保留，仅追加新 Gaussians；RGB 初始化 SH，opacity 使用明确记录的初始化策略，语义 embedding 由保留场景的合法邻域继承并冻结。记录 `source_kind=rgb_match|depth_fill`、观测帧/pixel_uv、匹配轨迹、权重、可选 normal、逐点编辑归属与合并映射。removed variant、target/surrounding ID 及临时 surrounding 恢复规则必须显式绑定；不得复活 target 或重复恢复 surrounding。

完成后至少输出初始模型、候选 sidecar、与最终行序对应的 editable mask、配对/候选统计及初始化 manifest。RGB-only 无有效三角化候选时清楚失败，不偷偷走 depth 或旧 seed 初始化。启用 depth 时允许匹配覆盖差而主要由 depth 补点，但必须仍执行匹配并报告实际两类贡献，不能冒充匹配成功。

当前实现边界：先按空间距离/朝向选近邻图对，再做 RoMa 双向 cycle 检查和 EDGS 两视图 weighted DLT；没有新增多视图轨迹 bundle adjustment。跨图对候选在世界网格保留高可信观测，不平均点位；normal 聚合来自该匹配的两次观测。depth 反投影候选带近邻视图的软一致性分数，密度间距来自候选附近的保留背景；复杂遮挡、不同深度层以及边界参考被远处背景污染的情况仍需真实评估。`init.max_points` 是资源保护，不是指定密度；超限清楚报错，不偷偷降低密度。

匹配精度控制已接入 `init.confidence_min=0.5`，原先只要求正向分数大于零的行为不再作为默认。先在正向命中的目标坐标插值反向 RoMa 分数，取两者最小值，再执行硬门槛、cycle 与三角化检查。门槛不能被抽样预算稀释；少于 `samples_per_pair` 就使用实际匹配数。单向高分/反向低分仍拒绝，非法分数、越界和非有限坐标不能通过边界插值被救回。RoMa 上游 `sample_thresh` 不等于本地硬门槛，本入口不调用其 `sample()`。

诊断逐图对记录 `matching.hole_candidates`、`bidirectional_confidence_quantiles`、`confidence_passed`、`cycle_passed`、`sampled`，再记录 `triangulated` 和 `accepted`；报告顶层保存实际 `matching_thresholds` 和 `confidence_kind=min_forward_reverse`。`support.npz` 中 RGB 匹配来源（`source_kind=0`）保存该双向最小分数；depth 补点（`source_kind=1`）的分数是另一类证据，不由 RoMa 门槛控制。阈值进入初始化和联合训练身份，修改后不能复用旧初始化/checkpoint。只需更严格匹配时，先保持 cycle=3 / reprojection=2，从置信度 0.5 开始，再单独提高到 0.7 或 0.8；仍有杂散点可另试 cycle=1 / reprojection=1，区分置信度与几何约束的作用，不能把减少点数等同于几何正确。

### J4. PGSR 外观与几何联合训练

不能直接使用现有 `paintmesh_local_data.LocalGaussians`：其 SH/opacity 是冻结 buffer，只更新 XYZ/rotation/scale。新模型适配器需将**新增点**的 XYZ、rotation、scale、SH、opacity 都纳入 optimizer；背景、全部语义字段和 classifier 冻结，整个场景一起 PGSR 渲染，维持正确遮挡。首批固定点数，关闭 clone/split/prune、全局剪枝和 opacity reset；若初始化不足，不以未声明的增密补救。

损失按 [PRINCIPLES 第 3.9b 节](PRINCIPLES.zh-CN.md#edgs-pgsr-direct) 组合，各自使用独立有效域，不能用 depth/normal valid mask 同时屏蔽 RGB 学习：

- RGB：hole 内 completed RGB 的鲁棒光度损失；若使用 SSIM，窗口边缘必须正确处理 hole/已知区交界。
- 已知区保持：hole 外与相同 PGSR 后端渲染的冻结 removed 基线比较，防止新增 splats 漫出编辑区域。梯度只回传新增参数。
- 覆盖保护：可见待补全表面的 alpha/空洞惩罚及覆盖审计，不能仅靠降低 opacity 来减小几何 loss；也不能为了提高 alpha 无条件制造不透明前景。
- 外部 depth：开启时在有效正深度上使用鲁棒 log-depth 误差，统一 scene-scale plane-z 定义，不把射线距离、未归一化累积深度或逐图归一化值混用。
- 外部 LaMa normal：开启时按同相机、同朝向、单位向量计算有符号 `1-dot`；支持 normal-only 外部监督，不额外读取 completed depth。
- 内部几何：模型渲染 depth 派生法线与 Gaussian rendered normal 一致性，以及可靠可见区域的多视角重投影一致性；内部项与外部监督分别记录，不把它们解释成独立真值。

LaMa RGB/depth/normal 是可能相互矛盾的伪目标，不能全图等权视作真实观测。分别记录目标有效性与质量权重；权重停止梯度，不能让模型靠改变权重逃避约束。关闭某个外部监督时不读取相应数组，也不用于另一项权重；`normal_valid` 只说明向量合法，不是可信度。洞内 removed alpha 低是补全原因，不能据此否决 hole 中所有目标。

调度使用从 0 开始的本次联合训练步数 `t`，RGB/已知区/覆盖项先生效；几何项使用 `a(t)=clip((t-s)/r,0,1)`，其中 `s=EDGS_GEOMETRY_FROM_ITER`，`r=EDGS_GEOMETRY_RAMP_ITERS`，`r=0` 时在 `t>=s` 直接启用。关闭监督的项不构造计算图，避免 `0*NaN`；其有效权重固定为 0。独立 YAML 支持逐项权重，当前几何项共享本地渐增调度，不继承全局 7000 步门槛。debug 默认开启，保存固定视角 RGB/depth/normal/alpha、mask、单视图 loss/权重 JSON 和原始 NPZ；不存在的外部目标标为 N/A，不制造占位真值。深度预览逐图用有效值分位数着色，定量比较须使用 raw NPZ。

### J5. 分阶段依赖、产物与恢复

预检按入口阶段分层：公共 Stage 1..3 仍验证上游声明的完整 completion；Stage 4 的数值输入仅取初始化实际启用的模态，Stage 5 仅加载训练实际启用的外部目标。读取共同 manifest 的来源身份，不等于允许用关闭模态的数组参与初始化。新路径必须具备不强制读取 depth/normal 的数据加载器，不能直接复用 Stage 5b 的强制三模态 `read_targets`。

独立输出如下，不覆盖已有模型、支持点或局部优化目录：

```text
<INPAINT_RUN_ROOT>/
├── edgs_init/
│   ├── config.resolved.json
│   ├── support.npz                 # 匹配/补点/融合来源；不是 D0 sidecar
│   ├── editable_mask.npy           # 对应初始模型行序
│   ├── point_cloud.ply
│   └── diagnostics.json
├── edgs_joint/
│   ├── config.resolved.json
│   ├── checkpoints/                # 参数、optimizer、t、RNG 与来源身份
│   ├── point_cloud/iteration_5000/point_cloud.ply
│   └── debug/
└── manifests/
    ├── edgs_init_manifest.json
    └── edgs_joint_manifest.json
```

初始化身份绑定：pipeline/实现版本、removed 模型及 variant、target/surrounding IDs、原始行映射、相机/N/顺序/坐标尺度、tracking masks、共同 completion ID、实际消费模态及 hash、匹配器权重、配对和初始化配置/随机种子。联合训练身份另外绑定初始模型/编辑归属、监督模态及配置、训练目标、optimizer/调度/步数和输出 hash。normal 的坐标与朝向、depth 的定义必须显式记录，不能根据目录名猜。

只改变训练配置时，可验证并复用相同身份的初始化，但必须创建/验证新的训练身份；改变初始化模式、RGB、相机或 masks 不能复用旧点云。相同输出目录参数不一致时明确拒绝，推荐新 run。失败不提交 complete，不回退旧路径；保存诊断和可恢复 checkpoint，不覆盖原有结果。

保留整数 Stage 1..8：`END_STAGE<=3` 不运行或要求初始化/训练产物；Stage 4 完成新初始化；Stage 5 只运行新联合训练。`START_STAGE=5` 验证初始化，`START_STAGE=6` 验证联合训练完成。新路径发布标签使用实际 `EDGS_INPAINT_ITERATIONS`，不借用旧 `FINETUNE_ITERATION` 假装已执行 RGB finetune；下游从 model manifest 读取所选标签及源 PLY。

Stage 6 按 `INPAINT_PIPELINE` 分派 producer 校验：旧路径仍验证自己的原链；新路径验证 `edgs_init_manifest → edgs_joint_manifest`，不要求 fusion/5a/5b receipt。两者共同验证 PLY 字段、SH 阶数、classifier、语义与背景保留、相机/场景兼容，Stage 7/8 必须绑定本次发布 artifact，拒绝过期 mesh。初始化/训练分支不新增第二套 normal completion 标记。

### J6. 代码落点与实现顺序

以下入口已新增/扩展；复用共享纯函数，不通过更改全局默认值实现新行为。

| 落点 | 已接通职责 |
|---|---|
| `scripts/paintmesh/run_inpaint.sh` | selector、冲突预检、按路径调度 Stage 4/5/6、独立参数与动态 N |
| `scripts/paintmesh/edgs_inpaint_io.py`、`configs/edgs_inpaint.yaml` | 新路径配置解析、模态契约、stage request/receipt 与身份验证 |
| `submodules/EDGS/tools/initialize_paintmesh_edgs.py`、`source/paintmesh_edgs_init.py` | EDGS 局部 RGB 匹配、可选 depth/normal 混合初始化及来源记录 |
| `submodules/EDGS/tools/train_paintmesh_pgsr.py` | 独立 PGSR 联合训练入口、保存/恢复和完成提交 |
| `submodules/EDGS/source/paintmesh_joint_data.py`、`paintmesh_joint_losses.py` | 可选目标加载、新点全参数 optimizer、背景冻结、鲁棒损失与调度 |
| `publish_inpainted_edgs_model.py`、`finalize_inpaint_result.py` 及调用它们的脚本 | 按 producer 校验、源 PLY/迭代标签传递、model/mesh/final 身份链 |

落地顺序：配置/条件加载与失败契约 → RGB-only 初始化 → depth 辅助 → normal 定向 → PGSR 联合训练 → 发布/恢复/debug → 合成及真实场景消融。不能先把开关显示在 shell 中却仍无条件走 seed/5a/5b。

### J7. 验收清单与验证边界

已运行的自动检查覆盖：三种初始化、开关继承/冲突、禁用模态 loader 哨兵、改变关闭模态不影响初值、合成三角化/带尺度反投影/法线 quaternion、新点 SH/opacity 等参数梯度、背景与语义保留、debug 缺失目标 N/A、四种监督组合的真实 PGSR CUDA 三步训练/中断恢复/复用，以及发布 producer 分流。下列综合验收项还包含完整场景要求，因此未全部勾选。

还以缓存的真实 RoMa 权重运行了两视图合成场景的完整初始化 worker，验证 RGB-D-N 初值、normal 定向、初始化 receipt 和复用；未替换网络为假匹配器。该项用 `PAINTMESH_ROMA_GPU_TEST=1` 单独启用，需要已有 RoMa outdoor/DINOv2 缓存。

2026-09-21 回归结果：PaintMesh 测试（同时开启上述 RoMa 与 PGSR CUDA 检查）77 passed / 6 skipped；Inpaint360GS tools 测试 148 passed / 5 skipped / 16 subtests passed；EDGS 测试 78 passed。Shell 语法和三个仓库的 diff 空白检查通过。跳过项不计入通过项，这些结果不替代完整场景训练与视觉验收。

- [ ] 三种初始化模式和默认值/继承正确；初始化 normal-only 报错；训练 normal-only 合法。新 selector 与 `VIRTUAL_RENDERER`/轨迹相互独立，旧默认路径输出不变。
- [ ] 初始化/训练数据层的“禁止读取”测试：被关闭的模态 loader 一旦被调用即失败；保持 RGB/相机/随机种子不变、改变未消费的 depth/normal 数组，初始化数值结果不变。共同 completion 的 hash 校验另测，不能用伪造 receipt 绕过公共校验。
- [ ] 开启模态缺文件、坐标不符、全无有效目标时清楚失败；局部少量无效 normal 仅回退未定向初值并统计。无 RGB 匹配候选时，RGB-only 不自动补 depth；RGB-D 模式记录 depth-fill 实际占比。
- [ ] 合成正对/倾斜平面与带尺度相机测试三角化、depth 点位、normal 变换、四元数轴及多视角符号；去重后无 N 层堆叠，稀疏匹配与重复匹配的预算/资源行为可解释。
- [ ] 新点 XYZ/rotation/scale/SH/opacity 有预期梯度；多步训练后背景、语义/classifier 不变，点数/行序不变；目标未覆盖处不出现 NaN，禁用 loss 不读取目标或参与权重。
- [ ] 渐增边界、空有效域、checkpoint 恢复、旧 receipt/错误 producer/过期 mesh 拒绝；新路径从 Stage 4/5/6 续跑，不能隐式执行 5a/5b；全局 `run_seg` 回归不受影响。
- [ ] 默认开启的新路径 debug 可检查初值、中间、最终的 RGB/depth/normal/alpha；显示实际启用模态、匹配/补点贡献、点数、loss 权重、hole 覆盖和洞外变化，不以目标误差小声称几何正确。
- [ ] 真实 kitchen 同一相机序列/同一 completion 下，对比旧路径及新 RGB-only/RGB-D/RGB-D-N。先固定训练监督比较初始化，再固定初始化比较外部监督，区分两种收益；报告匹配覆盖、局部密度/间距、浮点/重影、normal 接缝、RGB 外观和 mask 外扰动。LaMa 伪目标一致性不是隐藏表面真值评测；没有 GT 时明确局限。

验证命令（两个仓库的同名 Python 包分开运行）：

```bash
PAINTMESH_JOINT_GPU_TEST=1 conda run --no-capture-output -n paintmesh \
  python -m pytest -q scripts/paintmesh/tests/test_edgs_inpaint.py

PAINTMESH_ROMA_GPU_TEST=1 conda run --no-capture-output -n paintmesh \
  python -m pytest -q scripts/paintmesh/tests/test_edgs_inpaint.py -k real_roma

PYTHONPATH="$PWD/submodules/Inpaint360GS:$PWD/submodules/Inpaint360GS/seg/detectron2" \
conda run --no-capture-output -n paintmesh \
  python -m pytest -q submodules/Inpaint360GS/tools/tests

PYTHONPATH="$PWD/submodules/EDGS" conda run --no-capture-output -n paintmesh \
  python -m pytest -q submodules/EDGS/tests
```

2026-09-21 另对 `target_14_hemi90_upright` 的真实 90 帧 completion 做了三种模态加载检查；仅用第 0/1 帧、缓存的 RoMa outdoor/DINOv2 权重执行真实匹配，8192 个洞内双向对应通过两视图几何检查。没有改动该 run 的产物，没有执行完整 90 帧初始化或 5000 步训练。发布分流测试使用合成 producer；全链真实 Stage 4..8 的视觉质量仍需评估。

新增双向置信度门槛后，对 `edgs_rgbdn_joint` completion 的第 0/1 帧只读复测：同一份 RoMa 输出有 36,961 个洞内候选，双向最小分数中位数约 0.197。固定 cycle=3 / reprojection=2 时，阈值 0.5、0.8、0.9 分别有 4,994、69、1 个匹配通过回环与三角化。阈值 0 时有 31,423 个通过回环，再按预算采样 8,192 个。由此将默认硬门槛设为 0.5，较高门槛作为显式选项；这一单图对计数不代表整个场景的正确率或最佳参数。旧产物未改写，未重新执行完整初始化或训练。

门槛变更后的回归结果：PaintMesh 91 passed / 6 skipped（启用真实 RoMa 与 PGSR CUDA 检查），Inpaint360GS tools 148 passed / 5 skipped / 16 subtests passed，EDGS 78 passed。新增覆盖双向门槛、反向目标位置插值、匹配图/原图分辨率换算、无效分数/坐标、阈值先于采样、禁止低分补满、CLI/YAML 优先级、三角化误差上限与阈值变化拒绝旧初始化复用。

### J8. 运行方式

在仓库根目录执行，必须已有相应 removal run 的完整 tracker masks。建议使用新的 inpaint run，不能把已有 5a/5b 输出目录切换成新路径。以下开启 RGB-D-N 初始化，训练监督默认继承为 depth+normal：

```bash
REMOVAL_ROOT="$PWD/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14_hemi90_upright" \
INPAINT_RUN_NAME=edgs_rgbdn_joint_conf05 \
INPAINT_PIPELINE=edgs-pgsr \
EDGS_INIT_USE_DEPTH=true \
EDGS_INIT_USE_NORMAL=true \
EDGS_MATCH_CONFIDENCE_MIN=0.5 \
EDGS_INPAINT_ITERATIONS=5000 \
EDGS_GEOMETRY_FROM_ITER=100 \
EDGS_GEOMETRY_RAMP_ITERS=2000 \
LOCAL_GEOMETRY_REFINE=false \
SUPPORT_DENSITY_MODE=legacy \
END_STAGE=8 \
bash scripts/paintmesh/run_inpaint.sh mip-nerf/360_v2 kitchen 8 14 none 1
```

其他模式使用不同的 `INPAINT_RUN_NAME`，只替换初始化开关：

| 模式 | `EDGS_INIT_USE_DEPTH` | `EDGS_INIT_USE_NORMAL` |
|---|---|---|
| EDGS RGB-only | `false` | `false` |
| EDGS RGB + depth | `true` | `false` |
| EDGS RGB + depth + normal | `true` | `true` |

如需仅比较初始化而保持后续监督相同，三组显式设置 `EDGS_TRAIN_USE_DEPTH=true EDGS_TRAIN_USE_NORMAL=true`。若环境里曾设置 `INPAINT_RUN_ROOT`，请取消或指向新的独立目录，不能仅改 run name 却仍覆盖旧根目录。`END_STAGE=4` 先检查初始化；完成后最后一个位置参数改为 `5` 可继续训练。debug 位于 `<INPAINT_RUN_ROOT>/edgs_joint/debug/`，初始化来源统计位于 `edgs_init/diagnostics.json`。

旧 `edgs_rgbdn_joint` 等 run 中的点不会因为提高阈值而自动被删除。使用新的 run name **及新的/未设置的 `INPAINT_RUN_ROOT`** 从 Stage 1 建立独立产物；不要在已有 Stage 5 checkpoint 上追加匹配阈值。该参数仅属于同级 EDGS-PGSR inpaint 路径，不改动全局 `run_seg` 或原 Inpaint360GS renderer。

---

## 十、阶段 G：最终 PGSR render 和 mesh

Stage 7 始终使用 Stage 6 实际发布的 PLY：原路径局部优化关闭时是 5a，开启时是完成验证的 5b；路径 J 选择通过独立验证的联合训练输出。TSDF 算法与真实相机选择不变。以下最终真实视角 raw 导出仍为后续扩展，不是 Stage 5b 或路径 J 训练的前置条件；独立训练的诊断可直接保存 PGSR 返回的 raw 张量。

扩展 [`EDGS/render.py`]( /home/martin/code/gsagent/submodules/EDGS/render.py:443)，保留已有 PNG，同时增加原始数组：

```text
inpainted_3dgs/train/ours_5000/
├── renders/
├── renders_depth/
│   └── *.png                 # 可视化
├── renders_depth_raw/
│   └── *.npy                 # PGSR plane depth
├── renders_normal/
│   └── *.png                 # 可视化
├── renders_normal_raw/
│   └── *.npy                 # HxWx3 float32
├── renders_alpha/
│   └── *.npy
└── render_manifest.json
```

TSDF 仍然使用：

```text
PGSR RGB + PGSR plane_depth
```

不要直接把二维 normal 融入 TSDF。Open3D 提取 mesh 后重新计算 mesh vertex normals，再将 mesh normal 投影回视图，与 PGSR rendered normal 比较：

```text
normal angular error
depth-normal consistency
normal seam error
```

另外，当前 normal visualization 对无效零法线可能显示成灰色。应使用 alpha/depth validity，把无效区域显示为黑色。

---

## 十一、`run_remove.sh` 和 `run_inpaint.sh` 的改造

建议保持旧的 8 个 stage 编号，增加子阶段，避免破坏旧 RGB/depth 结果。

### `run_remove.sh`

Stage 4 改为：

```text
4a. 生成或验证 virtual_cameras.json（已接入 circle/hemisphere + 总数 N，默认 circle/30）
4b. 解析 VIRTUAL_RENDERER，检查模型与请求模态是否受支持
4c. 统一入口调用所选后端，渲染 full 和 removed，提交 render manifest
4d. 用所选后端的 removed RGB 打包 tracker archive
4e. 建立与 render artifact 绑定的 tracking session
```

### `run_inpaint.sh`

normal completion 根据上游数据自动执行，无需额外配置。默认原生分支没有 normal 时继续现有 RGB/depth 路径；PGSR 分支有 normal 时必须在同一个 Stage 3 完成 normal LaMa。虚拟视角 backend 不隐式启用局部优化。下面列出原 `inpaint360gs` 路径：Stage 5b 及其发布绑定已接入，Stage 4 与 seed 初始化保持原样。新 `INPAINT_PIPELINE=edgs-pgsr` 的 Stage 4/5/6 分派已按 J1/J5 接入，不使用下面的旧调用链。

本入口已从上游 manifest 读取 N，将下面当前的 30 个支持 PLY、LaMa 帧集合和训练相机集合统一为 N；不在 inpaint 阶段重新选择轨迹，不改变单 seed 与 RGB-D 反投影算法。

```text
Stage 2:
    准备 RGB/depth 输入；自动检测并准备 normal/valid/推理 mask

Stage 3:
    RGB LaMa + depth LaMa + normal LaMa（有 normal 就必须执行）
    按 required modalities 统一验证并提交 completion manifest

Stage 4:
    原有 RGB-D 反投影 -> 30 个 support PLY（不变）

Stage 5a:
    原有单 seed 初始化 + RGB 3DGS finetune（算法不变）
    局部优化开启时额外记录最终 editable mask 与 5a manifest

Stage 5b:
    仅 LOCAL_GEOMETRY_REFINE=true 时运行独立 PGSR 局部优化
    LaMa normal + depth + 内部一致性，按局部步数渐增几何权重

Stage 6:
    按开关选择并验证 5a 或 5b，发布 inpainted EDGS model

Stage 7:
    对所选模型 PGSR render + TSDF；最终 raw normal 扩展另行实施

Stage 8:
    semantic lifting 和最终提交
```

与本批 completion 相关的既有配置保持不变：

```bash
VIRTUAL_RENDERER=inpaint360gs
NORMAL_ALPHA_MIN=0.01
```

`VIRTUAL_RENDERER` 是已有渲染后端选择，`NORMAL_ALPHA_MIN` 是已有渲染有效性阈值，都不是 normal completion 开关。不新增 `NORMAL_PIPELINE`、`NORMAL_COMPLETION_MODE` 或 `ENABLE_NORMAL`。新 `LOCAL_GEOMETRY_REFINE` 仅控制完成 LaMa 和原 RGB finetune 之后的几何训练；关闭它也不会跳过已有的 normal completion。

使用方式（必须已有完整 PGSR removal/tracker 输入）：

```bash
REMOVAL_ROOT="$PWD/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14_pgsr" \
INPAINT_RUN_NAME=normal_local_geometry \
LOCAL_GEOMETRY_REFINE=true \
LOCAL_GEOMETRY_ITERATIONS=1000 \
LOCAL_GEOMETRY_FROM_ITER=100 \
LOCAL_GEOMETRY_RAMP_ITERS=400 \
END_STAGE=8 \
bash scripts/paintmesh/run_inpaint.sh mip-nerf/360_v2 kitchen 8 14 none 1
```

以上使用新的 inpaint run，避免把已发布的 RGB-only 模型与局部几何版本混用。已有合法 Stage 1..4 输入可按既有续跑规则从 Stage 5 开始。此开关不传入 `run_seg.sh`，也不覆盖其 PGSR loss 配置。

已实现的统一虚拟渲染入口：

```bash
python scripts/paintmesh/render_virtual_views.py \
    --backend "$VIRTUAL_RENDERER" \
    --model-path "$WORK_MODEL_ROOT" \
    --edgs-model-path "$EDGS_MODEL_ROOT" \
    --source-path "$SCENE_ROOT" \
    --iteration "$DISTILL_ITERATION" \
    --resolution "$RESOLUTION" \
    --camera-manifest "$VIRTUAL_CAMERA_MANIFEST" \
    --tracker-archive "$TRACKER_ROOT/images.zip" \
    --inpaint-python "$INPAINT_PYTHON" \
    --edgs-python "$EDGS_PYTHON"
```

已实现的 normal completion 接口（由 shell 自动调用）：

```bash
python submodules/Inpaint360GS/LaMa/bin/predict_normal.py \
    --input-dir "$LAMA_NORMAL_INPUT" \
    --output-dir "$LAMA_NORMAL_OUTPUT" \
    --input-manifest "$LAMA_INPUT_MANIFEST" \
    --model-path "$LAMA_MODEL_PATH"
```

`predict_normal.py` 由 `run_lama` 在现有 LaMa 环境/工作目录中自动调用，相机读取 Stage 2 的验证快照，用户无需额外运行。`normal/prediction.json` 仅作为推理来源记录；三路仍由一个 LaMa completion manifest 验收提交。

---

## 十二、manifest 依赖关系

复用并扩展现有完成链（不新增平行的 normal-only 完成标记）。下面 fusion 和 5a/5b 记录专属于现有 `inpaint360gs` 路径：

```text
render_manifest.json                # 已实现，full/removed 各一份
lama_input_manifest.json            # 扩展 normal 输入与 required modalities
lama_completion_manifest.json       # RGB/depth/normal 一起验收和提交
fusion_manifest.json                # 本轮保留既有 RGB-D 契约
```

上面 fusion manifest 在本轮继续使用既有 RGB-D 契约，不新增 normal 反投影依赖。启用局部优化时，增加以下训练/发布链；它们是训练产物记录，不是第二套 normal completion 标记：

```text
Stage 5a PLY + editable_mask -> rgb_finetune_manifest.json
    + lama_completion_manifest + cameras/masks + resolved local config
    -> local_geometry_manifest.json
    -> model_manifest.json -> render/mesh/semantic manifests -> inpaint_manifest.json
```

关闭开关时上述两个训练 manifests 均不是旧产物的必需字段；开关开启时必须验证完整链，不能仅凭某个 `point_cloud.ply` 存在而判定几何优化成功。`local_geometry_manifest` 至少记录算法/配置版本、上游 artifact IDs、RGB 与局部步数、启用/渐增参数、目标权重、编辑行 hash、输入输出 PLY hash、语义字段保留检查、覆盖/越界回退统计和完成状态。

路径 J 已接通的独立依赖链为：

```text
同一 lama_completion_manifest + removed 模型/语义 + cameras/masks
    + EDGS 初始化配置/匹配器身份/实际消费模态
    -> edgs_init_manifest.json
    + PGSR 联合训练配置/实际监督模态
    -> edgs_joint_manifest.json
    -> model_manifest.json -> render/mesh/semantic manifests -> inpaint_manifest.json
```

两个训练 producer 的验证分支互斥，公共发布检查不变。Stage 6/8 按记录的 pipeline 与本次请求交叉验证，不能只凭目录名选择来源，也不能要求新路径提供旧 fusion 或 RGB finetune 记录。详见 J5。

各阶段 manifest 记录自身数据及上游 artifact IDs；以下是 PGSR normal 分支的字段示意（H/W 是待替换的数值）：

```json
{
  "backend": "edgs-pgsr",
  "capabilities": ["rgb", "depth", "alpha", "normal"],
  "normal_source": "rendered_normal",
  "camera_manifest_id": "...",
  "model_artifact_id": "...",
  "removal_variant": "all_selected",
  "frames": ["00000", "00001"],
  "shape": [H, W],
  "normal_space": "camera",
  "normal_orientation": "toward_camera",
  "normal_layout": "HWC",
  "depth_type": "plane_z",
  "depth_unit": "scene",
  "alpha_threshold": 0.01,
  "required_modalities": ["rgb", "depth", "normal"],
  "normal_completion_method": "lama",
  "normal_encoding": "xyz_to_rgb_affine_v1",
  "normal_completion_trigger": "upstream_normal_present"
}
```

按后端能力与阶段检查以下条件；normal 检查只适用于声明有 normal 的产物：

- 恰好为相机 manifest 声明的 N 个有序 frame，默认 N=30；
- RGB、depth、normal、alpha、mask 尺寸一致；
- 没有 NaN/Inf；
- 有效 normal 的模长接近 1；
- 无效 normal 等于 `[0,0,0]`；
- mask 外 normal 与 removed normal 完全一致；
- camera manifest ID 一致；
- 轨迹类型、N 及生效参数纳入相机身份；不能仅凭帧数相同就复用另一轨迹；
- removed model variant 一致。
- backend、模态能力和 depth/normal 定义一致，不复用另一后端的缓存。
- 所需模态由上游推导；有 normal 时，不能复用缺 normal 的旧 completion manifest。
- 模型权重、共同 hole mask、normal 推理 mask 与输入/输出 hash 都参与缓存身份。

上述字段为概念示意；实际输入 manifest 在 `parameters.normal` 保存 method/encoding 与来源，`parameters.required_modalities` 保存三路需求。原生分支为兼容旧缓存不新增该字段（缺省含义为 RGB/depth），不生成占位 normal。method 是固定来源说明，不是运行模式选择。

---

## 十三、建议的落地顺序

### P0：稳定的 normal artifact 流程（已实现）

```text
同级虚拟视角后端选择（inpaint360gs / edgs-pgsr）
+ PGSR raw normal render
+ alpha
+ 在既有 LaMa manifests 中绑定 normal
+ 自动 removed normal → LaMa → completed normal
+ normal 可视化和验证
```

不修改现有 RGB finetune，风险最低。

### P1：独立局部优化（已接通，真实场景质量待评估）

```text
保留原 RGB-D 反投影 + 单 seed 初始化 + RGB finetune
+ 5a 输出编辑范围/来源记录
+ 独立局部开关、配置和 PGSR 训练入口
+ LaMa normal loss + depth loss + depth-normal 一致性
+ 按局部步数启用和渐增 + 背景冻结/覆盖保护
+ 5b 产物、发布和恢复验证
```

数据/编辑范围契约、调度与可微训练已接入 Stage 5b/6/8 发布链。接下来优先评价真实场景法线与几何稳定性，不将密度提升当作本轮验收目标；全局 run seg 的行为保持不变。

### P2：无手调密度阈值的自适应配额采样（已实现，完整训练质量待评估）

已接入像素表面元、自动参考密度、缺点配额、分层采样、多视角软几何证据及身份链，使用 `SUPPORT_DENSITY_MODE=mass_adaptive`。已通过真实 kitchen 预处理和小型 CUDA 集成，详见 D0.10。取消角度/覆盖率/密度比门槛及事后删点，保留数据有效性、原 gate 和资源保护。位于 Stage 4/5a 交界，与 P1 独立，不更改 depth 定义或 normal completion。完整训练质量、跨层估计改进、D1 全视角 RGB-D-N 融合、E normal-aware 初始化和训练期密度保持仍为后续工作。

### P3：提升 LaMa normal 预测质量

```text
专门 normal LaMa checkpoint
或 RGB-D-N joint completion 网络
```

当前实现直接复用当前 Big-LaMa 权重；后续若质量诊断不达标，再评估专用 normal 训练或 joint 网络，不把训练专用权重作为自动 normal 分支落地的前置条件。是否采用新网络属于后续方案评审，不增加本批模式选择。

### P4：同级 EDGS-PGSR 初始化与联合训练（已接通，完整质量待验收）

J6 代码已接入，可运行接口与测试记录见 J7/J8。复用公共 completion，不以旧路径的 P1/P2 或延期的 D1/E 为前置依赖；未修改 Inpaint360GS 初始化器或全局 `run_seg`。下一步是真实完整初始化/训练与质量消融，不能把短训练和单对匹配测试解释为完整几何质量提升。

本文记录设计要求与实现顺序；所列新增入口、配置和 worker 在落地并通过对应验收前，均不代表已有运行能力。
