# 语义条件先验场：语义指导 3DGS 空间分布与 Mesh 计算的框架设计

> **实现状态**：本设计已完整落地为独立项目 `submodules/SemanticPriorField/`
> （内部包 `semantic_prior_field/`，基于 GaussianWrappingGaga fork）。
> 核心模块：`semantic/prior_field.py`（SPF 本体）、
> `regularization/regularizer/semantic_prior.py`（正则消费者）、
> `densification/semantic_error.py`（致密化消费者）、
> `extraction/semantic.py` + `extraction/mesh.py`（提取消费者）。
> 训练入口 `train.py --semantic_prior`，消融开关 `--sp_*`，
> 一键脚本 `scripts/train_and_extract_spf.py`，
> 单元测试 `tests/test_semantic_prior_field.py`（12 项，全部通过）。

> 阶段二设计。前置文档 `doc/gaussian-wrapping-gaga-native-semantics-design.md`（阶段一）解决的是
> "语义如何进入 Gaussian Wrapping"（原生 16 维语义通道，已在 `submodules/GaussianWrappingGaga`
> 实现并跑通）。本文档解决的是反方向的问题：**语义如何反过来指导高斯的朝向、密度、大小、
> 球谐函数以及 mesh 提取**，目标是 PSNR 与 mesh 指标（Chamfer/F1）双提升。

## 1. 结论

在 `GaussianWrappingGaga` fork 上增加一个核心抽象——**语义条件先验场
（Semantic Prior Field, SPF）**——和它的三个消费者。不新建训练管线、不替换光栅器、
不改 CUDA 语义通道。

SPF 的定义：

```text
SPF = { per-Gaussian 实例标签 + 置信度 (live, 由嵌入+分类头导出)
        per-instance 几何代理 (在线拟合: 平面 / 二次曲面 / 局部法线场 / 细结构)
        per-view 边界权重图 (由关联掩码腐蚀得到, robust_association 已产出 core/boundary) }
```

三个消费者，各自复用 Gaussian Wrapping 已有的一种惯用法（idiom），因此是框架性接入而非补丁：

| 消费者 | 指导对象 | 复用的 GW 惯用法 |
|---|---|---|
| 正则器 | 朝向、大小、球谐 | 四文件 regularizer 模式（`regularization/regularizer/*` + yaml + CLI + kick-on） |
| 致密化控制器 | 密度、大小、剪枝 | 法线误差 gidx 散射 + spoke-splitting（`densification/normal_error.py`、`normal_field.py:304-437`） |
| Mesh 提取器 | 网格拓扑与属性 | MT 任意属性插值（`pivots_colors`）+ bad-edge 谓词（`extraction/mesh.py:187-208`） |

闭环结构：

```text
关联掩码(静态, robust_association, 带 confidence/valid/boundary)
      │ CE 监督
      ▼
16 维语义嵌入 + 1×1 分类头 (live, joint 模式已存在)
      │ argmax + margin, no_grad, 周期刷新
      ▼
per-Gaussian 标签/置信度 ──→ per-instance 几何代理拟合
      │                                │
      ├─→ 正则器(朝向/SH/flatten)      ├─→ 致密化(跨界分裂/漂浮物剪枝/预算)
      │                                │
      └────────────→ 更好的几何 ───────┘ ──→ mesh 提取(identity 门控)
                     更好的几何 → 语义渲染更准 → 嵌入更准 (闭环)
```

关键设计约束（吸取 PlanarGS 的教训并推广）：**语义先验永远是软的、置信度门控的、
按区域施加的**。PlanarGS 的先验族只有 {平面}；SPF 的先验族按实例自适应选择，
且拟合残差高的实例自动退化为"无先验"（等价于当前基线），保证最坏情况不劣于基线。

## 2. 代码事实基础（两次精读的要点）

设计依赖以下已验证的机制，均给出文件位置（GW = `submodules/GaussianWrapping`，
GWG = `submodules/GaussianWrappingGaga`，二者路径结构相同，落地在 GWG）：

1. **GW 的"有向面元"来自可学习法线通道**：`_gaussian_features` (N,4)，
   `convert_features_to_normals` 解码方向+符号（`scene/gaussian_model.py:465-497`）；
   法线场对齐损失在 20001 迭代启动（`regularization/regularizer/normal_field.py:70-162`）。
   → 语义代理法线可以直接作为该通道的额外监督，插入点现成。
2. **GW 已有"按误差信号驱动的致密化"惯用法**：`compute_normal_error`
   （`densification/normal_error.py:16-169`）用 `render_depth` 的 `gidx`（每像素最大贡献高斯索引）
   把逐像素误差 `index_add_` 散射到高斯上，取 top 5% 分裂；分裂时体积守恒、
   沿法线方向优先收缩（`normal_field.py:358-374`）。
   → 把误差信号从"法线误差"换成"语义误差"，机制完全复用。
3. **Mesh 提取天然支持携带任意 per-pivot 属性**：Marching Tetrahedra 用 SDF 权重插值
   `pivots_colors`（`extraction/mesh.py:75-86, 132-136, 184-185`），CLI 目前传 `None`
   （`pivot_based_mesh_extraction.py:396`）；bad-edge 过滤/塌缩机制已存在
   （`extraction/mesh.py:187-208`）。
   → identity 上 mesh、语义 bad-edge 谓词都有现成挂点。
4. **GWG joint 模式已实现**：`--semantic_masks` 下语义 CE 与光度损失同图训练，
   嵌入随致密化/剪枝完整生命周期管理，梯度策略为 embedding-only
   （GWG `train.py:359-383`，`docs/architecture.md:39-46`）。
   → 阶段二只需把梯度策略从"语义不动几何"升级为"语义经 SPF 显式地、受控地动几何"，
   而不是打开语义 CE 对 alpha/conic 的原始梯度（那是不可控的）。
5. **robust_association 已产出先验场的原料**：per-pixel confidence/valid/ignore、
   core/boundary 图（`scripts/gaga/robust_association/observations.py:16-33`）、
   per-Gaussian 共识标签+置信度（`refinement.py:318-377`，counter 上 76.7% 高斯有稳定标签）、
   QA 门（purity/ignore/jump-rate）。
   → 边界权重图与"identity 不稳定 ⇒ 漂浮物"信号是现成的。
6. **多视图 NCC/几何损失按 patch 计算**（`regularization/multiview_gggs.py:425-455`），
   PGSR 图像梯度加权已有挂点（`train.py:341-345`）。
   → 跨实例边界的 patch 是边缘伪影的经典来源，语义边界权重可无缝并入。

## 3. SPF 的构建（一个新模块）

新模块 `gaussian_wrapping/semantic/prior_field.py`（放在 GWG 已有的 semantic 包内）。

### 3.1 live per-Gaussian 标签

- 每 `refresh_interval`（默认 500）迭代，在 `no_grad` 下：
  `posterior = softmax(classifier(get_semantic_features))`，
  `label = argmax`，`conf = margin(top1 - top2)`；`conf < τ_label`（默认 0.3）者标记为无标签。
- 缓存失效复用 GW 已有钩子：任何拓扑变化后 `gaussians_have_changed`
  （`train.py:551-662`）同时重置 SPF 缓存（与 normal-field/MILo 缓存并列）。
- 不使用 robust_association 的静态 per-Gaussian 共识标签做训练期指导
  （那是关联期模型的高斯，与训练中的高斯不对应）；静态共识只用于初始化验证与离线诊断。
  2D 掩码（静态、鲁棒）才是监督源，per-Gaussian 标签始终 live 导出——这是闭环成立的前提。

### 3.2 per-instance 几何代理（先验族自适应）

对每个像素占比超过阈值的实例 k（其余实例归入"无先验"）：

1. 取该实例高斯的 `xyz`（置信度加权），RANSAC 拟合平面；
   内点率 ≥ 0.8 且残差 < ε → 代理 = **平面**（法线场 = 常量）。
2. 否则拟合二次曲面；拟合优度达标 → 代理 = **二次曲面**（法线场 = 解析梯度）。
3. 否则统计实例内高斯各向异性与最小轴分布：
   高度各向异性（辐条/栏杆/植物类）→ 代理 = **细结构**（无法线先验，
   但携带"保护标记"：禁止对其施加 flatten 与大尺寸剪枝）。
4. 兜底 → 代理 = **自由曲面**（局部 MLS 平滑法线场，仅提供弱平滑先验）。

每个代理输出：`normal_at(x)`、`fit_residual`（决定先验权重
`w_k = conf_instance × exp(-fit_residual/σ)`）、`prior_type`。
这是 `SemanticGaussianWrapping` 五专家思想（planar/curved/thin/freeform/fuzzy）的
数据驱动简化版：不学 `geometry_logits`，直接由拟合优度判型，无额外参数、可解释、可关断。

### 3.3 per-view 边界权重图

由关联掩码 8 邻域腐蚀生成（robust_association 的 `reliable_core_map` 逻辑直接搬用），
预计算缓存，`B(p) ∈ [0,1]`，边界处 → 1。

## 4. 四个自由度 + Mesh 的指导机制

### 4.1 朝向（orientation）

- **代理法线先验损失**（新 regularizer `semantic_prior.py`）：
  `L_orient = Σ_i w_{k(i)} · (1 - |<n_i, n_proxy(x_i)>|)`，
  `n_i` 来自 `convert_features_to_normals`。对平面/二次曲面代理的实例生效；
  细结构/自由曲面实例权重为零或极小。与法线场损失（20001 起）同窗口启动。
  作用：GW 的法线监督来自 median-depth 法线（有噪声、纹理弱区域不可靠），
  代理法线在纹理弱的平坦区（墙/桌面/地面）恰好最可靠——两者互补。
- **边界掩蔽的深度-法线一致性**：`train.py:304-349` 的一致性损失乘以 `(1-B(p))^2`，
  与已有 PGSR 图像梯度加权（`train.py:341-345`）取乘积。
  实例边界处深度不连续是合法的，当前损失在此处强行平滑，同时伤 PSNR（边缘糊）和
  mesh（边界圆角）。
- **边界门控的多视图 NCC**：patch 内掩码标签不纯（纯度 < 0.9）的 patch 权重衰减
  （`regularization/regularizer/multiview.py` 入口处按 patch 中心查掩码即可，无需改 PatchMatch 内核）。

### 4.2 密度与大小（density & size）

三个机制，全部走 GW 的误差散射惯用法：

- **跨界分裂（semantic-error splitting）**：新建 `densification/semantic_error.py`，
  镜像 `compute_normal_error`：对每个训练相机渲染语义特征 → 分类 → 与关联掩码逐像素
  CE（或预测熵），经 `gidx` `index_add_` 散射到高斯、按 count/area 归一。
  高语义误差高斯 = 足迹横跨两个实例（嵌入被两边拉扯）或漂浮物。
  取 top-q（默认 5%，与法线误差一致）分裂，复用体积守恒分裂器，但分裂方向沿
  **局部边界切向的垂直方向**（用该高斯足迹内掩码边界的 2D 方向反投影估计；
  拿不到时退化为最长轴方向）。调度：与 normal-field densification 同窗（22000–26000 每 1000），
  错峰 500 迭代避免同帧拓扑抖动。
  这是 PSNR/mesh 双赢通道：RGB 边缘更锐利，mesh 边界 F1 更高。
- **identity 稳定性剪枝**：SPF 标签刷新时顺带统计每个高斯的
  "可见性高但标签置信度低"集合（`denom`/`get_average_contribution` 高、`conf` 低）。
  该集合与 GW 非极大剪枝（`normal_field.py:413-437`）取**交集**剪除——
  双条件保守门控，避免误杀真实表面。漂浮物同时是新视角 PSNR 与 mesh 噪声/悬浮碎片的来源，
  也是双赢通道。
- **按先验类型调制致密化阈值与尺寸**：给 `densify_and_prune_radegs`
  增加可选 per-Gaussian 阈值乘子 `τ_i = τ · m_{prior(i)}`：
  平面实例 m=1.5（少分裂，省预算），细结构实例 m=0.7（多分裂），无先验 m=1。
  省下的预算自然流向光度复杂区域——这是"语义重分配预算"的最简实现（一个 (N,) 张量、
  两处比较运算，不改选择逻辑）。
  **选择性 flatten**：PGSR min-scale 损失（`train.py:352-359`，现为全局开关且默认关）
  改为按 `w_k` 加权只作用于平面代理实例；细结构实例显式豁免。
  全局 flatten 是"为 mesh 牺牲 PSNR"的典型来源，按区域 gating 后该权衡被拆解。

### 4.3 球谐函数（SH）

- **区域内 SH 一致性**：复用 GWG `semantic/losses.py` 的 chunked-KNN-KL 机制
  （`spatial_consistency_loss`, :94-132），但作用对象改为 `_features_rest`：
  同标签 kNN 邻居间高阶 SH 的平滑正则。目的：阻止"几何误差被烘焙成视角相关颜色"——
  这是同时损害 mesh（几何错但光度对）与测试视角 PSNR（过拟合训练视角）的失败模式。
- **漫反射区域高阶 SH 衰减**：SPF 刷新时按实例统计视角残差
  （各训练视角渲染色与均值色的 L1，GW 的 contribution 采样机制可复用）；
  视角残差低的实例施加 `‖f_rest‖²` 衰减（权重 × w_k）。
  高光/金属/玻璃实例（视角残差高）不受任何约束，保留完整 15 阶容量。
  不做 per-Gaussian SH 阶数门控（需改光栅器，违反"不动 CUDA"边界；衰减在数学上等效且免费）。
- 天空/远景类（户外场景）：不进入代理拟合，标记为"无先验 + 免剪枝豁免"，
  避免误伤（MipNeRF360 室外场景的已知坑）。

### 4.4 Mesh 计算

- **identity 随 MT 上网格**：`pivot_based_mesh_extraction.py:392-403` 把
  per-Gaussian 语义嵌入（或 one-hot 标签+置信度）`repeat(1, n_pivots, 1)` 后作为
  `pivots_colors` 传入——MT 用与顶点相同的 SDF 权重插值，mesh 顶点直接获得语义。
  替换现有 `semantic_mesh.py` 的事后 cKDTree 近邻转移（后者在边界处系统性出错）。
- **语义 bad-edge 谓词**：`extraction/mesh.py:187-208` 现有 bad-edge 判据是
  `‖p0-p1‖ > scale0+scale1`；增加第二判据：两端 pivot 的标签不同**且**双方置信度均 > τ_edge。
  命中后按现有 `--filter_large_edges` / `--collapse_large_edges` 同样处理（丢弃面或向
  |SDF| 小端塌缩）。作用：消灭跨物体桥接（相邻物体间的"蹼"）和边界渗色——
  这是 TSDF/MT 类提取的经典伪影，长度判据抓不到（贴得近的两个物体边不长）。
- **按实例提取模式（可选 `--per_object`）**：SDF 求值时按标签子集分别评估
  （`integrate` 的输入高斯集合按标签划分），每实例独立 Delaunay+MT+二分精化，
  场景 = 各实例水密网格的并集。默认关闭（全场景指标用统一模式），
  服务于编辑/资产导出场景。
- 训练期几何被 4.1–4.3 改善后，SDF 场本身更干净（漂浮物剪除直接减少假表面），
  `min-over-views` 集成的 SDF 与二分精化的精度随之提升——mesh 指标的主要增益
  其实来自训练期，提取期两项是收尾。

## 5. 训练时间表（并入 GW 现有 timeline）

| 迭代 | 事件 | 归属 |
|---|---|---|
| 0– | joint 语义 CE（embedding-only，已存在） | GWG 现状 |
| 7000 | 深度-法线一致性、多视图 NCC 启动 → **同时启用边界掩蔽/门控**（只需掩码，不依赖嵌入收敛） | 4.1 |
| 3000 起每 500 | SPF 刷新（标签→代理→统计），拓扑变化即失效重算 | 3 |
| 7000–15000 | per-prior 致密化阈值乘子生效 | 4.2 |
| 15000 | identity 稳定性剪枝首轮（与 Mip filter 刷新同帧） | 4.2 |
| 15000– | 选择性 flatten、漫反射 SH 衰减、区域 SH 一致性 | 4.2/4.3 |
| 20001 | 法线场启动 → **代理法线先验同窗启动** | 4.1 |
| 22000–26000 | normal-field densification 与 semantic-error splitting 交替（错峰 500） | 4.2 |
| 30000 | 保存；提取期启用 pivots_colors + 语义 bad-edge | 4.4 |

原则：**每个语义-几何通道启动时，其依赖信号必须已可靠**（掩码类通道 7000 即可，
嵌入类通道 15000 后，代理法线随法线场 20001）。

## 6. 为什么 PSNR 与 mesh 能双升

几何正则与 PSNR 的对立（SuGaR/2DGS/PGSR 的共同现象）根源是**全局先验在先验为假的
区域也生效**。SPF 把一个全局 bias–variance 权衡拆成逐区域权衡。分通道核算：

| 通道 | PSNR | mesh | 风险与门控 |
|---|---|---|---|
| 漂浮物剪枝 | ↑（新视角浮渣消失） | ↑（假表面消失） | 误杀真表面 → 双条件交集 + contribution 门槛 |
| 跨界分裂 | ↑（边缘锐化） | ↑（边界 F1） | 高斯数膨胀 → top-q 限额 + N_max |
| 边界掩蔽一致性/NCC | ↑（边缘不再被强行平滑） | ↑（边界不圆角） | 掩码错误 → 权重来自 confidence 图 |
| 代理法线先验 | ≈（纹理弱区本就欠约束） | ↑↑（平坦区法线收敛快而稳） | 代理拟合错 → w_k 随残差指数衰减 |
| 选择性 flatten | ≈/↑（相对全局 flatten 是净解除约束） | ↑ | 误判平面 → 仅 RANSAC 高内点率实例 |
| 区域 SH 正则 | ↑（测试视角，抗过拟合） | ↑（防几何误差烘焙） | 材质突变实例 → 仅低视角残差实例 |
| 预算重分配 | ↑（预算流向光度复杂区） | ↑（细结构密度） | 类型误判 → 乘子范围钳制 [0.7, 1.5] |

没有一个通道是"用 PSNR 换 mesh"的；每个通道要么双赢，要么单边收益+另一边中性，
且都有独立开关（消融矩阵见 §8）。这继承并简化了 `SemanticGaussianWrapping` 的
Pareto guard 思想：不做梯度手术，靠**置信度门控 + 通道级开关 + 启动调度**保护主目标。

## 7. 插入点清单（GWG fork 内）

| # | 改动 | 位置 | 性质 |
|---|---|---|---|
| 1 | `semantic/prior_field.py`（标签/代理/边界图/统计） | 新文件 | 新模块 |
| 2 | `regularization/regularizer/semantic_prior.py` + `configs/semantic_prior/default.yaml` + CLI/kick-on | 四文件模式（模板：`mesh_in_the_loop.py`） | 新 regularizer |
| 3 | 边界权重并入深度-法线一致性 | `train.py:304-349`（乘权重，`:341-345` 旁） | ~10 行 |
| 4 | patch 纯度门控 | `regularization/regularizer/multiview.py:64-161` 入口 | ~20 行 |
| 5 | `densification/semantic_error.py` | 镜像 `densification/normal_error.py` | 新文件 |
| 6 | semantic splitting + identity 剪枝调度 | `train.py` 致密化段（`:537-577` 旁）+ `normal_field.py:304-437` 模式 | 中等 |
| 7 | per-Gaussian 阈值乘子 | `gaussian_model.py:1884-1922`（`densify_and_prune_radegs` 加可选参数） | ~15 行 |
| 8 | 选择性 flatten | `train.py:352-359`（权重从标量改 (N,)） | ~5 行 |
| 9 | SH 区域正则/衰减 | `train.py:351-471` 损失段 + 复用 `semantic/losses.py` KNN 机制 | 小 |
| 10 | `pivots_colors` 传语义 | `pivot_based_mesh_extraction.py:392-403` | ~10 行 |
| 11 | 语义 bad-edge 谓词 | `extraction/mesh.py:187-208` 加谓词参数 | ~20 行 |
| 12 | SPF 缓存失效 | `train.py:551-662`（`gaussians_have_changed` 旁） | ~5 行 |

注意事项（来自精读）：新增 per-Gaussian 张量必须 (N,D) 形状（SparseGaussianAdam
CUDA 约束）；SPF 的统计缓冲不进优化器（仿 `filter_3D` buffer 模式），不必走
densification postfix 的 7 个站点——分裂/剪枝后直接整体重算（刷新本来就便宜）。

## 8. 不做什么

1. **不打开语义 CE 对几何的原始梯度**（alpha/conic）。语义对几何的影响全部经 SPF
   显式通道，可解释、可关断、可消融。embedding-only 保持为语义通道的梯度策略。
2. **不改 CUDA**。全部机制在 PyTorch 层；语义光栅化通道维持阶段一的 16 维现状。
3. **不引入固定语义类别表**。SPF 是 class-agnostic 的：先验类型由拟合优度判定，
   不查"墙=平面"字典（Gaga 本身就是 class-agnostic 的，保持一致）。
4. **不做硬约束**（吸附到代理面、量化朝向等）。掩码和嵌入都有错误率，
   硬约束会把 2D 错误固化成 3D 伪影。
5. **不以 `SemanticGaussianWrapping` 为基线**（沿用阶段一决定），
   但迁移其三个思想的简化版：几何专家（→ 代理判型）、观测适配（已迁）、
   Pareto 保护（→ 门控+调度）。

## 9. 验证与消融

- **指标**：MipNeRF360 PSNR/SSIM/LPIPS（GW `metrics.py`）+ DTU Chamfer
  （`evaluate_dtu_mesh.py`）/ TnT F1 + Gaga 分割指标（`semantic_eval.py`，
  确保几何指导不反噬分割质量）。
- **消融矩阵**：每通道独立 flag（`--sp_orient / --sp_split / --sp_prune /
  --sp_budget / --sp_flatten / --sp_sh / --sp_mesh_edges / --sp_pivot_colors`），
  基线 = GWG joint 全关。先单通道验证（预期最大单项：剪枝与跨界分裂），
  再全开验证无相互抵消。
- **验收**：全通道关闭时与 GWG joint 数值一致（回归）；每通道开启后
  PSNR 与 Chamfer 不得同时变差（Pareto 验收准则）；counter 场景先行
  （已有 two-stage/joint 产物可直接对比）。
- **诊断产物**：SPF 每次刷新导出实例代理类型/拟合残差/权重直方图；
  被剪/被裂高斯的标签与位置快照——保证每个通道的行为可审计。
