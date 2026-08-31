# PaintMesh / Inpaint360GS 原理说明

本文解释当前仓库中 `scripts/paintmesh` 三条流水线背后的实际计算过程：

1. EDGS-PGSR 3DGS 与基础 mesh 如何构建；
2. 二维实例 mask 如何变成语义 3DGS 和语义 mesh；
3. 3DGS 与 mesh 如何完成目标移除和场景补全。

本文以当前代码为准，而不是对上游论文流程的泛化描述。运行命令、断点续跑和目录参数请参阅 [README.zh-CN.md](README.zh-CN.md)。

## 0. 先建立三个正确认识

### 0.1 “3DGS with mesh”不是联合训练

当前实现只训练 3D Gaussian。mesh 不含可学习参数，也不参与反向传播。3DGS 训练结束后，系统从训练相机渲染 RGB 和 PGSR 平面深度，再用 Open3D TSDF 融合生成 mesh：

```text
多视角 RGB + COLMAP
        │
        ▼
EDGS 初始化与 3DGS 优化
        │
        ▼
PGSR RGB / plane-depth 渲染
        │
        ▼
TSDF 融合与网格提取
```

因此，本项目中的 mesh 是 3DGS 的几何后处理结果。

### 0.2 “语义”是场景内实例，不是类别名称

CropFormer 先产生每张图片内部的实例 mask；跨视角关联后，系统得到仅在当前场景中有效的 instance ID。除非另外提供 ID 到类别名称的映射，否则 ID `14` 只表示“本次场景关联出的第 14 个实例”，不等于固定的 `chair` 或 `table` 类别。

- 二维和 3DGS 中的 `0` 是背景；
- mesh sidecar 中默认用 `65535` 表示未知或支撑不足；
- Gaussian PLY 里的 `obj_dc_0..15` 是 16 维 embedding，不是 16 个离散标签。

### 0.3 mesh 没有被直接删除或补洞

目标移除和补全首先发生在 3DGS 上。每次编辑结束后，系统都从编辑后的 3DGS 重新渲染真实训练视角，并从头进行 TSDF 重建：

```text
removed 3DGS   ──PGSR + TSDF──> removed mesh
inpainted 3DGS ──PGSR + TSDF──> inpainted mesh
```

不存在“在旧 mesh 上删除若干三角形，再对边界直接 hole filling”的步骤。

### 0.4 关键数据表示

每个 Gaussian (G_i) 包含：

| 字段 | 含义 | 激活后的物理量 |
|---|---|---|
| `xyz` | 三维中心 μᵢ | 直接使用 |
| `f_dc_*`, `f_rest_*` | RGB 球谐系数 | 由观察方向计算颜色 |
| `opacity` | 不透明度 logit | αᵢ = sigmoid(`opacity`) |
| `scale_0..2` | 三轴 log-scale | sᵢ = exp(`scale`) |
| `rot_0..3` | 四元数 | 归一化后得到旋转矩阵 |
| `obj_dc_0..15` | 可选的实例 embedding | 经线性 classifier 得到实例 logits |

相机提供内参 (K)、旋转 (R)、平移 (T)、图像宽高和 FoV。所有二维到三维以及三维到二维的操作都必须使用与图像严格对应的相机参数。

---

## 1. 现有的 3DGS with mesh 是如何训练的？

### 1.1 输入和输出

#### 输入

| 输入 | 作用 |
|---|---|
| `images/` 或指定的图像目录 | 多视角 RGB 监督与 RoMa 匹配 |
| `sparse/0/cameras.*` | COLMAP 相机内参 |
| `sparse/0/images.*` | COLMAP 相机外参与图像对应关系 |
| `sparse/0/points3D.*` | 初始 SfM 稀疏点云 |
| RoMa outdoor 权重 | 稠密两视图 correspondence |
| `configs/train.yaml` 与 `configs/gs/pgsr.yaml` | 初始化、优化器、renderer 和损失参数 |
| `resolution` | 加载图像时的缩放倍率 |

COLMAP 场景读取位于 [`dataset_readers.py`](../../submodules/EDGS/submodules/gaussian-splatting/scene/dataset_readers.py)。对于路径名包含 `360` 且启用 evaluation 的数据，排序后每第 8 张图片作为测试视角，其余作为训练视角。

#### 输出

```text
<PIPELINE_ROOT>/edgs/
├── config.yaml
├── cfg_args
├── chkpnt30000.pth
├── point_cloud/iteration_30000/point_cloud.ply
└── mesh/ours_30000/
    ├── tsdf_fusion.ply
    └── tsdf_fusion_post.ply
```

其中 `point_cloud.ply` 是训练出的 3DGS，`tsdf_fusion_post.ply` 是后处理后的基础 mesh。

### 1.2 模块 A：Scene 与临时 SfM Gaussian 初始化

#### 输入

- COLMAP 稀疏点位置和颜色；
- 相机集合；
- `sh_degree=3`；
- 背景颜色和分辨率配置。

#### 计算过程

[`Warper3DGS`](../../submodules/EDGS/source/networks.py) 创建 EDGS 的 `GaussianModel` 和 `Scene`。每个 COLMAP 点先被转成一个临时 Gaussian：

- 中心取稀疏点的 XYZ；
- RGB 转成 SH 的 DC 项，高阶 SH 初始化为 0；
- 尺度由邻近点距离估计；
- 旋转初始化为单位四元数；
- 初始实际 opacity 为 0.1。

实现位于 EDGS 的 [`gaussian_model.py`](../../submodules/EDGS/submodules/gaussian-splatting/scene/gaussian_model.py)。这批 SfM Gaussians 在当前配置中主要用于创建结构完整的模型，随后会被 RoMa correspondence 初始化结果替换。

#### 输出

一个临时的、由 COLMAP 稀疏点初始化的 `GaussianModel`。

### 1.3 模块 B：EDGS RoMa correspondence 初始化

入口依次是 [`train.py`](../../submodules/EDGS/train.py)、[`trainer.py`](../../submodules/EDGS/source/trainer.py) 的 `init_with_corr`，以及 [`corr_init.py`](../../submodules/EDGS/source/corr_init.py)。当前主要参数为：

```yaml
matches_per_ref: 15000
num_refs: 180
nns_per_ref: 3
scaling_factor: 0.001
proj_err_tolerance: 0.01
roma_model: outdoors
add_SfM_init: false
```

#### 输入

- 训练视角 RGB；
- 所有训练相机的投影矩阵；
- RoMa outdoor 模型；
- 上一步的临时 GaussianModel。

#### 计算过程

##### B1. 参考相机和邻居相机选择

代码将每个相机的 (4\times4) `world_view_transform` 展平，以 K-means 选出最多 180 个覆盖场景的参考相机；再按展平矩阵的欧氏距离，为每个参考相机选择 3 个近邻相机。

##### B2. 稠密两视图匹配

RoMa 对参考图像 (I_a) 和邻居图像 (I_b) 预测稠密 warp 与每像素 certainty。系统依据 certainty 无放回采样最多 15,000 个 correspondence。

##### B3. 两视图三角化

对匹配像素 (p_a=(u_a,v_a))、(p_b=(u_b,v_b)) 及投影矩阵 (P_a,P_b)，构造线性系统并使用 `torch.linalg.lstsq` 求三维点 (X)。再把 (X) 重投影到两张图：

$$
e_a=\left\|\pi(P_aX)-p_a\right\|_1,
\qquad
e_b=\left\|\pi(P_bX)-p_b\right\|_1.
$$

如果同一个参考像素由多个邻居得到多个候选点，则按

$$
e=\max(e_a,e_b)
$$

选择误差最小的解。

##### B4. correspondence 转成 Gaussian

每个三角化点初始化为：

$$
\mu_i=X_i,
\qquad
s_{ix}=s_{iy}=s_{iz}
=\|X_i-C_a\|_2\times0.001.
$$

颜色取参考图像的匹配像素并转成 SH DC；高阶 SH 为 0。若重投影误差大于 `0.01`，raw opacity 被设为 `-10`，使其实际 opacity 极低。

##### B5. 替换 SfM 初始化

因为 `add_SfM_init=false`，系统删除最初的 COLMAP/SfM Gaussians，只保留 RoMa 三角化得到的 Gaussians，并把保留点的实际尺度再缩小一半。

#### 输出

一个由稠密 correspondence 三角化初始化的 EDGS GaussianModel。与只从稀疏 SfM 点开始相比，它为后续表面几何优化提供更密集的初值。

### 1.4 模块 C：PGSR plane-aware renderer

`gs=pgsr` 通过 [`renderers/__init__.py`](../../submodules/EDGS/source/renderers/__init__.py) 选择 [`PGSRRenderer`](../../submodules/EDGS/source/renderers/pgsr.py)。这里替换的是 renderer 和 loss composer；Scene、相机、GaussianModel、EDGS 初始化及 optimizer 仍沿用 EDGS。

#### 输入

- 当前训练相机；
- Gaussian 的 XYZ、SH、opacity、scale、rotation；
- 是否需要 plane depth 和 depth normal；
- 背景颜色。

#### 计算过程

##### C1. 把最小尺度轴解释为表面法线

对 Gaussian (i)：

$$
k_i=\arg\min_{a\in\{x,y,z\}}s_{ia},
\qquad
n_i=R(q_i)e_{k_i}.
$$

系统再根据相机位置翻转法线方向，使其面向相机。实现见 [`pgsr_geometry.py`](../../submodules/EDGS/source/pgsr_geometry.py)。

##### C2. 投影和 alpha compositing

一个 Gaussian 投影为屏幕空间椭圆。其对像素 (u) 的 alpha 可概括为：

$$
a_i(u)=\min\left(
0.99,
\alpha_i\exp\left[-\frac12\Delta u^TQ_i\Delta u\right]
\right).
$$

前向透射率和颜色为：

$$
T_i(u)=\prod_{j<i}(1-a_j(u)),
$$

$$
C(u)=\sum_iT_i(u)a_i(u)c_i(u)+T_{\mathrm{final}}C_{bg}.
$$

##### C3. 合成法线、alpha、平面距离与 plane depth

PGSR 同时合成相机坐标法线、alpha 和平面距离。对像素射线 (r=(x,y,1))，由合成法线 \(\bar n\) 与距离 \(\bar d\) 解出平面 z-depth：

$$
D_{plane}(u)=
\frac{\bar d(u)}
{-\left(\bar n_xx+\bar n_yy+\bar n_z+\epsilon\right)}.
$$

renderer 的主要输出是：

```text
RGB render
visibility / radii
rendered normal
rendered alpha
rendered plane distance
metric plane z-depth
depth-derived normal（按需）
```

#### 输出

可微的 RGB 和几何渲染包，用于 EDGS 光度监督、PGSR 几何约束以及训练完成后的 TSDF 融合。

### 1.5 模块 D：EDGS + PGSR 联合损失

损失实现集中在 [`pgsr_losses.py`](../../submodules/EDGS/source/pgsr_losses.py)，参数来自 [`base.yaml`](../../submodules/EDGS/configs/gs/base.yaml) 和 [`pgsr.yaml`](../../submodules/EDGS/configs/gs/pgsr.yaml)。当前总损失为：

$$
\begin{aligned}
L={}&0.8L_1+0.2(1-\operatorname{SSIM})\\
&+100L_{scale}\\
&+\mathbf 1[t>7000]\left(
0.015L_{normal}+0.03L_{geo}+0.15L_{NCC}
\right).
\end{aligned}
$$

条件在代码中是严格的 `step > 7000`，所以后三个几何项从第 7001 步开始生效。

#### D1. EDGS 光度损失

$$
L_{photo}=(1-\lambda)L_1+\lambda(1-\operatorname{SSIM}),
\qquad \lambda=0.2,
$$

$$
L_1=\operatorname{mean}|I_{render}-I_{GT}|.
$$

#### D2. PGSR scale loss

对当前视角可见的 Gaussian：

$$
L_{scale}=\frac{1}{|\mathcal V|}
\sum_{i\in\mathcal V}\min(s_{ix},s_{iy},s_{iz}).
$$

它促使一个尺度轴变小，使 Gaussian 更接近局部平面 splat。该项从训练开始即启用。

#### D3. 单视图 depth-normal 一致性

先将 plane depth 反投影成相机空间点 (P(u))，用相邻点叉积得到深度法线：

$$
N_D(u)=\operatorname{normalize}
\left[(P_{u+1}-P_{u-1})\times(P_{v-1}-P_{v+1})\right].
$$

再与 rasterizer 合成的 Gaussian 法线 (N_G) 比较：

$$
L_{normal}=\operatorname{mean}_u
w_I(u)\|N_D(u)-N_G(u)\|_1.
$$

其中 (w_I) 是停止梯度的图像边缘权重。RGB 梯度越大，权重越低，以减少深度不连续处的错误约束。

#### D4. 多视图几何重投影

候选邻居最多 8 个，默认要求相机朝向夹角小于 30°，相机中心距离位于 `(0.01, 1.5)`。每次训练随机选一个候选邻居。

计算过程是：

1. 用参考视角 plane depth 把像素 (p) 反投影成世界点 (X)；
2. 把 (X) 投影到邻居相机并双线性采样邻居 depth；
3. 用采样 depth 在邻居相机重建 (X')；
4. 把 (X') 投回参考相机得到 (p')；
5. 计算闭环误差 (e(p)=\|p'-p\|_2)。

默认仅接受 (e<1) 像素的点，并使用停止梯度权重：

$$
w_{geo}=\exp(-e),
\qquad
L_{geo}=\operatorname{mean}(w_{geo}e).
$$

#### D5. 多视图 LNCC

最多采样 102,400 个有效像素。系统由参考平面法线 (n)、距离 (d) 和相对位姿建立平面单应：

$$
H=K_n\left(R-\frac{tn^T}{d}\right)K_r^{-1}.
$$

配置中的 `patch_size=3` 是半径，因此实际比较 (7\times7) 灰度 patch。局部相关损失为：

$$
\rho^2=
\frac{[(x-\bar x)^T(y-\bar y)]^2}
{\|x-\bar x\|_2^2\|y-\bar y\|_2^2+\epsilon},
$$

$$
L_{NCC}=\operatorname{mean}\left[w_{geo}(1-\rho^2)\right].
$$

方差不足、非有限或局部损失大于等于 0.9 的 patch 被丢弃。

### 1.6 模块 E：优化、剪枝和当前训练调度

[`trainer.py`](../../submodules/EDGS/source/trainer.py) 每步从训练视角栈中无放回随机取一个相机，栈空后重新填充；每 1,000 步提升一次 active SH degree，最高到 3。

默认 Adam 学习率如下：

| 参数 | 学习率 |
|---|---:|
| XYZ | `0.00016 -> 0.0000016` 指数调度 |
| SH DC | `0.0025` |
| SH rest | `0.0025 / 20` |
| opacity | `0.025` |
| scale | `0.005` |
| rotation | `0.001` |

当前 PaintMesh Stage 1 默认设置 `NO_DENSIFY=true`。因此它不 clone/split，而是在第 15,000 步前删除实际 opacity 小于 `0.005` 的 Gaussian。同期每 10 步还会降低一次 opacity 参数。若显式设置 `NO_DENSIFY=false`，则从第 500 步后每 100 步按视空间位置梯度执行 clone、split 与 prune，直到第 15,000 步。

另一个需要按代码理解的细节是 `train.max_lr=true`：XYZ scheduler 使用 `max(step, 8000)`，因此前 8,000 步都采用调度器第 8,000 步对应的学习率。

#### 输出

训练结束时得到只含几何和外观字段的 EDGS 3DGS PLY。PLY 中的 `opacity` 与 `scale_*` 保存的是未激活参数，渲染时分别经过 sigmoid 与 exp。

### 1.7 模块 F：由 3DGS 提取 TSDF mesh

mesh 入口为 [`submodules/EDGS/render.py`](../../submodules/EDGS/render.py)。

#### 输入

- 训练完成的 Gaussian PLY；
- 原始真实训练相机；
- PGSR renderer；
- `max_depth=5.0`；
- `voxel_size=0.002`；
- 默认保留 1 个最大连通分量。

#### 计算过程

##### F1. 逐训练视角渲染

系统从训练完成的 3DGS 渲染 RGB、metric plane depth 和 normal。TSDF 使用的是 3DGS 渲染 RGB，不是 GT RGB。`renders_depth/*.png` 只是逐图归一化后的彩色可视化，不是用于融合的米制深度文件；真正的 plane depth 在渲染循环中以内存 Tensor 直接送入 TSDF。

##### F2. 深度过滤

基础有效条件为：

$$
D\text{ finite},\qquad D>0,\qquad D\le5.0.
$$

如相机带 alpha mask，还要求 alpha mask 不低于 0.5。默认 `USE_DEPTH_FILTER=false`；启用后才额外过滤视线与法线夹角大于 80° 的像素。

##### F3. Open3D TSDF 融合

系统创建：

```python
ScalableTSDFVolume(
    voxel_length=0.002,
    sdf_trunc=0.008,
    color_type=RGB8,
)
```

截断距离为四倍 voxel size。PGSR depth 按 metric z-depth 使用，所以 `depth_scale=1.0`。概念上，每个视角对体素 (X) 产生：

$$
\operatorname{tsdf}_i(X)=
\operatorname{clip}\left(
\frac{D_i(\pi_i(X))-z_i(X)}{\mu},-1,1
\right),
\qquad \mu=4v,
$$

再融合多视角观测。

##### F4. mesh 提取和后处理

Open3D 从 TSDF 的零交叉面提取三角网格，先保存 `tsdf_fusion.ply`，随后：

1. 聚类三角形连通分量；
2. 默认保留最大的 1 个分量；
3. 忽略小于 50 个三角形的小分量；
4. 删除未引用顶点和退化三角形；
5. 重新计算顶点法线；
6. 保存 `tsdf_fusion_post.ply`。

#### 输出

基础 3DGS 与从它派生的 TSDF mesh。二者共享场景几何，但不是同一组参数，也没有联合优化关系。

---

## 2. 3DGS 和 mesh 如何完成二维到三维语义分割？

### 2.1 总体数据流

```text
多视角 RGB
   │
   ▼
CropFormer：逐视角局部实例 mask
   │
   ▼
Gaussian 中心投影 + 深度筛选 + 跨视角关联
   │
   ▼
跨视角一致的 scene-local 2D instance ID
   │
   ▼
冻结几何和外观，蒸馏每个 Gaussian 的 16D embedding
   │
   ├──────────────> 16D feature render -> classifier -> 2D instance render
   │
   ▼
语义 3DGS
   │
   ▼
空间 / 尺度 / opacity / 法线加权插值
   │
   ▼
mesh vertex label -> triangle consensus -> 语义 mesh
```

关键区别是：3DGS 语义由二维 mask 监督学习得到；mesh 语义不是再次直接反投影二维 mask，而是从已经学好的 Gaussian embedding 提升得到。

### 2.2 模块 A：CropFormer 逐视角实例分割

实现位于 [`raw_mask_sam.py`](../../submodules/Inpaint360GS/seg/raw_mask_sam.py)。虽然命令参数沿用 `--method hqsam`，当前该分支实际加载 Detectron2 CropFormer Hornet 配置和 checkpoint。

#### 输入

- 对应分辨率的场景 RGB，例如 `images_8/`；
- `CropFormer_hornet_3x_03823a.pth`；
- 默认置信度阈值 `0.5`。

#### 计算过程

模型返回 instance masks (M_k) 和 scores (s_k)。只保留：

$$
s_k\ge0.5.
$$

保留项按 score 从低到高排序，依次写入当前图片的 ID `1..N`。因此重叠区域中，后写入的高分实例覆盖低分实例。

#### 输出

```text
raw_hqsam/<view>.png
raw_hqsam_color/<view>.png
```

原始 mask 是二维 `uint16` PNG：背景为 0，前景 ID 仅在本张图片内有效。

### 2.3 模块 B：利用 Gaussian 建立跨视角实例对应

实现位于 [`mask_associate.py`](../../submodules/Inpaint360GS/seg/mask_associate.py)。

#### 输入

- 所有 `raw_hqsam/*.png`；
- EDGS Gaussian 中心 XYZ；
- COLMAP/EDGS 相机；
- 默认 `16 x 16` patch 网格；
- 关联阈值 `0.1`。

#### 计算过程

##### B1. 局部实例二值化

一张图内 ID 为 (m) 的实例变成：

$$
B_m(p)=[I(p)=m],\qquad m=1,\ldots,M.
$$

背景 0 不生成前景 mask。

##### B2. 投影 Gaussian 中心

对 Gaussian 中心 (x_i)：

$$
p_i^h=[x_i,1]P,
\qquad
p_i^{ndc}=p_{i,xyz}^h/(p_{i,w}^h+\epsilon).
$$

NDC 转到像素并四舍五入，只保留图像范围内且投影深度为正的中心。这里使用的是 Gaussian 中心投影，不是完整二维 Gaussian footprint，也没有用 alpha rasterization 做精确遮挡。

##### B3. patch 内前景深度筛选

图像分成 (16\times16) 个 patch。对落入“当前实例 mask 与 patch 交集”的 Gaussian：

1. 候选不少于 2 个时，对投影深度做 `KMeans(n_clusters=2)`；
2. 选择均值更近的 cluster；
3. 在该 cluster 中再取最近的 30%，至少 1 个；
4. 候选不足 2 个时，保留最近的至少 1 个。

这一步近似抑制被遮挡的背景 Gaussian，得到局部实例对应的 Gaussian 索引集合 (S_m)。

##### B4. 匹配全局实例数据库

全局实例 (j) 保存已关联的 Gaussian 索引集合 (G_j)。代码中名为 `IOU_highlight` 的 score 实际不是标准 IoU，而是：

$$
score_{jm}=
\frac{|G_j\cap S_m|}
{|S_m|+|G_j\cap S_m|+\epsilon}.
$$

选择最高分的 (j^*)。若最高分小于 `0.1`，创建新全局实例；否则合并到 (G_{j^*})。第一张图直接初始化数据库，后续图像按排序顺序关联，因此最终 ID 与视图顺序有关。

已有实例的更新会过滤已分配 Gaussian；但新实例分支直接加入集合，所以不应把当前实现描述为数学上严格互斥的 Gaussian 分配。

##### B5. 写回全局二维 ID

内部全局索引 `0..K-1` 写回时加 1，背景仍为 0。最终类别数：

$$
C=K+1.
$$

#### 输出

```text
associated_hqsam/<view>.png
associated_hqsam_color/<view>.png
associated_hqsam/scene.json
```

`scene.json` 记录包含背景的 `num_classes`、mask 路径和 patch 配置。

### 2.4 模块 C：把二维实例监督蒸馏到 3D Gaussians

实现位于 [`distillation.py`](../../submodules/Inpaint360GS/seg/distillation.py)、Inpaint360GS [`gaussian_model.py`](../../submodules/Inpaint360GS/scene/gaussian_model.py) 和 [`gaussian_renderer`](../../submodules/Inpaint360GS/gaussian_renderer/__init__.py)。

#### 输入

- 基础 EDGS Gaussian PLY；
- 跨视角一致的 `associated_hqsam` masks；
- `scene.json`；
- 相机集合；
- 默认 2,000 次蒸馏配置。

#### 计算过程

##### C1. 给每个 Gaussian 增加 16 维实例 embedding

$$
f_i\in\mathbb R^{16}.
$$

维度在 Python 模型和 CUDA rasterizer 中均固定为 16。普通 EDGS PLY 没有 `obj_dc_*`，加载时先补零，保存后则具有 `obj_dc_0..15`。

##### C2. 冻结几何和外观

蒸馏 optimizer 只包含 `_objects_dc`。XYZ、RGB SH、opacity、scale 和 rotation 全部被冻结，因此语义蒸馏不会改变基础几何或外观。另有一个 `1x1 Conv2d` classifier：

$$
z(p)=WF(p)+b,
\qquad W\in\mathbb R^{C\times16}.
$$

classifier 使用 Adam，学习率 `5e-4`。

##### C3. 可微 alpha 合成 16D feature map

使用与 RGB 相同的前向 alpha 权重：

$$
F(p)=\sum_iT_i(p)a_i(p)f_i.
$$

object feature 没有 RGB 背景项，未被 Gaussian 覆盖的像素 feature 为零。

##### C4. 二维交叉熵监督

随机选择一个训练视角，得到 logits 后计算：

$$
L_{2D}=\frac{1}{HW\log C}
\sum_p\operatorname{CE}(WF(p)+b,y(p)).
$$

背景像素也参与监督。若 mask 与加载后的相机尺寸不同，使用 nearest-neighbor resize，避免对离散 ID 做连续插值。

##### C5. 当前真正生效的三维正则

默认每 50 步最多从 200,000 个 Gaussian 中随机选 1,000 个 query，并查询 5 个 XYZ 近邻。函数内部同时计算 KL 与 cosine 项，但当前调用写成：

```python
_, loss_obj_3d_sim = loss_cls_3d_cosin(...)
loss = loss_obj_2d + loss_obj_3d_sim
```

所以 KL 返回值被丢弃，真正加入总损失的只有：

$$
L_{cos}=0.0005\left[
1-\operatorname{mean}_{i,j}\cos(f_i,f_j)
\right].
$$

当前实际目标为：

$$
L=\begin{cases}
L_{2D}+L_{cos},& t\bmod50=0,\\
L_{2D},& \text{其他步骤}.
\end{cases}
$$

配置中的 `reg3d_lambda_val` 只作用于被丢弃的 KL 返回值，因此目前不会影响反向传播。

#### 输出

```text
semantic_3dgs/point_cloud/iteration_2000/
├── point_cloud.ply
└── classifier.pth
```

语义 3DGS 的完整语义定义是 `point_cloud.ply + classifier.pth`。单独读取 `obj_dc_*` 不能得到正确 ID；必须使用配套 classifier。

### 2.5 模块 D：语义 3DGS 的二维推理

#### 输入

- 含 16D embedding 的 Gaussian PLY；
- classifier；
- 目标相机。

#### 计算过程

先 alpha-composite 得到每像素 16D feature (F(p))，再计算：

$$
\hat y(p)=\arg\max_c[WF(p)+b]_c.
$$

该渲染路径不使用 mesh lifting 的 confidence、margin 或 unknown 阈值，每个像素都会经 argmax 分到一个有效类别。

#### 输出

```text
semantic_3dgs/<train|test>/ours_2000/
├── renders/
├── objects_pred/
├── objects_pred_color/
└── depth/
```

### 2.6 模块 E：把 Gaussian 语义提升到 mesh

实现位于 [`lift_gaussian_semantics_to_mesh.py`](../../submodules/Inpaint360GS/tools/lift_gaussian_semantics_to_mesh.py)。

#### 输入

- 语义 Gaussian PLY；
- 配套 classifier；
- `scene.json`；
- PGSR `tsdf_fusion_post.ply`；
- 默认 `K=8`、`opacity_min=0.01`、`support_sigma=3.0`、`normal_power=2.0`。

#### 计算过程

##### E1. Gaussian 自身分类

对 embedding (f_i)：

$$
p_i=\operatorname{softmax}(Wf_i+b).
$$

最高概率 (p_{i,1}) 必须至少为 `0.10`，且与第二名之差至少为 `0.02`；否则为 unknown `65535`。实际 opacity 小于 `0.01` 的 Gaussian 也直接变为 unknown。

##### E2. 构建可用 Gaussian 的 cKDTree

仅用 opacity 不低于 `0.01` 的 Gaussian 建立空间索引。每个 mesh vertex 查询最近的 8 个候选。

##### E3. 插值 16D embedding 到 mesh vertex

Gaussian 的支撑尺度取最大激活尺度：

$$
r_i=\max_a\exp(s_{ia}^{log}).
$$

Gaussian 法线取最小尺度轴经四元数旋转后的方向。对 mesh 顶点 (v) 和 Gaussian (i)：

$$
\delta_{vi}=\frac{\|x_v-x_i\|_2}{\max(r_i,10^{-8})}.
$$

只接受 δᵥᵢ ≤ 3.0，并使用：

$$
w_{vi}=\alpha_i
\exp\left(-\frac12\delta_{vi}^2\right)
|n_i^Tn_v|^2.
$$

绝对点积使平面法线的正负方向等价。顶点 embedding 为：

$$
f_v=\frac{\sum_iw_{vi}f_i}{\sum_iw_{vi}}.
$$

随后用同一个 classifier 和相同 confidence/margin 阈值得到 vertex ID。没有有效支撑、坐标或法线非法、总权重过小的顶点均为 unknown。

这里插值的是连续的 16D embedding，不是对相邻 Gaussian 的离散 ID 做多数投票。

##### E4. 三角面 consensus

默认 `face_min_agreement=2`。三角形至少两个已知顶点具有相同标签，face 才采用该标签；否则为 unknown。face confidence 为同意该标签的顶点 confidence 总和除以固定分母 3：

$$
confidence_f=
\frac{\sum_{v\in f,\ label_v=label_f}confidence_v}{3}.
$$

#### 输出

```text
semantic_mesh/
├── gaussian_instance_id.npy
├── gaussian_confidence.npy
├── vertex_instance_id.npy
├── vertex_confidence.npy
├── face_instance_id.npy
├── face_confidence.npy
├── palette.json
├── semantic_mesh.ply          # 可选的着色副本
└── semantic_manifest.json
```

`semantic_mesh.ply` 的 instance ID 是 vertex 属性；权威 face 结果仍是 `face_instance_id.npy`。语义提升不会改变 mesh 的几何位置或三角拓扑。

### 2.7 两条三维语义路径的本质区别

| 结果 | 二维到三维的机制 | 权威表示 |
|---|---|---|
| 语义 3DGS | 2D 跨视角 ID 监督可微渲染，学习每个 Gaussian 的 16D embedding | `point_cloud.ply` + `classifier.pth` |
| 语义 mesh | 从已训练的 Gaussian embedding 按空间、尺度、opacity 和法线插值，再做 face consensus | 六个 `.npy` sidecar + manifest |
| 语义二维渲染 | 渲染 16D feature，经 classifier 后 argmax | `objects_pred/*.png` |
| 彩色语义 mesh | 将 vertex ID 映射为颜色的可视化副本 | `semantic_mesh.ply` |

---

## 3. 3DGS 和 mesh 如何分别完成 inpaint？

### 3.1 总体数据流

```text
语义 3DGS
   │
   ▼
classifier threshold + 3D convex hull
   │
   ▼
删除目标 Gaussian，得到 removed 3DGS
   │
   ├──PGSR + TSDF──> removed mesh
   │
   ▼
full / removed 3DGS 在相同的 30 个虚拟相机下渲染
   │
   ▼
SAM-Track 得到 30 张目标 mask
   │
   ▼
LaMa 补全 removed RGB 与 metric depth
   │
   ▼
mask 内 RGB-D 反投影 -> 30 个 support PLY
   │
   ▼
一个 seed PLY 初始化新 Gaussians
   │
   ▼
30 个 completed RGB 视角监督 3DGS finetune
   │
   ▼
空间门控提交 -> inpainted 3DGS
   │
   ├──PGSR + TSDF──> 新建 inpainted mesh
   │
   └──Gaussian semantics relift──> inpainted semantic mesh
```

### 3.2 前置模块 A：从语义 3DGS 中删除目标

实现位于 [`edit_object_removal.py`](../../submodules/Inpaint360GS/edit_object_removal.py) 和 [`GaussianModel.removal_setup`](../../submodules/Inpaint360GS/scene/gaussian_model.py)。

#### 输入

- 语义 Gaussian PLY；
- classifier；
- `target_ids`；
- 可选 `surrounding_ids`；
- 默认 `removal_thresh=0.7`。

#### 计算过程

每个 Gaussian 的 16D embedding 先经 classifier 和 softmax。对目标 ID (c)：

$$
M_c(i)=[p(c\mid f_i)>0.7].
$$

仅靠概率阈值可能漏掉实例内部 Gaussian，因此系统：

1. 对命中点逐轴做 IQR 异常值过滤；
2. 用 Delaunay 构建三维凸包；
3. 找出位于凸包内的所有 Gaussian；
4. 将概率 mask 与凸包 mask 取并集。

如果有效点不足以张成三维体积，则退回概率 mask。目标半径取过滤后点到中心距离的第 80 百分位，用于后续选择虚拟相机距离。

`removal_setup` 把模型拆成剩余场景和每个选中实例的子模型。若凸包重叠，按插入顺序把 Gaussian 归给第一个对象；目标 ID 先于 surrounding ID。所有选中项先被移走，然后再把 `surrounding_ids` 对应子模型追加回来，所以最终只永久删除 `target_ids`。

#### 输出

```text
removed_3dgs/point_cloud/iteration_<N>/point_cloud.ply
```

该 PLY 仍含 RGB、几何和 16D embedding。随后从 removed 3DGS 重新做 PGSR + TSDF，得到 removed mesh；旧 mesh 不参与删除运算。

### 3.3 模块 B：生成 30 个严格一致的虚拟相机

实现位于 [`virtual_pose.py`](../../submodules/Inpaint360GS/tools/virtual_pose.py)、[`pose_utils.py`](../../submodules/Inpaint360GS/utils/pose_utils.py) 和 [`virtual_camera_manifest.py`](../../submodules/Inpaint360GS/utils/virtual_camera_manifest.py)。

#### 输入

- full semantic 3DGS；
- removed 3DGS；
- 真实相机分布；
- 上一步计算的目标半径。

#### 计算过程

系统用相机位姿 PCA 估计场景主轴和归一化尺度，计算观察焦点，并依据目标半径与相机 FoV 估算能覆盖目标的相机距离。随后在目标周围生成 30 个匀速圆形/椭圆轨迹相机。

同一组相机分别渲染：

- full 3DGS 的 RGB、语义、depth；
- removed 3DGS 的 RGB、语义、depth。

removed RGB 被按 `00000.png..00029.png` 打包为 tracker `images.zip`。每个相机的完整精度 `R`、`T`、FoV、宽高、near/far、scene translation/scale 写入 `virtual_cameras.json`。后续所有模块复用该 manifest，而不是用四舍五入后的半径重新生成轨迹，从而避免 mask、RGB、depth 和反投影之间的像素偏移。

#### 输出

```text
tracker/images.zip
tracker/virtual_cameras.json
work_model/virtual/ours_<N>/...                  # full
work_model/virtual/ours_object_removal/...       # removed
```

### 3.4 模块 C：交互式 SAM-Track 得到跨虚拟视角 mask

#### 输入

- removed-scene `images.zip`；
- 用户在第一帧提供的点、框或涂画提示；
- SAM、AOT/SegTracker checkpoint。

#### 计算过程

SAM 在首帧产生目标 mask，AOT/SegTracker 把对象状态传播到其余虚拟帧；用户可以在界面中检查或修正结果。PaintMesh 对会话进行严格绑定：

- 必须正好输出 `00000.png..00029.png`；
- 每张 mask 必须与对应 tracker RGB 同尺寸；
- mask 必须属于当前 tracking session；
- archive、camera manifest 与 mask 的 hash 会写入会话记录。

#### 输出

```text
tracker/results/images/images_masks/00000.png ... 00029.png
tracker/tracking_session.json
```

这些 mask 描述“需要补全的二维洞区域”。它们不是新的场景语义训练标签。

### 3.5 模块 D：准备 LaMa 输入

实现位于 [`prepare_paintmesh_lama_data.py`](../../submodules/Inpaint360GS/tools/prepare_paintmesh_lama_data.py)。

#### 输入

每个虚拟帧包含：

- removed RGB；
- removed metric depth；
- full-scene metric depth，作为深度数值范围参考；
- tracker mask。

#### 计算过程

所有非零 tracker ID 都视为 hole。mask 先做 8 邻域连通域分析：

1. 默认移除面积小于 50 的连通分量；
2. 若真实目标很小而所有分量都被移除，保留最大分量；
3. 用半径 10 对应的 `21x21` 椭圆核膨胀一次；
4. 拒绝空 mask 和覆盖整图的 mask。

系统为 RGB 和 depth 写入完全相同的二值 mask，同时验证四类输入形状一致、depth 有限、非负且具有非零数值范围。

#### 输出

```text
lama/input/color/<frame>.png
lama/input/color/<frame>_mask.png
lama/input/depth/<frame>.npy
lama/input/depth/<frame>_mask.png
lama/input/depth/depth_original/<frame>.npy
manifests/lama_input_manifest.json
```

### 3.6 模块 E：LaMa RGB 与 depth completion

实现位于 [`predict_color.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_color.py)、[`predict_depth.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_depth.py) 及 LaMa [`evaluation/data.py`](../../submodules/Inpaint360GS/LaMa/saicinpainting/evaluation/data.py)。

#### E1. RGB completion

LaMa 对 removed RGB 和 mask 预测补全图，但最终保存时明确只替换 mask 内像素：

$$
I_{out}(p)=\begin{cases}
I_{LaMa}(p),&M(p)=1,\\
I_{removed}(p),&M(p)=0.
\end{cases}
$$

所以 mask 外 RGB 在字节级保持原样。

#### E2. depth completion

对 removed depth (D_r)，使用对应 full-scene depth 的最小值 (d_{min}) 和最大值 (d_{max}) 做仿射归一化：

$$
\tilde D_r=\frac{D_r-d_{min}}{d_{max}-d_{min}}.
$$

LaMa 将该单通道深度复制成三通道形式进行补全。预测的第一通道再反归一化：

$$
D_{pred}=\tilde D_{pred}(d_{max}-d_{min})+d_{min}.
$$

最终也只替换 mask 内：

$$
D_{out}(p)=\begin{cases}
D_{pred}(p),&M(p)=1,\\
D_r(p),&M(p)=0.
\end{cases}
$$

这里的 full-scene depth 只提供归一化范围，不会被直接复制进洞区域。

#### 输出

```text
lama/output/color/00000.png ... 00029.png
lama/output/depth/00000.npy  ... 00029.npy
manifests/lama_completion_manifest.json
```

验证器要求 mask 外 RGB 和 depth 与 removed 输入完全一致。

### 3.7 模块 F：completed RGB-D 反投影成支持点云

实现位于 [`edit_object_removal_plyfusion.py`](../../submodules/Inpaint360GS/edit_object_removal_plyfusion.py) 和 [`point_utils.py`](../../submodules/Inpaint360GS/utils/point_utils.py)。

#### 输入

- 30 张 completed RGB；
- 30 张 completed metric depth；
- 同一批二值 mask；
- `virtual_cameras.json`。

#### 计算过程

由 FoV 和图像尺寸得到：

$$
f_x=\frac{W/2}{\tan(FoV_x/2)},
\qquad
f_y=\frac{H/2}{\tan(FoV_y/2)},
\qquad
c_x=W/2,\ c_y=H/2.
$$

每个像素 ((u,v)) 与 z-depth (D(u,v)) 在相机坐标下反投影为：

$$
x_c=(u-c_x)D/f_x,
\qquad
y_c=(v-c_y)D/f_y,
\qquad
z_c=D.
$$

再用 camera-to-world 矩阵转到世界坐标。系统只保存 mask 内的点及其 completed RGB 颜色。

#### 输出

```text
fused/mask/00000.ply ... 00029.ply
manifests/fusion_manifest.json
```

可选的 `fused/hole/` PLY 是 removed RGB-D 的诊断输出，不是最终 mesh。

一个重要的当前实现细节是：虽然生成 30 个支持 PLY，Stage 5 只选择 `FUSION_SEED_FRAME` 指定的一张，默认 `00004.ply`，用来初始化新 Gaussian；30 张 completed RGB 则全部用于后续多视角优化。

### 3.8 模块 G：从 seed 支持点初始化新的 Gaussians

实现位于 [`GaussianModel.inpaint_setup`](../../submodules/Inpaint360GS/scene/gaussian_model.py) 和 [`compose_utils.py`](../../submodules/Inpaint360GS/utils/compose_utils.py)。

#### 输入

- source semantic 3DGS；
- 目标 ID 与 classifier；
- 默认 `00004.ply` 支持点云；
- `opacity_init=0.1`。

#### 计算过程

系统再次用 classifier threshold 与凸包找出目标区域，保留非目标 Gaussians。seed PLY 先经 Open3D statistical outlier removal：`nb_neighbors=5, std_ratio=4.0`。每个剩余支持点初始化一个新 Gaussian：

| 属性 | 初始化方式 |
|---|---|
| XYZ | RGB-D 支持点的世界坐标 |
| SH DC | completed RGB 转 SH |
| 高阶 SH | 0 |
| opacity | `logit(0.1)` |
| scale | 支持点最近邻距离的平方根，再取 log |
| rotation | 单位四元数 |
| 16D embedding | 最近 5 个保留 Gaussian embedding 的均值 |

最终模型按以下顺序拼接：

```text
[保留场景 Gaussians, 新支持 Gaussians]
```

新 embedding 只是三维近邻继承，并没有根据 LaMa 生成的 RGB 重新做语义识别。

#### 输出

一个可优化的临时 GaussianModel，既含保留场景，也含补全区域的新 splats。

### 3.9 模块 H：30 视角 3DGS finetune

实现位于 [`edit_object_inpaint.py`](../../submodules/Inpaint360GS/edit_object_inpaint.py)。这个阶段使用 Inpaint360GS 的 Gaussian renderer；PGSR renderer 在后续重建 mesh 时才再次使用。

#### 输入

- 30 个来自 camera manifest 的虚拟相机；
- 每个相机的 completed RGB，作为 `original_image`；
- 每个相机的二值 hole mask；
- 上一步的临时 Gaussians；
- 默认 5,000 次迭代。

#### 计算过程

每步随机选择一个虚拟相机并渲染 RGB。默认配置：

```text
lambda_dssim = 0.8
lambda_lpips = 0.0005
finetune_iteration = 5000
```

损失为：

$$
L=(1-\lambda_{dssim})L_{1,known}
+\lambda_{dssim}(1-SSIM_{full})
+\lambda_{lpips}L_{LPIPS,bbox}.
$$

其中：

- (L_{1,known}) 只在 mask 外计算，用来维持已知区域；
- (SSIM_{full}) 在整张 completed RGB 上计算，所以包含洞区域；
- LPIPS 在 hole 的 bounding box 内切成 `2x2` patch 后计算，单个 patch 小于 `32x32` 时跳过。

这里没有显式 depth loss。completed depth 只通过 RGB-D 反投影提供新 Gaussian 的三维初值；训练时没有逐像素深度监督。这里也没有 semantic classification loss，所以新 Gaussian 的 16D embedding 通常保持其 KNN 初始化。

##### H1. densify 与 prune

当 `500 < t < 5000` 且 `t` 是 100 的倍数时，系统根据可见 Gaussian 的屏幕空间位置梯度：

- clone 尺度较小的高梯度新 Gaussian；
- split 尺度较大的高梯度新 Gaussian；
- prune opacity 低于 `0.005` 或异常大的新 Gaussian。

保留场景的前缀被排除于 densification，并在当前调用的 size threshold 分支中受到 prune 保护。

##### H2. 优化后的空间门控提交

系统不会无条件保存所有 finetune 结果。它以 seed 相机和 seed PLY 构造 gate：

1. 将 seed mask 膨胀到面积至少约为原来的 110%；
2. 把优化后 Gaussian 中心投影到 seed 相机；
3. 只保留落入膨胀 mask 的候选；
4. 对 seed 支持点建 cKDTree；
5. 还要求候选到支持点的最近距离小于 `3 x 支持点各轴标准差的均值`。

随后先恢复优化前的保留场景快照，再只在该 gate 内写回训练后的属性。严格地说，optimizer 参数包含拼接后的整个 tensor，因此训练循环中不应简单理解为“只有新点能产生梯度”；最终结果的局部性主要由“背景恢复 + 空间 gate 提交”保证。gate 内的原有点也可能保留其优化后属性。

#### 输出

```text
work_model/point_cloud_object_inpaint_virtual/
└── iteration_5000/point_cloud.ply
```

该工作 PLY 含 RGB/几何和 `obj_dc_0..15`。

### 3.10 模块 I：发布 EDGS 可加载的 inpainted 3DGS

实现位于 [`publish_inpainted_edgs_model.py`](../../submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py)。

#### 输入

- 工作区 inpainted PLY；
- classifier；
- EDGS `config.yaml` 和 `cfg_args`；
- removal、tracker、camera、LaMa、fusion 等 manifests。

#### 计算过程

发布器先验证所有上游 artifact ID、hash、参数、目标 ID 和迭代号。主 Gaussian PLY 采用原子普通文件复制：

1. 分块读取源 PLY并同时计算 SHA-256；
2. 写入目标目录中的临时文件；
3. `fsync` 后校验字节数与 hash；
4. 用 `os.replace` 原子提交；
5. 验证目标不是 symlink，且与源文件不是同一 inode。

因此：

```text
inpainted_3dgs/point_cloud/iteration_5000/point_cloud.ply
```

是独立的真实文件，而不是指向工作区 PLY 的符号链接。`config.yaml`、`cfg_args` 和 `classifier.pth` 等附属文件仍可使用受控的相对 symlink。

#### 输出

```text
inpainted_3dgs/
├── config.yaml
├── cfg_args
├── model_manifest.json
└── point_cloud/iteration_5000/
    ├── point_cloud.ply
    └── classifier.pth
```

### 3.11 模块 J：inpainted mesh 的真正生成方式

#### 输入

- 已发布的 inpainted 3DGS；
- 原始真实训练相机，而不是 30 个虚拟相机；
- PGSR renderer；
- TSDF 参数。

#### 计算过程

系统调用与基础 mesh 相同的 EDGS [`render.py`](../../submodules/EDGS/render.py)：

1. 在真实训练视角渲染 inpainted 3DGS；
2. 得到 RGB、PGSR metric plane depth 和 normal；
3. 过滤无效或过远 depth；
4. 将所有训练视角 RGB-D 融入新的 TSDF volume；
5. 从零交叉面提取全新的 triangle mesh；
6. 做连通分量和退化三角形后处理。

因此，LaMa 的 30 个支持点云只是 3DGS 初始化依据，不会直接拼接成最终 mesh。最终 mesh 的几何依据是“补全后的 3DGS 在真实训练相机下的 PGSR plane depth”。

#### 输出

```text
inpainted_3dgs/mesh/ours_5000/
├── tsdf_fusion.ply
├── tsdf_fusion_post.ply
└── mesh_manifest.json
```

### 3.12 模块 K：给新 mesh 重新附加语义

TSDF 新 mesh 不携带 Gaussian embedding，因此系统再次运行第 2.6 节的 semantic lifting：

```text
inpainted Gaussian embedding
  -> 空间 / 尺度 / opacity / 法线加权插值
  -> vertex embedding
  -> classifier
  -> vertex ID
  -> face consensus
```

这一步只给新 mesh 附加语义，不修改几何。由于新补全 Gaussian 没有 semantic loss，其 embedding 来自保留 Gaussian 的 5-NN 均值，所以补全区域的实例 ID 表示上下文继承结果，而不是对 LaMa 新内容进行独立语义识别。

#### 输出

```text
inpainted_mesh/
├── geometry.ply                 # 指向最终 TSDF post mesh 的相对链接
├── gaussian_instance_id.npy
├── gaussian_confidence.npy
├── vertex_instance_id.npy
├── vertex_confidence.npy
├── face_instance_id.npy
├── face_confidence.npy
├── palette.json
├── semantic_mesh.ply            # 启用彩色输出时
└── semantic_manifest.json
```

### 3.13 模块 L：最终一致性提交

[`finalize_inpaint_result.py`](../../submodules/Inpaint360GS/tools/finalize_inpaint_result.py) 检查：

- published 3DGS 主 PLY 是有效普通文件并与源内容一致；
- Gaussian、mesh vertex、triangle 数量有效；
- semantic sidecar 长度与对应几何元素数量一致；
- `geometry.ply` 精确指向本次 PGSR post mesh；
- removal、workspace、tracking、camera、LaMa、fusion、model、mesh 和 semantic artifact IDs 形成完整的一致链；
- 目标实例在最终 Gaussian/vertex/face 结果中的残留数量可被统计。

最终生成：

```text
inpaint/default/inpaint_manifest.json
```

manifest 是整条 remove -> virtual views -> tracker -> LaMa -> RGB-D -> 3DGS -> PGSR/TSDF -> semantic relift 的提交标记。

---

## 4. 三个问题的最短答案

| 问题 | 3DGS | mesh |
|---|---|---|
| 如何训练？ | EDGS RoMa 稠密对应初始化；PGSR plane-aware rasterization；EDGS 光度损失加 scale、normal、多视图几何和 LNCC | 不训练；从已训练 3DGS 的真实训练视角 RGB/plane-depth 做 TSDF 融合 |
| 如何从 2D 得到语义？ | CropFormer 局部 mask -> Gaussian 投影关联全局 ID -> 可微渲染蒸馏 16D embedding + classifier | 从语义 Gaussian 的 embedding 按距离、尺度、opacity、法线加权插值到 vertex，再由 triangle consensus 得到 face ID |
| 如何 inpaint？ | 删除目标 Gaussian；LaMa 补全虚拟视角 RGB/depth；一个 seed RGB-D PLY 初始化新 splats；30 视角 RGB finetune；空间门控提交 | 不直接补洞；从 inpainted 3DGS 重新 PGSR 渲染真实视角，再从头 TSDF 建 mesh，最后重新 lift 语义 |

---

## 5. 设计边界与当前实现限制

1. 语义标签是 scene-local instance ID，不能跨场景直接比较。
2. `mask_associate` 使用 Gaussian 中心和启发式深度筛选，并非精确 alpha/visibility association。
3. 蒸馏函数计算了 KL，但当前训练调用只使用 cosine 正则。
4. LaMa depth 是把深度当图像做补全，几何一致性主要依赖多虚拟视角、RGB-D 初值和后续 3DGS 优化，而不是显式多视图 depth loss。
5. 当前只用一个 `FUSION_SEED_FRAME` PLY 初始化 splats，未把 30 个支持 PLY 联合配准或融合成一个初始化点云。
6. 3DGS finetune 没有 depth loss，也没有 semantic loss。
7. 补全 mesh 的细节上限受 PGSR plane depth、真实训练相机覆盖、TSDF voxel size 和连通分量过滤共同限制。
8. `semantic_mesh.ply` 是便于查看的 vertex 着色副本；需要精确 face 语义时应读取 `.npy` sidecar 和 manifest。

## 6. 主要源码索引

| 环节 | 源码 |
|---|---|
| EDGS 入口与训练 | [`train.py`](../../submodules/EDGS/train.py)、[`trainer.py`](../../submodules/EDGS/source/trainer.py) |
| RoMa 初始化 | [`corr_init.py`](../../submodules/EDGS/source/corr_init.py) |
| PGSR renderer / losses | [`pgsr.py`](../../submodules/EDGS/source/renderers/pgsr.py)、[`pgsr_losses.py`](../../submodules/EDGS/source/pgsr_losses.py)、[`pgsr_geometry.py`](../../submodules/EDGS/source/pgsr_geometry.py) |
| TSDF mesh | [`EDGS/render.py`](../../submodules/EDGS/render.py) |
| 逐视角分割 | [`raw_mask_sam.py`](../../submodules/Inpaint360GS/seg/raw_mask_sam.py) |
| 跨视角关联 | [`mask_associate.py`](../../submodules/Inpaint360GS/seg/mask_associate.py) |
| Gaussian 语义蒸馏 | [`distillation.py`](../../submodules/Inpaint360GS/seg/distillation.py) |
| Gaussian -> mesh 语义提升 | [`lift_gaussian_semantics_to_mesh.py`](../../submodules/Inpaint360GS/tools/lift_gaussian_semantics_to_mesh.py) |
| Gaussian remove | [`edit_object_removal.py`](../../submodules/Inpaint360GS/edit_object_removal.py) |
| 虚拟相机 | [`virtual_pose.py`](../../submodules/Inpaint360GS/tools/virtual_pose.py) |
| LaMa 输入与验证 | [`prepare_paintmesh_lama_data.py`](../../submodules/Inpaint360GS/tools/prepare_paintmesh_lama_data.py) |
| RGB/depth completion | [`predict_color.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_color.py)、[`predict_depth.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_depth.py) |
| RGB-D 反投影 | [`edit_object_removal_plyfusion.py`](../../submodules/Inpaint360GS/edit_object_removal_plyfusion.py) |
| 3DGS inpaint | [`edit_object_inpaint.py`](../../submodules/Inpaint360GS/edit_object_inpaint.py)、[`compose_utils.py`](../../submodules/Inpaint360GS/utils/compose_utils.py) |
| EDGS 模型发布 | [`publish_inpainted_edgs_model.py`](../../submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py) |
| 最终一致性验证 | [`finalize_inpaint_result.py`](../../submodules/Inpaint360GS/tools/finalize_inpaint_result.py) |
