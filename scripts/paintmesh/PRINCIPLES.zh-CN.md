# PaintMesh / Inpaint360GS 原理说明

本文解释当前仓库中 `scripts/paintmesh` 三条流水线背后的实际计算过程：

1. EDGS-PGSR 3DGS 与基础 mesh 如何构建；
2. 二维实例 mask 如何变成语义 3DGS 和语义 mesh；
3. 3DGS 与 mesh 如何完成目标移除和场景补全。

本文以当前代码为准，而不是对上游论文流程的泛化描述。运行命令、断点续跑和目录参数请参阅 [README.zh-CN.md](README.zh-CN.md)。明确标记“待实现”的段落描述后续约定，详细代码方案见 [plan.md](plan.md)，不代表当前运行能力。

现有路径的实现边界：保留 depth completion、RGB-D 反投影、单 seed 初始化及 Inpaint360GS RGB finetune；在其后可运行默认关闭的独立 EDGS-PGSR 局部几何优化，直接用 LaMa completed normal 做二维监督。新增开关和渐增调度只属于 `run_inpaint.sh`，不改变 `run_seg.sh` 的全局训练。详见第 3.9a 节；代码已接入并通过小型合成 CUDA 链路测试，真实场景完整训练质量仍待评估。

新增同级技术路径见 [第 3.9b 节](#edgs-pgsr-direct)（**已接通，完整场景质量待评估**）：在公共 RGB/depth/normal completion 后，使用 EDGS RGB 匹配初始化，可选 depth 辅助及 normal 定向，然后直接 PGSR 联合训练，不经过 Inpaint360GS finetune 或 Stage 5b。它不替换现有路径，不修改自动 normal completion；以下涉及 seed、5a/5b 的实现描述仍专指现有路径。

自适应密度匹配是正式实现，详见第 3.7a 节和 [plan.md 的 D0 节](plan.md#d0-无手调密度阈值的自适应配额采样已实现)。设置 `SUPPORT_DENSITY_MODE=mass_adaptive`，使用 `support_mass.py` / `configs/support_mass.yaml`；`legacy` 仅表示不启用密度匹配、保持原生反投影流程，不是另一种密度匹配算法。Stage 4b 位于 Stage 4 与 Stage 5a 之间，与 Stage 5b 开关独立，不改变自动 normal completion，也不要求 normal 反投影。

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

这里描述的是 `run_seg.sh` 最初的全局重建，后续局部补全设计不会修改这一配置或公式。第 3.9a 节的 LaMa normal loss 使用独立入口、独立配置和从零开始的局部步数；不能为了让短程局部优化生效而把全局 `single_view_from_iter` 改小。

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

公共前端仍是 remove → 同相机渲染 → tracking → RGB/depth/normal LaMa completion。同级选择为 `INPAINT_PIPELINE=inpaint360gs|edgs-pgsr`，默认保留 `inpaint360gs`；此接口已接入，不能与选择虚拟渲染后端的 `VIRTUAL_RENDERER` 混为一谈。下面展示原路径默认关闭局部优化时的流程；另一条路径在第 3.9b 节定义。

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
LaMa 补全 removed RGB 与 metric depth；有 normal 时自动平级补全 normal
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

在 RGB finetune 的空间门控提交之后、模型发布之前，现已增加可选 Stage 5b：

```text
Stage 5a RGB finetune 已提交结果
    ├── LOCAL_GEOMETRY_REFINE=false -> 原样发布
    └── LOCAL_GEOMETRY_REFINE=true  -> 独立 PGSR 局部优化 -> 验证后发布
                                         ↑
                         completed depth / normal + RGB / cameras / masks
```

该分支不改变前面的反投影和初始化，也不改变后面的 RGB-D TSDF 算法。normal completion 仍然是“上游有 normal 就自动执行”，不受局部优化开关控制。

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

### 3.3 模块 B：生成严格一致的虚拟相机（默认 30 个）

实现位于 [`virtual_pose.py`](../../submodules/Inpaint360GS/tools/virtual_pose.py)、[`pose_utils.py`](../../submodules/Inpaint360GS/utils/pose_utils.py) 和 [`virtual_camera_manifest.py`](../../submodules/Inpaint360GS/utils/virtual_camera_manifest.py)。

已实现同级圆环/半球生成器与可配置总帧数。B1 保留原圆环算法，B2 说明半球螺旋；新方式不改变 normal completion。所有下游统一读取 camera manifest 的帧表，30 为默认值；后续章节出现的 30 帧及 `00029` 均按默认配置理解。

#### 输入

- full semantic 3DGS；
- removed 3DGS；
- 真实相机分布；
- 上一步计算的目标半径。

#### 计算过程

##### B1. 现有圆环：circle

系统用全部真实训练相机位姿做 PCA，估计场景主轴和归一化尺度，再通过相机光轴计算观察焦点 $F=(f_x,f_y,f_z)$。目标半径和第一台训练相机的 FoV 用来估算轨迹半径 $R$；配置中的 `circle_radius` 是相对场景尺度的比例，不是直接的世界坐标半径。

虽然底层函数名为 `generate_ellipse_path`，圆环入口使用 `is_circle=True`、`n_frames=N`（默认 30），高度变化默认为零。因此，实际相机中心位于 PCA 坐标系的 $z=0$ 平面：

$$
C=(f_x,f_y,0),\qquad
P_{circle}(\theta)=C+R(\cos\theta,\sin\theta,0).
$$

30 帧沿一圈近似等弧长分布，全部看向 $F$。需要区分**圆环几何圆心 $C$** 与**注视点 $F$**：两者只在 $f_z=0$ 时重合。当前没有启用对象中心模式，不能把这个焦点直接理解为被删除对象的中心。

##### B2. 新增半球：hemisphere（已实现）

新增独立轨迹选择 `VIRTUAL_CAMERA_PATH=circle|hemisphere`，默认 `circle` 保持现有行为；`VIRTUAL_CAMERA_COUNT=N` 配置两种模式的相机总数，默认 30，不是每圈数量。它们与 `VIRTUAL_RENDERER=inpaint360gs|edgs-pgsr` 正交：轨迹决定相机序列，renderer 决定渲染后端。以下接口已接入 `run_remove.sh`：

```bash
VIRTUAL_CAMERA_PATH=hemisphere
VIRTUAL_CAMERA_COUNT=90                       # 示例；默认 30
VIRTUAL_HEMISPHERE_MAX_ELEVATION_DEG=85        # 默认 85，接近但不取极点
```

半球严格继承同次输入下 B1 的几何圆心 $C$ 和半径 $R$，**球心不改成 $F$ 或目标对象中心，球直径等于原圆环直径 $2R$**。以原圆环平面作为赤道，沿 PCA 的正 z 方向上升；PCA 已依据训练相机平均 up 方向校正符号。这里的“上”是相机估计的场景方向，不承诺等于物理重力方向。

**球面均匀性不能用曲线等弧长替代。** 仰角从赤道量起，球面面积元为：

$$
dA=R^2\cos\phi\,d\phi\,d\theta=R^2\,dh\,d\theta,
\qquad h=\sin\phi.
$$

因此应在归一化高度 $h$ 上均匀分配样本，而不是在仰角上等间隔，或让各高度环拥有相同点数。这个面积关系也用于 [PBRT 的均匀半球采样](https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#UniformlySamplingHemispheresandSpheres)；下述连续螺旋与自动螺距是本项目据此设计的确定性近似均匀布局，不是直接使用随机采样器。

设上限仰角 $\phi_{max}=85^\circ$，$h_{max}=\sin\phi_{max}$，$i=0,\ldots,N-1$，采用：

$$
h_i=h_{max}\frac{i}{N-1},\qquad \phi_i=\arcsin(h_i).
$$

相邻高度对应的完整球带面积恒为 $2\pi R^2h_{max}/(N-1)$。起点在赤道，终点达到配置仰角；除端点效应外，各等面积球带具有相近的样本预算。这里只约束高度分布，还需合理安排方位角才能覆盖整个球面。

为兼顾相邻帧的空间连续性与二维覆盖，令 $A=2\pi R^2h_{max}$，用典型表面间距 $d=\sqrt{A/(N-1)}$ 决定每一整圈的仰角增量（螺距）$\delta=d/R$：

$$
\theta_i=\theta_0+\frac{2\pi}{\delta}\phi_i,\qquad
K=\frac{\phi_{max}}{\delta},\qquad
\delta=\sqrt{\frac{2\pi h_{max}}{N-1}}.
$$

这里 $K$ 是从首帧到末帧累计的绕行圈数，允许非整数，由 $N$ 自动得到，不再额外暴露固定圈数。原因是：沿子午线相邻螺旋圈约相距 $R\delta=d$，远离极点处相邻样本的横向间距约为 $R\cos\phi\,\Delta\theta\approx2\pi R\Delta h/\delta=d$，两方向尺度相近。若保持固定 3 圈却不断增加相机数，只会加密曲线本身，不会同步填补圈间空隙。

相机位置为：

$$
P_i=C+R\begin{pmatrix}
\sqrt{1-h_i^2}\cos\theta_i\\
\sqrt{1-h_i^2}\sin\theta_i\\
h_i
\end{pmatrix}.
$$

由此始终有 $\|P_i-C\|_2=R$，高度和展开方位角单调递增，水平截面半径随高度收缩。这是真正同心同径的球面轨迹，不是圆柱螺旋。按 $i$ 直接生成 `00000..{N-1:05d}`，一圈结束后沿同一螺旋继续上升，不在接缝重置方位、反向或回到赤道；不额外叠加一圈底层相机，不做会破坏高度面积分配的二次等弧长重采样。

在 85° 上限下，30/60/90 个相机约对应 3.19/4.56/5.59 圈。顶部完整圈的面积较小，因此自然分配更少的相机，不要求每圈数量相同。85° 以上的小顶帽不采样，故“均匀”指赤道至该仰角的球面带；有限点集是近似等面积、近似等间距，而非任意两点等距或每个球面 Voronoi 单元严格等面积。低帧数的端点效应更明显，必须用球面覆盖诊断验收，不能只检查点落在球面上。

起始方位沿用原圆环首帧，首尾都保留；不能套用闭合圆环“删除重复尾点”的处理而丢失半球终点。末端仰角满足 $0<\phi_{max}<90^\circ$，不生成重复极点；默认 85° 不是质量保证。$N$ 必须为至少 2 的整数，小 $N$ 仅表示数值上可生成，不能保证绕满多圈或获得可用 tracking，亦须满足后续 seed/debug 索引约束。

**姿态固定锚定场景 up，不累计滚转。** 所有相机仍看向原注视点 $F$，而不是自动改看球心或重新定位对象；内参、尺寸和坐标变换与圆环模式一致。场景参考上方向 $U$ 与原圆环完全相同：取 PCA 中训练相机平均 up 的最大绝对分量轴及其符号。这里的 $U$ 是构图基准，区别于半球位置沿 `+z_pca` 上升的方向。

对每个位置独立构建 look-at 基，不读取上一帧的 right/up：

$$
b_i=\frac{P_i-F}{\|P_i-F\|},\qquad
r_i=\frac{U\times b_i}{\|U\times b_i\|},\qquad
u_i=b_i\times r_i.
$$

$(r_i,u_i,b_i)$ 是右手系的 right/up/backward；按现有轴翻转、PCA 逆变换写成 renderer 的 right/down/forward 相机。这样相机 up 始终对齐 $U$ 在像平面的投影，相对于场景参考的 roll 为零，不随绕行圈数累积。首帧直接复用圆环首帧的精确矩阵；其余帧独立计算。底层已有 PCA 尺度和 depth 尺度保持不变，不新增姿态模式开关。

默认最高仰角为 85°，不经过极点。若视线长度退化或 $\|U\times b_i\|\le10^{-12}$，构图的 up/roll 无法可靠定义，生成器明确报错，需降低末端仰角或检查输入相机；不通过继承上一帧姿态掩盖退化。无累计滚转不表示顶部每帧的整体旋转角都很小：方位仍在变化，需查看轨迹诊断及实际图像。

本模式**只有首帧在赤道**，不是先生成一整圈赤道再上升；例如 N=90、85° 时第 23 帧（索引 23）已升至约 14.92°。与原圆环比较时只有首帧保证位置和姿态相同，同名后续帧不对应同一观察位置。这一取舍优先保证球面近似均匀覆盖。

同样的 30 帧要覆盖更多高度，每圈会比原单圈稀疏；可以显式增大 $N$，但渲染、LaMa 与 tracker 工作量也随之增加。空间连续不等于画面一定重叠，必须检查 tracker 传播、目标遮挡及高处空洞；相机中心均匀也不代表表面观测、图像分辨率或补全质量均匀。默认 `FUSION_SEED_FRAME=4` 仍表示第 5 帧，要求 `0 <= seed < N`，越界报错、不静默换帧；切换轨迹或帧数后应重新检查 seed 覆盖，不复用旧 masks、LaMa 或 fusion 产物。

##### B3. 精确相机复用与渲染

以下为现有的共同下游契约，新增半球模式也必须遵守。新生成器只运行在 `run_remove.sh` Stage 4；不影响 `run_seg.sh` 全局训练，不让各个 depth/normal 模块独立生成相机。

同一组相机分别渲染：

- `VIRTUAL_RENDERER=inpaint360gs`（默认）：full/removed 的 RGB、语义、原生 `depth_3dgs` 和 alpha；
- `VIRTUAL_RENDERER=edgs-pgsr`：full/removed 的 RGB、PGSR `plane_depth`、直接 Gaussian normal 和 alpha。

调度位于 [`render_virtual_views.py`](render_virtual_views.py)，两个同级适配器位于 [`render_virtual_worker.py`](render_virtual_worker.py)。调度器在独立子进程中加载所选项目，避免同名 `scene` / `utils` 模块冲突。相机先由 `virtual_pose.py --poses-only` 写入 manifest，两个后端均读取同一份参数；full/removed 使用相同后端。在设置 `surrounding_ids` 时，补全输入延续原流程，来自 target + surrounding 均被移除的版本。

每次渲染同步导出 `rgb_raw/*.npy`（float32 H×W×3）、`depth/*.npy` 和 `alpha/*.npy`（float32 H×W）。PGSR 还导出 `normal/*.npy`（相机坐标系、朝相机的单位向量）、`normal_valid/*.png` 和 `normal_vis/*.png`；低 alpha、非法深度或退化法线对应的 normal 置零。原生后端不提供直接 Gaussian normal，manifest 中声明该能力缺失。

`render_manifest.json` 记录 backend、depth 定义、normal 来源、模型与相机 hash、逐帧输出 hash。原生 depth 保留原数值定义，PGSR depth 是场景尺度下的平面 z-depth，不能仅凭名称假定是已标定的米制深度。虚拟后端选择独立于基础训练、finetune 和最终 TSDF 配置。

removed RGB 按 manifest 中有序的 N 个帧名打包为 tracker `images.zip`（默认 `00000.png..00029.png`）；tracker、LaMa、fusion、密度匹配、Stage 5a/5b 和发布校验全部读取同一个 N。每个相机的完整精度 `R`、`T`、FoV、宽高、near/far、scene translation/scale 写入 `virtual_cameras.json`。后续所有模块复用该 manifest，而不是用四舍五入后的半径重新生成轨迹，从而避免 mask、RGB、depth 和反投影之间的像素偏移。

新 tracking session 还绑定 removed render artifact。切换后端必须使用新的 removal run，已有会话和下游工作区不能复用另一后端的几何产物。normal/alpha 会随 inpaint 工作区一起校验和链接；normal LaMa completion 已接入 Stage 2/3，normal loss 位于可选的独立 Stage 5b。

半球模式已经把轨迹类型、N、实际球心/半径、注视点、坐标基、自动螺距/圈数、末端仰角、面积采样规则及 `orientation=scene_up` / `scene_up_pca` 纳入 schema 2 相机 manifest 的身份校验；默认圆环 30 帧保持 schema 1。圆环 30 帧 manifest 按原身份读取，不能补字段后冒充旧缓存；切换轨迹、帧数、姿态策略或其他有效参数必须使用新的 removal run，重新渲染并完成 tracking。生成算法身份不匹配会提前拒绝复用，不保留其他半球姿态实现。详细接口、动态帧数改造、兼容性和验收计划见 [plan.md 的 A0 节](plan.md#a0-同级圆环半球虚拟相机生成方式已实现)。

normal completion 规则是：只要上游提供完整的 removed normal，就自动执行 `removed normal → LaMa → completed normal`，与 depth 分支平级；不新增启用开关，不以 completed depth 推导法线替代此分支。

#### 输出

```text
tracker/images.zip
tracker/virtual_cameras.json
tracker/camera_trajectory.json                # 位置、仰角、相邻姿态/视轴角、scene_up_roll、球面覆盖
tracker/camera_trajectory.svg                 # 带编号的俯视/侧视轨迹
work_model/virtual/ours_<N>/...                  # full
work_model/virtual/ours_object_removal/...       # removed
```

### 3.4 模块 C：交互式 SAM-Track 得到跨虚拟视角 mask

PaintMesh 使用专用简化页面 [`tracker_web.py`](tracker_web.py) / [`tracker_web.html`](tracker_web.html)。页面自动载入当前 run 的序列，用户在首帧添加正/负点、预览分割后点击 Start Tracking；无需上传、解压或手动初始化。SAM 点选与 DeAOT 传播复用原 tracker 的模型接口，不加载 GroundingDINO，也不执行周期性 segment-everything / 新对象发现，以避免把无关物体加入补全 mask。

#### 输入

- removed-scene `images.zip`；
- 用户在第一帧提供的正/负点提示（支持撤销、清空）；
- SAM、DeAOT checkpoint。

#### 计算过程

SAM 在首帧产生目标 mask，DeAOT 把对象状态传播到其余虚拟帧。

简化页允许跟踪前修正首帧，跟踪完成后逐帧预览；暂不提供原复杂页面的逐帧回滚重标功能。跟踪失败不发布半成品，成功才原子发布完整 masks；已有 masks 拒绝覆盖。点击“完成并返回流水线”会退出网页服务，再由原 Stage 5 校验器提交会话。

PaintMesh 对会话进行严格绑定：

- 必须输出 camera manifest 声明的全部有序 N 帧（默认 `00000.png..00029.png`）；
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

#### D1. 自动 normal 输入扩展

在上述输入准备阶段自动读取上游模态声明：有完整 removed normal 时，增加 `lama/input/normal/<frame>.npy` 和与 RGB/depth 完全相同的 `<frame>_mask.png`，并保存 `valid/<frame>.png`。normal 为 float32、H×W×3、相机坐标系的单位向量，无效值为零；不使用 `normal_vis` 作为数值输入，也不使用 full normal 填洞。

normal predictor 内部使用 `M_infer = hole_mask | ~normal_valid` 排除无效上下文，但最终只允许改写共同 hole mask 内的像素。上游明确不提供 normal 时保持 RGB/depth-only；声明提供却缺帧或校验失败时必须报错，不能静默跳过。normal 输入与相关 hash 纳入现有 `lama_input_manifest.json`。

### 3.6 模块 E：LaMa RGB、depth 与自动 normal completion

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

#### E3. 同级 normal LaMa completion

目标流程是三条平级分支，不要求三个推理进程同时运行：

```text
removed RGB    → LaMa → completed RGB
removed depth  → LaMa → completed depth
removed normal → LaMa → completed normal   # 上游有 normal 就自动执行
```

实现为与 `predict_depth.py` 同级的 [`LaMa/bin/predict_normal.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_normal.py)，复用现有 `LAMA_MODEL_PATH` 与 depth 的 refinement 设置。独立 normal loader 与 CPU 编解码/校验函数位于 [`tools/paintmesh_normal.py`](../../submodules/Inpaint360GS/tools/paintmesh_normal.py)，因为现有 `.npy` loader 按 depth 读取 `depth_original/` 并做逐图 min/max 归一化，不能直接用于向量法线。

三个通道固定表示 nx、ny、nz；输入使用 `(N + 1) / 2` 编码，预测经 `2 * clip(P, 0, 1) - 1` 解码。对洞内有限、非退化向量单位化，再按同一相机的像素射线校正朝向。非有限或近零向量标记无效并置零；mask 外直接复制 removed raw normal 与 validity，保持数值完全不变。洞内全无有效预测或 mask 外无有效上下文时失败，不以 depth-derived normal 隐式回退。

新增产物：

```text
lama/output/normal/00000.npy ... 00029.npy      # float32 HWC
lama/output/normal/valid/00000.png ... 00029.png
lama/output/normal/vis/00000.png ... 00029.png  # 仅预览
```

Stage 3 根据输入 manifest 自动运行 normal predictor，不增加 normal 开关或模式。现有 `lama_completion_manifest.json` 扩展为校验全部预期模态：只要输入含 normal，输出缺 normal 就不能提交成功或复用旧 RGB/depth-only 缓存。`normal/prediction.json` 记录推理的输入、模型/config 和逐帧输出 hash，不替代总 completion 标记。

这是使用 RGB LaMa 权重对编码法线做补全的基线，不是专门训练的法线模型；单位化不能保证与 depth 或多视角几何一致。completed depth 派生法线用于后续质量诊断，不覆盖 LaMa normal。completed normal 已可作为第 3.9a 节的软监督；第 3.9b 节已接通同级路径的可选 normal 初始化与训练监督，不要求先改造原 RGB-D 融合器。

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

一个重要的当前实现细节是：虽然生成 30 个支持 PLY，Stage 5 只选择 `FUSION_SEED_FRAME` 指定的一张，默认 `00004.ply`，用来初始化新 Gaussian；30 张 completed RGB 则全部用于后续多视角优化。此依赖仅属于现有路径；第 3.9b 节的新路径不要求 seed PLY 或此 fusion manifest。

现有路径的 Stage 5b 局部优化方案保留以上计算和输出不变：不反投影 normal，不新增 RGB-D-N 融合，不将 N 个 PLY 合并成新的初始化点云。completed depth 继续提供三维初值，同时在启用 Stage 5b 后新增二维深度监督；completed normal 直接作为同相机的二维方向监督。第 3.9b 节的新路径在 completion 后分流，不需要此处的支持 PLY。

### 3.7a 无手调密度阈值的自适应配额采样（已实现）

#### 先改变问题定义：按需要的数量采样，不靠过滤后剩下多少点

原反投影已是 mask 内每像素一个点，但初始化只使用一张 seed。因此它受图像分辨率、距离和表面倾角限制，不自动匹配周边 Gaussian 的密度。后续 RGB densify/prune 和 gate 还会改变点数；Stage 5b normal loss 不负责补点。

自适应采样以**表面面积对应的中心数量**为目标，通过面积与参考密度计算采样配额。密度估计与几何可信程度分开：几何不确定性应被报告，不能简单乘到点数上，让不确定的洞越来越稀。

#### 自动建立周边密度参考

参考点来自与 Stage 5a 删除规则一致的保留 Gaussian，不使用被删物体的点，也不按 30 个视角重复计数；精确重复 XYZ 只保留一个位置。目标是中心数/表面积，而不是三维包围盒体密度或 opacity 总和。当前估计器以局部欧氏邻域的 `πr²` 近似表面积，在光滑单表面上适用；紧邻双层、折叠或厚体积点分布可能偏高，不能宣称已经严格分离所有表面。

代码按 `k=2,4,8,16,32` 的可用邻域，用外环留出计数的 Poisson 预测分数对密度做模型平均，不依赖用户设定一个 k 或环带宽度。这是版本化的有限模型族，不是无限尺度搜索。已知区 alpha、opacity、深度残差和可见性形成连续观测权重，跨视角取每个中心的最大可靠度，不重复加点。随后在 seed 有效深度网格上用逆深度差的连续边权求解 `log(rho)`；没有角度/覆盖率门槛，也不读取 LaMa normal 来决定密度。

参考估计使用全部保留中心构建邻域，已知区投影提供局部边界条件，避免只取薄环导致边缘偏差。少于三个不同参考中心、没有有效参考观测或有效深度连通域完全没有锚点时返回 `no_reference`；这是当前统计估计器的可辨识边界，不是可调的 `min_reference` 密度门槛。

#### 每个像素拥有表面元，不要求先形成三角网

每个有效 seed 像素先形成一个属于自己的表面元：以 completed z-depth 为中心锚点，用自动尺度的单侧局部逆深度模型给子像素射线赋深度。像素单元裁剪到 mask 与原 gate 内；不同单元之间不连接跨深度层的三角面，也不对跨层深度做普通双线性平均。

候选包含前平行模型及半径 1/2/4 像素的全邻域、四种单侧逆深度模型。留一预测分数和多视角软残差选择一个假设，不对前后层深度求平均；不能辨认斜率时仍可选前平行面元，不删除像素。`geometry_uncertainty` 记录候选歧义，不保证真实几何。面积由世界坐标曲面的雅可比积分计算：`A = ∫ ||∂X/∂u × ∂X/∂v|| du dv`，当前采用每像素 4×4 子单元求积；刚性相机下前平行面元为 `z²/(fx*fy)`，斜面使用完整雅可比。

#### 缺点配额决定补多少点

对每个表面元 u：

```text
目标点质量 t_u = ∫ rho_target(x) dA
现存点贡献 b_u = 保留背景在该表面的软归属（每个中心总贡献不超过 1）
需补质量   m_u = max(t_u - b_u, 0)
总新增点数 N   = round(Σ m_u)
```

`rho_target` 直接来自周边，不再设置密度倍率。举例：同一表面周边约 20,000 点/场景单位²，洞表面积 0.10 场景单位²，已有 300 个有效中心贡献，则目标约 2,000、需补约 1,700 个点。这是数量推导，不是固定配置；COLMAP 未标定时不能把场景单位称作米。

将连续质量按空间顺序做系统分层重采样，得到总和严格为 N 的整数配额；每个面元获得其归一化配额的向上或向下取整。在本面元内按面积分层放置不同子像素点，不复制 XYZ、不三维随机撒点、不放大 splat 伪装稠密。背景过密不删背景；无需新点时允许 N=0。

其他视角通过连续的残差似然、遮挡模型和视角相关性参与面元的几何假设选择，不按“支持比例超过阈值”删点。自适应采样不做统计离群点/半径过滤，防止生成的配额被二次削减；若可行几何不存在则报告未解决域，不以非法坐标充数。采样密度提高不是增加真实几何信息，也不能证明 LaMa depth 正确。

#### 保持 normal 分支与编辑边界

```text
completed depth -> 像素表面元 -> 周边密度配额 -> 新支持点 -> Stage 5a
completed normal ---------------------------------------> Stage 5b normal loss
```

normal 不反投影，也不作为硬采样门槛。`init_support` 与原 `gate_support` 仍分离，Stage 5a/5b 的编辑范围不扩大，背景不修改；世界尺度、相机和原始 depth 定义不变。原 gate 虽有已有距离界限，但那是编辑许可边界，不是新的密度调节阈值。

“无阈值控制密度”不等于取消所有约束：NaN/Inf、非正深度、来源不匹配、无参考、不可行几何和资源耗尽仍应明确处理。点数由面积与密度决定，不能受 500k/8 倍限制截断后还宣称匹配；内存不足分块或报告所需资源。算法仍有统计模型、离散选择和数值精度设置，不应宣传为零参数。

`diagnostics.json` 报告参考密度分位数、全 hole/原 gate 内面积、gate 外目标质量、已有贡献、过密 surplus、新增配额与取整误差。`density_ratio` 是**配额兑现比**，不是独立估计的真实几何精度。初始化、5a gate 前后及 5b 另有支持邻域占用/间距审计；这些诊断不删点或阻断质量，不可当作训练后密度保证。RGB/alpha/depth/normal 的视觉质量仍需结合既有 PGSR 渲染/debug 检查，不声称本预处理已经验证完整训练效果。

#### 代码、产物与使用

入口为 [`prepare_density_support.py`](prepare_density_support.py)，算法为 [`support_mass.py`](support_mass.py)。新增 `fused/density/density_field.npz`、`quota.npz`、`support.npz`、`init_support.ply`、`diagnostics.json`，debug 包含目标密度、配额、模型不确定性和 seed 支持点预览。`manifests/support_density_manifest.json` 绑定模式、算法版本、配置、源码与所有输入/输出；提交完成标记前检查配额守恒及 PLY/sidecar 一致性。

使用 `SUPPORT_DENSITY_MODE=mass_adaptive`，默认自动选择 `configs/support_mass.yaml`；该文件只允许算法版本、seed、batch size、内存预算和 debug，传入角度/覆盖率/密度倍率等不支持的参数会报错。切换模式或算法版本必须用新 `INPAINT_RUN_NAME`。支持单独 `END_STAGE=4` 验证补点，然后以同一配置从 Stage 5 续跑；完整命令见 [plan.md 的 D0.10](plan.md#d010-运行命令与验证记录)。

当前已验证合成几何/守恒、真实 kitchen Stage 4b、CUDA 初始化及小型局部优化链路，未执行 kitchen 的 5000 步完整训练。没有修改上游 normal completion 或 `run_seg` 全局训练。

### 3.8 模块 G：从 seed 支持点初始化新的 Gaussians

实现位于 [`GaussianModel.inpaint_setup`](../../submodules/Inpaint360GS/scene/gaussian_model.py) 和 [`compose_utils.py`](../../submodules/Inpaint360GS/utils/compose_utils.py)。

#### 输入

- source semantic 3DGS；
- 目标 ID 与 classifier；
- 默认 `00004.ply` 支持点云；
- `opacity_init=0.1`。

#### 计算过程

系统再次用 classifier threshold 与凸包找出目标区域，保留非目标 Gaussians。`legacy` seed PLY 经 Open3D statistical outlier removal：`nb_neighbors=5, std_ratio=4.0`；绑定有效 density manifest 的初始化跳过此二次过滤，保持采样配额。零新增点时保留背景并跳过追加；只有 1～3 个新点时，将保留背景加入近邻 scale 估计，避免 `distCUDA2` 因不足三个邻居产生无限 scale，不增加额外点。其余情况下每个支持点初始化一个新 Gaussian：

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

现有路径保持这些初始化规则不变。使用 normal 调整初始 quaternion/各轴 scale 不是接入 Stage 5b 的前置条件；其可选实现归入第 3.9b 节独立路径，不修改这里的初始化器。

### 3.9 模块 H：30 视角 3DGS finetune

实现位于 [`edit_object_inpaint.py`](../../submodules/Inpaint360GS/edit_object_inpaint.py)。这个阶段使用 Inpaint360GS 的 Gaussian renderer，称为 Stage 5a，算法保持不变；PGSR renderer 用于可选的 Stage 5b 及后续 mesh 重建。Stage 5b 单独见第 3.9a 节。

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

独立开关 `RGB_FINETUNE_DENSIFY=true|false` 控制 Stage 5a 的 clone/split/prune，默认 `true` 保持原行为。它与 `SUPPORT_DENSITY_MODE`、`LOCAL_GEOMETRY_REFINE` 独立，不影响 `run_seg` 或 Stage 5b。

开启时，当 `500 < t < 5000` 且 `t` 是 100 的倍数时，系统根据可见 Gaussian 的屏幕空间位置梯度：

- clone 尺度较小的高梯度新 Gaussian；
- split 尺度较大的高梯度新 Gaussian；
- prune opacity 低于 `0.005` 或异常大的新 Gaussian。

保留场景的前缀被排除于 densification，并在当前调用的 size threshold 分支中受到 prune 保护。

关闭时同时跳过 max radii 更新、densification 梯度统计以及整个增删点调用；RGB loss、反向传播和 optimizer 更新保持执行。初始化筛选和最终空间 gate 不受开关影响，因此关闭只保证训练循环不主动增删 Gaussian，不保证最终提交点数或空间密度不变。

Shell 将开关传给 `edit_object_inpaint.py --disable_rgb_densify`（仅关闭时传入）。在 Stage 5a PLY 同目录保存 `point_cloud.density_policy.json`，并在存在 RGB finetune receipt 时记录 `parameters.rgb_densify`。原生 RGB-D、未启用 normal/局部优化的 run 也校验 policy。更改开关或恢复缺少 policy 的结果需新运行目录，不能复用另一设置下的训练结果。

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

### 3.9a 模块 H 扩展：独立 EDGS-PGSR 局部几何优化

#### H3. 为什么直接使用二维 normal loss

depth 表示表面位置，normal 表示表面朝向。法线没有距离信息，不能单独反投影出三维点；两个位置不同的平行平面也可以具有相同法线。本轮不修改第 3.7 节的反投影，只在 PGSR 渲染的同一虚拟相机下比较预测和 LaMa completed normal。

两者都使用相机坐标系 `+x 右、+y 下、+z 前`，有效法线单位长度且朝向相机。目标来自 `lama/output/normal/<frame>.npy` 和 `valid/<frame>.png`，不是 `vis/*.png`，也不是 full-scene normal。无需将 normal 变换到世界坐标。

#### H4. 独立入口、开关与不变部分

入口为 [`finetune_pgsr_geometry.py`](../../submodules/EDGS/tools/finetune_pgsr_geometry.py)，由 `run_inpaint.sh` 在独立子进程调用，直接加载 Stage 5a 的最终 PLY；不重新运行 RoMa、全局重建或 seed 初始化。复用 PGSR 可微 renderer，不调用全局 `PGSRLossComposer`。局部损失位于独立 [`paintmesh_local_losses.py`](../../submodules/EDGS/source/paintmesh_local_losses.py)，配置位于 [local_geometry.yaml](configs/local_geometry.yaml)，跨项目 CPU 产物契约位于 [local_geometry_io.py](local_geometry_io.py)。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `LOCAL_GEOMETRY_REFINE` | `false` | 唯一的局部优化启用开关 |
| `LOCAL_GEOMETRY_CONFIG` | `scripts/paintmesh/configs/local_geometry.yaml` | 局部训练配置，不覆盖全局 PGSR 配置 |
| `LOCAL_GEOMETRY_ITERATIONS` | `1000` | 局部总步数 `T` |
| `LOCAL_GEOMETRY_FROM_ITER` | `100` | 几何权重开始渐增的位置 `s` |
| `LOCAL_GEOMETRY_RAMP_ITERS` | `400` | 几何权重渐增长度 `r` |

这些数值是首轮实验起点，不是已验证的最佳超参数。显式环境变量覆盖局部 YAML，解析后的配置保存到当前 run。关闭开关时不要求局部配置、normal 目标或新 sidecar；自动 normal LaMa 仍照常执行。开启并进入 Stage 5b 时必须有经过验证、与当前相机/模型一致的 completed normal 和 plane z-depth，否则报错。

`run_seg.sh`、全局 `configs/gs/pgsr.yaml`、`source/pgsr_losses.py` 的公式和 7000 步门槛不变；基础 `edgs/`、语义 3DGS、removed 模型和它们的配置均只读。`FINETUNE_ITERATION=5000` 仍只控制原 RGB 阶段，与局部步数独立。

#### H5. 新增 LaMa normal 监督与局部总损失

PGSR 返回的 normal 为 alpha 加权向量，训练时在 Tensor 内解码并保留梯度：

$$
\hat N=\operatorname{normalize}
\left(\frac{N_{render}}{\max(A_{render},\epsilon)}\right).
$$

目标从 HWC 转为 CHW 后，与同像素预测比较：

$$
L_{LaMa\ normal}
=\frac{\sum_p w_p[1-\operatorname{clamp}(\hat N_p^T N^*_{LaMa,p},-1,1)]}
{\sum_p w_p+\epsilon}.
$$

`w` 限制在 hole 内的合法目标和有效渲染区域，当前实现有效目标统一权重，不把 `normal_valid` 叫作置信度。统一朝相机后使用有符号点积；使用绝对点积可能掩盖反向或相机轴错误。

另有两项几何约束：

```text
L_depth       = SmoothL1(log(D_pred), log(D_completed)) 的有效域均值
L_consistency = 1 - dot(N_pred, N_depth(D_pred)) 的有效域均值
```

前者限制表面位置，后者限制模型内部深度与法线协调；`N_depth(D_pred)` 保留对预测深度的梯度。LaMa normal 是外部伪目标监督，与第 1.5 节原有的内部 normal 一致性不同，必须分别记录。

局部阶段开始时，用同一 PGSR renderer 缓存输入模型的 RGB/alpha 基线 `I0/A0`，停止其梯度。总目标为：

$$
L(t)=L_{RGB}+\lambda_A L_{alpha}
+a(t)\left(\lambda_D L_{depth}
+\lambda_N L_{LaMa\ normal}
+\lambda_C L_{consistency}\right).
$$

RGB 在洞内对齐 completed RGB、洞外对齐固定的 `I0`；洞外不能用含被删除对象的真实 RGB 作目标。alpha 保持项在有效 completed depth 目标域内惩罚 `A_pred` 低于 `max(A0, 0.1)`，防止通过降低覆盖率逃避几何监督。初始建议 `λD=0.10、λN=0.05、λC=0.015、λA=0.1`，均是需要实验验证的局部参数，不继承全局损失权重。

深度要求同尺度的正值 plane z-depth；法线要求 finite/valid；预测 alpha 过低时不算角度项，但必须统计覆盖率。不能用洞内近零的 removed alpha 屏蔽新预测。差分法线须排除无效邻域、洞边缘和深度跳变，先选择有效值再计算 log 等运算。空有效域返回可反传的零并计数，持续无监督或覆盖崩溃时不发布成功。

LaMa RGB 权重产生的 normal 只是伪目标，有限且单位长度不代表几何正确。当前实现用较小权重、渐增与一致性诊断控制其影响，不用 depth-derived normal 覆盖 LaMa 输出。

#### H6. 局部启用时间与逐步增加权重

局部循环使用 `t=0..T-1`，不是基础重建或 RGB finetune 的累计步数：

$$
a(t)=\operatorname{clip}\left(\frac{t-s}{r},0,1\right),\qquad r>0.
$$

例如 `s=100,r=400`：`t<=100` 仅做 RGB/alpha 保持；`t=300` 几何项为一半权重；`t>=500` 达到完整目标权重。要求 `T>0、0<=s<T-1、r>0、s+r<=T-1`，在训练前校验。恢复时同时恢复 local step、optimizer 和随机状态，不重新开始渐增。

是否请求 plane/normal 输出由局部实际损失决定，不能沿用全局 `required_outputs(step)` 的 7000 步门槛。将局部总步数改为 5000 不会更改任何全局训练设置。

#### H7. 如何真正保证“局部”

仅用二维 mask 计算 loss 不等于冻结三维背景。启用局部优化时，Stage 5a 最终输出处额外保存 `editable_mask.npy` 与 `rgb_finetune_manifest.json`，记录既有 seed 空间 gate 允许编辑的最终 PLY 行，包括 gate 内新增点和允许提交的原有点；绑定点数、行序、PLY、seed、相机和 mask hash。只记录范围，不更改原 gate、训练或初始化算法；恢复的 surrounding 对象追加行标记为冻结。

Stage 5b 仅为编辑行的 XYZ、rotation、scale 创建 optimizer 参数；其他行以及 SH、opacity、16D embedding、classifier 全部冻结。局部张量与常量背景一起做完整场景渲染以保留遮挡。当前实现不 densify/prune、不 reset opacity、不改变点数/行序，不使用全局 scale、多视角几何或 LNCC 项。

保存时保留输入 PLY 所有字段，只更新许可行/字段；检查越过原空间 gate 的更新并回退到 Stage 5a 值，再计算最终诊断。冻结背景参数之外，还要检查 mask 外渲染变化，因为局部 Gaussian 的投影覆盖仍可能影响已知像素。

normal loss 不会自动增加点数，Stage 5b 本身不承诺解决稀疏覆盖；第 3.7a 节已在初始化前单独改善支持点密度。多视角混合初始化与可选 normal-aware 初始化属于第 3.9b 节的新路径，不纳入 Stage 5b。

#### H8. 独立产物与发布选择

局部 debug 默认开启：独立配置 `debug.enabled=true`，每 100 次更新固定渲染视角 `00004`，输出至 `local_geometry/debug/`。拼图展示 completed/当前 RGB、depth、normal，以及洞内法线角度误差和 alpha/hole 边界；配套 JSON 记录固定视角 loss、实际权重和覆盖率。还保存优化前与最终 gate 后图像。额外渲染不参与反向传播或随机视角采样，使用固定目标深度色阶；不修改 `run_seg` 全局 debug 和 CUDA `pipeline.debug`。详见 README Stage 5b 的配置说明。

```text
<INPAINT_RUN_ROOT>/local_geometry/
├── config.resolved.yaml
├── editable_mask.npy
├── checkpoints/
├── point_cloud/iteration_1000/point_cloud.ply
└── diagnostics/
<INPAINT_RUN_ROOT>/manifests/local_geometry_manifest.json
```

局部输出不覆盖 Stage 5a PLY。manifest 绑定输入模型、编辑范围、LaMa completion、相机/mask、配置/实现版本、随机种子、调度、输出 hash 和完成状态。失败不得静默退回 Stage 5a 并宣称局部优化成功。

保留原 Stage 1..8 编号：5a/5b 同属 Stage 5，`END_STAGE=5` 包含开启后的局部优化；从 Stage 6 续跑必须验证所选来源已完成。重入 Stage 5 先验证/复用 5a，再运行或恢复 5b；旧 PLY 没有可信编辑 sidecar 时要求重建 5a，不能猜测行对应关系。

Stage 6 按开关选择 5a/5b；关闭时保持旧发布契约，开启时要求局部完成记录。发布目录仍用 `iteration_<FINETUNE_ITERATION>` 作为兼容标签，manifest 分别记录 RGB 步数与局部步数，例如 `5000 + 1000`，不能把 `iteration_5000` 当作全部优化步数。新的 model artifact 会使旧 render/mesh/semantic 缓存失效。

验收重点是关闭开关回归、局部/全局调度隔离、法线坐标与梯度正确、冻结字段不变、覆盖率及 mask 外变化、断点/过期产物拒绝。完整文件落点和测试清单见 [plan.md](plan.md) 阶段 F。

<a id="edgs-pgsr-direct"></a>

### 3.9b 同级技术路径：EDGS 可选 RGB-D-N 初始化 + PGSR 联合训练（已接通）

#### 路径定位与公共前端

这是与现有 `Inpaint360GS RGB finetune → 可选 PGSR 局部几何优化` 平级的路径，不是其后追加的 Stage 5c。两者复用同一 removal、tracking、精确相机与 LaMa completion 契约，之后分流；不重估相机，不重新训练全局场景。

```text
completed RGB / depth / normal + masks + N 个精确相机
    ├── inpaint360gs（现有、默认）
    │     RGB-D 支持点 → 单 seed 初始化 → RGB finetune → 可选局部几何优化
    └── edgs-pgsr（独立实现）
          EDGS RGB 匹配 / 三角化
            + 可选 depth 位置辅助及补点
            + 可选 normal 表面定向
          → removed 场景 + 新 Gaussians → PGSR 外观/几何联合训练
                         ↓
             验证并发布 → PGSR RGB-D TSDF → 语义提升
```

EDGS 的基础能力是从稠密 RGB 对应三角化得到 Gaussian 初值；PGSR 提供平面感知渲染与几何优化能力。这里的局部混合初始化、开关和损失组合是本项目的设计，不是上游现成的 inpaint 功能。参见 [EDGS 官方实现](https://github.com/CompVis/EDGS)、[PGSR 论文](https://arxiv.org/abs/2406.06521)。

#### 三种初始化配置

以下参数已由 `run_inpaint.sh` 解析，独立默认配置为 `configs/edgs_inpaint.yaml`：

| `EDGS_INIT_USE_DEPTH` | `EDGS_INIT_USE_NORMAL` | 初始化内容 |
|---|---|---|
| `false`（默认） | `false`（默认） | 仅 EDGS completed RGB 匹配与三角化 |
| `true` | `false` | EDGS RGB 匹配 + completed depth 辅助初始化 |
| `true` | `true` | EDGS RGB 匹配 + completed depth + completed normal 混合初始化 |
| `false` | `true` | 本设计不支持，预检报错，不自动打开 depth |

本批按上述三种需求定义初始化：RGB 匹配始终执行，depth/normal 不是必需前置条件；normal 辅助作为 RGB-D 初始化的扩展。初始化开关只控制消费哪些信息，不控制公共 LaMa 阶段：只要上游提供 removed normal，仍默认完成 normal completion。

训练监督另设 `EDGS_TRAIN_USE_DEPTH` / `EDGS_TRAIN_USE_NORMAL`：未指定时分别继承解析后的初始化开关，显式指定时独立生效。例如 RGB-only 初始化可以后续开启 depth/normal loss；要做全程 RGB-only 消融，则初始化和外部监督的两个开关都关闭。PGSR 自身渲染的 depth/normal 及模型内部一致性不是 LaMa 外部监督，不由初始化开关关闭。

#### RGB 匹配始终是初始化基础

在有重叠且基线适当的视图对上对 completed RGB 做 RoMa 稠密匹配；保留洞外上下文，但新增候选只提交到补全区域。匹配器使用冻结权重，只服务初始化，不参与后续 Gaussian 参数的联合训练。候选经过双向/多视角对应检查、正深度、三角化条件和鲁棒重投影校验，保留参考帧、像素、支持视图及置信度。N 从相机 manifest 读取，不固定 30，不只按编号相邻选对；半球序列也考虑跨圈重叠。

RGB-only 时不得用 completed depth/normal 做点位筛选、可见性门控、置信度、尺度估计或隐式补点。匹配可靠性来自 RGB 对应、相机和三角化几何；弱纹理时覆盖可能不足，无有效候选必须失败，不能暗中回退 depth 或现有 seed 路径。初始化颜色来自 completed RGB，旋转与尺度沿用 EDGS 的单位四元数/各向同性初值；不能把 normal PNG 当特征图输入匹配器。

全局 `Trainer.init_with_corr` 包含删除已有初始化点和统一调整尺度的逻辑，不能直接施加于 removed 场景。新入口只复用匹配与三角化能力，再将候选追加到冻结背景，不替换背景。

匹配使用**双向原始置信度硬门槛**，而不是只按分数加权随机抽样。对于 $p_A\to p_B$，在反向置信度图的 $p_B$ 位置插值得到 $c_{B\to A}(p_B)$，令

$$
c_{pair}=\min\bigl(c_{A\to B}(p_A),c_{B\to A}(p_B)\bigr),\qquad c_{pair}\geq\tau_c.
$$

`init.confidence_min` / `EDGS_MATCH_CONFIDENCE_MIN` 默认 **0.5**，越大越严格；这是网络置信分数，不是经过标定的几何正确概率。分数必须有限且处于有效概率范围，零分始终丢弃。门槛在抽样、三角化前执行，不能用同一数组下标代替反向目标坐标，也不能用两方向平均值让高分掩盖低分。通过门槛后，仍需满足双向回环误差 `init.cycle_pixels`（默认 3 个原图像素）、三角化重投影误差 `init.reprojection_pixels`（默认 2 个原图像素）与视差角条件。后两者是误差**上限**，减小才会更严格；不随 RoMa 内部匹配分辨率直接改变单位。`samples_per_pair` 仅是采样上限，合格匹配不足时保留实际数量，不降低门槛或补回被拒绝的匹配。匹配器不调用 RoMa 的 `sample()`；其上游 `sample_thresh` 是采样权重处理参数，不能代替这里的硬筛选。

每个图对在 `edgs_init/diagnostics.json` 中记录洞内候选、双向置信度分位数、置信度通过数、回环通过数、抽样数、三角化与最终保留数。`support.npz` 中 `source_kind=0` 的 `confidence` 是双向分数的最小值；`source_kind=1` 是 depth 补点，其分数仍是独立的深度一致性证据，不能套用 RoMa 门槛解释。提高阈值通常会减少低可信匹配，但不保证清除几何杂散点，尤其不能直接过滤 depth 补点。LaMa 伪纹理仍可能形成高置信错误对应；过高门槛也可能损失弱纹理覆盖。改阈值必须重新初始化，不能在旧 checkpoint 上继续以为已经筛掉旧点；推荐新建 inpaint run，旧产物不自动删除或改写。

#### 可选 depth 与 normal 如何参与

启用 depth 后，三角化点位用同场景尺度的 completed plane z-depth 作为软先验精修；匹配覆盖不足处可由 depth 反投影增加候选。这一步属于 EDGS 初始化器内部，不要求运行 Inpaint360GS RGB finetune，也不依赖现有 `FUSION_SEED_FRAME`。多视图候选先做可见性检查、去重和局部密度预算；不能直接叠加 N 张点云，也不能把互相矛盾的表面平均成中间一层。预算可依据周边可见表面间距与洞内面积确定，不以手调密度比例作为统一质量门槛。

启用 normal 后，对可信候选关联 `normal/*.npy` 和 validity，将相机法线转换到世界坐标，统一符号后做鲁棒聚合，用于 Gaussian 的最短尺度轴定向；切向尺度来自局部表面间距，法向厚度保持正值。对于 camera-to-world 的线性部分 $A$：

$$
n_w=\operatorname{normalize}(A^{-T}n_c).
$$

相机只含刚性旋转时等价于旋转法线；带 PCA 相似尺度时必须正确归一化。不能把平移加到法线上，不能直接平均不同相机坐标系下的向量。normal 不提供三维位置，不单独“反投影成点”；写入 PLY 的 `nx,ny,nz` 也不能代替实际 quaternion/scale 初始化。关闭 normal 时，不读取 completed normal，也不偷偷用 depth-derived normal 替代这个辅助开关。启用但缺少整路有效输入时预检失败；少量点缺少可信 normal 可保留未定向初值并记录比例。

#### PGSR 联合训练

训练从 `removed 背景 + 新 Gaussians` 开始，整段使用 PGSR 可微 renderer，同时输出 RGB、plane-depth、normal、alpha；不运行现有 Stage 5a/5b。新增点的 XYZ、rotation、scale、SH 和 opacity 可优化；首批背景、语义 embedding 与 classifier 冻结，完整场景一起渲染以保留遮挡。首批固定点数，不启用 clone/split/prune 或 opacity reset；任何拓扑更新须另立局部契约，不能继承全局调度。

令 $M$ 为 hole mask，$I_0$ 为同相机下冻结 removed 场景的 PGSR 基线：

$$
L=L_{RGB}(M)+\lambda_K L_{preserve}(\bar M,I_0)+\lambda_A L_{coverage}(M)
  +a_D(t)\lambda_D L_D+a_N(t)\lambda_N L_{LaMaN}
  +a_G(t)\bigl(\lambda_C L_{D\leftrightarrow N}+\lambda_M L_{MV}\bigr).
$$

RGB 项拟合 completed RGB；洞外保持项保护原场景；覆盖项避免靠透明化、露洞降低损失。depth loss 为有效正深度上的鲁棒 log-depth 残差；LaMa normal loss 在同一相机坐标系单位向量之间用有符号 $1-\langle\hat n,\hat n^*\rangle$，统一朝向后再比较。后两项分别约束模型自身深度/法线一致性和可见对应处的多视角几何，不能把内部几何输出当作外部真值。

外部 depth/normal loss 由训练开关决定，关闭时不要求相应目标或借用它构造权重；normal-only 外部监督不得依赖 completed depth 才能形成有效域。RGB/覆盖保护从开始生效，几何项按独立局部步数渐增，不继承全局 7000 步门槛。LaMa 的三种输出都是伪目标，分别计算停止梯度的可信度；`normal_valid` 不等于置信度，洞内低 removed alpha 也不能否决应补全的目标。

初始化入口是 `submodules/EDGS/tools/initialize_paintmesh_edgs.py`，联合训练入口是 `submodules/EDGS/tools/train_paintmesh_pgsr.py`。共享 CPU 契约位于 `scripts/paintmesh/edgs_inpaint_io.py`，新增点全参数模型、loss 与 debug 位于 `source/paintmesh_joint_*.py`。匹配器使用 EDGS vendored RoMa，三角化复用其 `correspondence/geometry.py`；不调用全局 Trainer。发布器和最终提交按 producer 区分两条路径。

当前实现采用空间/朝向近邻选对、双向匹配和两视图 DLT；跨图对在世界网格中保留高可信候选，不是完整的多视图轨迹 BA。depth 补点采用表面面积预算和近邻视图的软一致性证据；normal 对匹配的两次观测做世界坐标符号对齐聚合。训练的目标权重是各模态内部连续性的冻结启发式权重，不是经过标定的置信度，仍需检查伪目标冲突、分层与重影。

已通过合成初始化、可选模态隔离、真实 PGSR 短训练/恢复与发布分流测试；只读验证 kitchen 90 帧数据契约，并在真实第 0/1 帧验证 RoMa 匹配和三角化。未执行完整 kitchen 初始化及 5000 步联合训练，不承诺密度或法线质量已提升。参数、运行命令及后续消融见 [plan.md 的同级 EDGS-PGSR 路径](plan.md#edgs-pgsr-direct-plan)。

### 3.10 模块 I：发布 EDGS 可加载的 inpainted 3DGS

实现位于 [`publish_inpainted_edgs_model.py`](../../submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py)。

#### 输入

- 工作区 inpainted PLY；
- classifier；
- EDGS `config.yaml` 和 `cfg_args`；
- removal、tracker、camera、LaMa、fusion 等 manifests。

输入 PLY 按第 3.9a 节的局部优化开关选择，开启时另验 `local_geometry_manifest.json`；关闭时保留现有身份，不要求新字段。发布器不能忽略已请求但未完成的局部优化。

以上是原路径发布输入。发布器现已按所选 producer 验证来源：第 3.9b 节的新路径使用 `edgs_init_manifest.json` / `edgs_joint_manifest.json`，不要求原路径的 fusion、seed、RGB finetune 或 local geometry receipt；保留 PLY 语义字段、classifier 和模型/mesh 来源一致性的共同检查。

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

后续若开启局部几何优化，最终提交还必须追溯 `5a -> local_geometry -> published model` 的身份链，并确认 mesh 来源于相同的已发布 PLY；不能把几何优化前的 mesh 与优化后的模型组合为一个成功结果。

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
4. LaMa depth 是把深度当图像做补全，补全器本身不施加三维多视图一致性；默认路径主要依赖 RGB-D 初值与 RGB finetune，显式几何约束由可选 Stage 5b 或同级新路径承担。
5. 默认路径只用一个 `FUSION_SEED_FRAME` PLY 初始化 splats，未把 N 个支持 PLY 联合配准或融合成一个初始化点云；第 3.9b 节的多视角候选初始化是另一条可选路径。
6. Stage 5a RGB finetune 没有 depth/normal loss，也没有 semantic loss；只有显式开启的 Stage 5b 使用局部 depth、LaMa normal 与内部一致性监督。
7. 补全 mesh 的细节上限受 PGSR plane depth、真实训练相机覆盖、TSDF voxel size 和连通分量过滤共同限制。
8. `semantic_mesh.ply` 是便于查看的 vertex 着色副本；需要精确 face 语义时应读取 `.npy` sidecar 和 manifest。
9. 自动 normal completion 与局部优化是两种独立行为：前者由上游模态触发，后者由默认关闭的 `LOCAL_GEOMETRY_REFINE` 显式启用。LaMa normal 不是几何真值，normal loss 也不等于增密。
10. 所有局部几何参数只属于当前 inpaint run，禁止写回最初的全局训练配置或模型；第 1 节全局 `step > 7000` 调度保持不变。
11. 同级 `edgs-pgsr` 路径及 `EDGS_INIT_USE_*` / `EDGS_TRAIN_USE_*` 已接通，但完整场景质量仍待评估。初始化或监督关闭不等于跳过公共 completion；RGB-only 不保证弱纹理覆盖，多种 LaMa 伪目标联合使用也不保证真实隐藏几何。

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
| 自动 normal completion | [`predict_normal.py`](../../submodules/Inpaint360GS/LaMa/bin/predict_normal.py)、[`paintmesh_normal.py`](../../submodules/Inpaint360GS/tools/paintmesh_normal.py) |
| RGB-D 反投影 | [`edit_object_removal_plyfusion.py`](../../submodules/Inpaint360GS/edit_object_removal_plyfusion.py) |
| 3DGS inpaint | [`edit_object_inpaint.py`](../../submodules/Inpaint360GS/edit_object_inpaint.py)、[`compose_utils.py`](../../submodules/Inpaint360GS/utils/compose_utils.py) |
| EDGS 模型发布 | [`publish_inpainted_edgs_model.py`](../../submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py) |
| 最终一致性验证 | [`finalize_inpaint_result.py`](../../submodules/Inpaint360GS/tools/finalize_inpaint_result.py) |
