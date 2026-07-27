
这条命令实际是一个三阶段流水线：训练 30,000 次迭代 → 从高斯提取语义网格 → 优化网格顶点颜色。任一步子进程返回非 0，后续步骤都会停止。[入口脚本](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/scripts/train_and_extract_spf.py:70)

```text
COLMAP 图像 + 相机 + EntitySeg masks
              │
              ▼
  联合训练 RGB Gaussian + 16D 语义特征 + SPF
              │
              ▼
   Pivot + Delaunay + Marching Tetrahedra
              │
              ▼
      连通域后处理 + 顶点颜色优化
```

## 1. 入口脚本展开参数

你的命令会被展开成近似下面三个子命令。

训练：

```bash
python train.py \
  --rasterizer ours \
  --feature_dc_lr 0.0013 \
  --feature_rest_lr 0.00011 \
  --exposure_compensation \
  --data_device cpu \
  --N_max_gaussians 6000000 \
  --semantic_prior \
  -s data/mip-nerf/360_v2/counter \
  -m outputs/semantic_prior_field/counter_full \
  --semantic_masks data/mip-nerf/360_v2/counter/entityseg_mask_robust_v2 \
  -r 4
```

网格提取：

```bash
python pivot_based_mesh_extraction.py \
  --rasterizer ours \
  --sdf_mode ours \
  --dtype int32 \
  --isosurface_value 0 \
  --n_binary_steps 10 \
  --iteration 30000 \
  --use_valid_mask \
  --postprocess \
  --filter_large_edges \
  --use_semantics \
  --data_device cpu \
  -s data/mip-nerf/360_v2/counter \
  -m outputs/semantic_prior_field/counter_full \
  -r 4
```

纹理优化：

```bash
python texture_mesh.py \
  --rasterizer ours \
  --mesh outputs/semantic_prior_field/counter_full/mesh_ours_2pivots_post.ply \
  -s data/mip-nerf/360_v2/counter \
  -m outputs/semantic_prior_field/counter_full \
  -r 4
```

`-r 4` 表示图像宽高都缩小到原来的 1/4，不是缩放到 4 像素。语义 mask 会用最近邻插值缩放到相同分辨率。[camera_utils.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/utils/camera_utils.py:21)

## 2. 训练初始化

训练入口在 [train.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/train.py:1185)。

首先根据 `source_path/sparse` 判定这是 COLMAP 数据集，读取：

- `sparse/images.bin`、`cameras.bin`、`points3D.bin`
- `images/` 中的训练图像
- COLMAP 相机内外参和初始稀疏点云

由于没有 `--eval`，所有相机都会进入训练集。[dataset_readers.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/scene/dataset_readers.py:132)

随后创建 GaussianModel，主要包含：

- 3D 位置、尺度、旋转、不透明度
- RGB 球谐系数，最高 SH degree 3
- 4 维可学习 normal-field 特征
- 每个 Gaussian 的 16 维语义 embedding
- 每张训练图像的曝光补偿参数
- 3D Mip filter
- 最多 600 万个 Gaussian

语义 mask 目录会被 `GagaObservationStore` 检查。默认要求每个训练相机都能找到对应 mask；类别数优先从 `info.json` 的 `num_mask/num_instances` 得到，否则扫描所有 mask 推断。[observations.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/observations.py:29)

然后建立一个 `Conv2d(16, num_classes, kernel_size=1)` 语义分类头，将渲染出的 16D 特征图转换为实例类别 logits。[head.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/head.py:9)

## 3. 每次训练迭代

默认运行 30,000 次。每次迭代执行：

1. 从尚未遍历的训练相机中随机取一张。
2. 使用 `ours` rasterizer 渲染：
   - RGB
   - 16D 语义特征图
   - 可见 Gaussian、屏幕半径
   - 激活几何正则后还会渲染深度和法线
3. 计算 RGB 损失：

```text
0.8 × L1 + 0.2 × (1 - SSIM)
```

4. 读取当前相机的 EntitySeg mask。
5. 将 16D 语义特征图送入分类头，计算置信度加权的语义交叉熵。
6. 加入当前阶段已经激活的几何/SPF损失。
7. `loss.backward()`。
8. 更新 Gaussian 参数、语义 embedding、语义分类头和曝光补偿。
9. 按计划 densify、split、prune 或重置 opacity。

核心循环见 [train.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/train.py:324)。

## 4. SPF 的时间流程

| 迭代区间 | 主要行为 |
|---|---|
| 1–6999 | RGB + 语义 CE 联合训练；普通 Gaussian densification 从 600 开始，每 100 次一次 |
| 7000 起 | 深度-法线正则、快速多视图正则、SPF 开始工作 |
| 7000–14900 | SPF 根据实例类型重新分配普通 densification 预算 |
| 15000 | 普通 densification 结束；第一次 identity-stability pruning |
| 20000 | 第二次 identity-stability pruning |
| 20001 起 | Normal Field 和 SPF orientation prior 激活 |
| 日志迭代 22000–26000 | 每 1000 次执行 normal-field densification |
| 日志迭代 22500–26500 | 每 1000 次执行 semantic-boundary splitting |
| 30000 | 保存最终 Gaussian 和语义 checkpoint |

SPF 刷新时会：

1. 用语义分类头直接分类每个 Gaussian 的 16D embedding。
2. 用 top-1 与 top-2 概率差作为置信度。
3. 置信度低于 `0.3` 的 Gaussian 标为未知。
4. 跳过背景和少于 512 个 Gaussian 的小实例。
5. 对实例拟合：
   - thin
   - RANSAC plane
   - quadric
   - 拟合失败则 freeform
6. 生成 proxy normal、prior weight 和 densification multiplier。

实现位于 [prior_field.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/prior_field.py:203)。

这些结果用于：

- 平面/曲面实例选择性压扁
- 实例内部 SH 一致性
- 高阶 SH 异常值抑制
- 学习法线与实例 proxy normal 对齐
- 实例边界附近降低深度-法线和多视图平滑权重
- 平面减少 densification、细长结构增加 densification
- 分裂语义冲突最高的约 5% Gaussian
- 删除“语义身份不稳定且从未成为像素主贡献者”的 floaters

最终保存：

```text
outputs/semantic_prior_field/counter_full/
├── point_cloud/iteration_30000/point_cloud.ply
├── semantic/semantic_chkpnt30000.pth
├── semantic/semantic_metadata.json
├── diagnostics/
├── cfg_args
├── cameras.json
├── input.ply
└── time.txt
```

## 5. 网格提取

训练成功后，从 `iteration_30000/point_cloud.ply` 重新加载 Gaussian。

每个 Gaussian 默认生成两个 pivot：

- Gaussian 中心
- 沿学习法线方向偏移 `3 × 法线方向标准差` 的点

代码见 [pivots.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/extraction/pivots.py:10)。

之后：

1. 对每个 pivot，在所有训练视角中积分 Gaussian alpha。
2. 构造：

```text
SDF = 0.5 - integrated_alpha
```

因此 `isosurface_value=0` 对应约 `alpha=0.5` 的表面。

3. 将任何相机都看不到的 pivot 的 SDF 设置为 `0.5`。
4. 加载 `semantic_chkpnt30000.pth`，为 pivot 继承所属 Gaussian 的实例标签和置信度。
5. 对所有 pivots 做 Delaunay tetrahedralization。
6. 用 Marching Tetrahedra 提取零等值面。
7. 对交点执行 10 次二分搜索，提高表面定位精度。
8. 融合多视图颜色，生成初始顶点颜色。
9. 只保留最大的连通网格，清除 floaters。

入口见 [pivot_based_mesh_extraction.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/pivot_based_mesh_extraction.py:561)。

产生：

```text
mesh_ours_2pivots.ply
mesh_ours_2pivots.semantic.npz
mesh_ours_2pivots_semantic.ply
mesh_ours_2pivots_post.ply
```

一个容易误解的细节：日志会打印“semantic edge filtering”，但当前包装脚本没有加入 `--filter_semantic_edges`，所以实际不会因为两个 pivot 属于不同实例而删除三角面；这里只导出了顶点语义。[train_and_extract_spf.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/scripts/train_and_extract_spf.py:37)

另外，`.semantic.npz` 对应的是后处理前的原始网格；连通域后处理会改变顶点索引，因此它不能直接按索引套到 `_post.ply` 上。

## 6. 纹理优化

最后加载 `mesh_ours_2pivots_post.ply`。

它先用训练后的 Gaussian、但仅使用 SH degree 0 渲染全部训练视角，然后把网格顶点颜色设为可学习参数，运行 1000 次：

```text
0.8 × L1(mesh_render, gaussian_render)
+ 0.2 × (1 - SSIM)
```

这里只优化顶点 RGB，不优化顶点位置和三角面。[texture_mesh.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/texture_mesh.py:98)

最终文件名是：

```text
outputs/semantic_prior_field/counter_full/
mesh_ours_2pivots_post_texture_refined_999.ply
```

之所以是 `_999` 而不是 `_1000`，是因为循环变量从 `0` 到 `999`，保存文件名直接使用了最后一个循环索引。

最关键的结论是：

> 你指定了 `--rasterizer ours`，但在“RGB + 16D语义”的主训练渲染中，实际优先调用的是 `diff_gaussian_rasterization_spf`。它是 Ours median-depth rasterizer 的语义扩展版，额外渲染 16D 特征并在反向传播时统计语义冲突。

SPF 本身不是第三个可训练网络，也不是被 rasterizer 渲染出来的体场。真正联合优化的是：

- RGB Gaussian 的几何和外观参数
- 每个 Gaussian 的 16D 语义 embedding
- 场景级 `Conv1×1(16,C)` 语义分类头
- 4D normal-field 特征
- 曝光补偿参数

SPF 则是从当前 16D embedding 定期推导出来的、无梯度更新的临时先验场，然后通过显式损失和 densify/split/prune 操作反过来指导 Gaussian。

## 1. 实际使用了哪几个 rasterizer

你的参数首先让训练脚本导入：

```python
from gaussian_renderer.ours import render_ours as render
```

见 [train.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/train.py:1393)。

但 `render_ours()` 内部还会根据是否渲染语义，选择不同的 CUDA 扩展。

| 使用场景 | 实际 CUDA 后端 | 输出 |
|---|---|---|
| 普通 RGB/深度辅助渲染 | `diff_gaussian_rasterization_ours` | RGB、alpha、median depth、normal |
| RGB + 语义，但不用统计 | `diff_gaussian_rasterization_gw_ours_semantic` | 上述输出 + 16D 特征图 |
| 你的主训练渲染 | `diff_gaussian_rasterization_spf` | 上述输出 + 16D 特征图 + 每 Gaussian 语义统计 |

因为你的命令满足：

```text
semantic_masks != None
semantic_prior = True
sp_stats = True（默认）
rasterizer = ours
```

所以训练创建 `SemanticStatsAccumulator`，每次主渲染都会传入 `semantic_stats_sink={}`。[train.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/train.py:153)

`render_ours()` 检测到这个 sink 后请求 `"spf"` backend：

```python
requested_backend = "spf" if semantic_stats_sink is not None else "ours"
```

见 [ours.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/gaussian_renderer/ours.py:43)。

最终加载的原生包是：

```text
diff_gaussian_rasterization_spf
```

见 [semantic_runtime.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/gaussian_renderer/semantic_runtime.py:12)。

如果 SPF CUDA 扩展没有安装，它会打印：

```text
[WARNING] SPF rasterizer unavailable; falling back to the
ours semantic rasterizer (no stats channel).
```

然后退回 `diff_gaussian_rasterization_gw_ours_semantic`。这种情况下 RGB 和语义仍然能训练，只是语义边界 splitting 失去实时 conflict statistics，需要回退到全相机扫描。

## 2. Ours rasterizer 是什么

它是一个基于 CUDA 的、可微分的 tile-based 3D Gaussian splatting rasterizer，修改自原始 `diff-gaussian-rasterization`，增加了：

- alpha 输出
- median depth 输出
- normal 输出
- 对这些几何输出的反向传播

每个 3D Gaussian 包含：

```text
位置             xyz_i       ∈ R³
尺度             s_i         ∈ R³
旋转             q_i
不透明度         o_i
球谐颜色         SH_i
语义 embedding   e_i         ∈ R¹⁶
normal feature   nfeat_i     ∈ R⁴
```

### 投影

对于当前相机，rasterizer 把 3D Gaussian 投影成屏幕空间的椭圆 Gaussian：

```text
G_i(p) = exp(-½ Δpᵀ Σ'⁻¹_i Δp)
```

其中：

- `Σ'_i` 是投影后的二维协方差
- `Δp` 是像素到 Gaussian 中心的屏幕空间偏移

然后计算该 Gaussian 在像素上的 alpha：

```text
α_i(p) = min(0.99, opacity_i × G_i(p))
```

Gaussians 按深度排序并进行前向 alpha compositing。第 `i` 个 Gaussian 前面的透射率是：

```text
T_i(p) = ∏_{j<i} (1 - α_j(p))
```

最终贡献权重：

```text
w_i(p) = T_i(p) × α_i(p)
```

RGB、语义、normal 都共享这个 `w_i(p)`。

## 3. RGB 是怎么渲染的

每个 Gaussian 的颜色来自球谐函数：

```text
c_i(v) = SH_i(view_direction)
```

然后：

```text
RGB(p) = Σ_i w_i(p)c_i(v) + T_final(p) × background
```

对应 CUDA 代码中的：

```cpp
C[ch] += feature_i[ch] * alpha * T;
```

见 [render_forward.cu](/home/martin/code/gsagent/submodules/SemanticPriorField/submodules/diff-gaussian-rasterization-spf/cuda_rasterizer/render_forward.cu:504)。

训练初期只使用 SH degree 0，之后：

```text
iteration 1000 → degree 1
iteration 2000 → degree 2
iteration 3000 → degree 3
```

你的命令还启用了曝光补偿。每个相机学习两个参数 `a_v,b_v`：

```text
RGB'_v = exp(a_v) × RGB_v + b_v
```

L1 使用补偿后的图像，SSIM 使用原始渲染图像。[loss_utils.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/utils/loss_utils.py:123)

RGB 损失为：

```text
L_rgb = 0.8 L1(RGB', GT) + 0.2 (1 - SSIM(RGB, GT))
```

## 4. Ours 如何渲染 median depth 和 normal

Ours 不是简单的期望深度 rasterizer。它沿每条像素光线寻找累计透射率下降到 `0.5` 附近的位置，即近似的中值表面：

```text
T(depth_median) ≈ 0.5
```

CUDA 中先得到初始深度区间，再进行多轮细分：

- 每轮分成 8 段
- 训练模式进行 5 轮细化

见 [config.h](/home/martin/code/gsagent/submodules/SemanticPriorField/submodules/diff-gaussian-rasterization-spf/cuda_rasterizer/config.h:15)。

这就是所谓的 Ours “median-depth rasterizer”。

对于 normal，rasterizer从 Gaussian 的尺度、旋转和 ray-plane 几何推导每个 Gaussian 的表面法线，然后以相同的 alpha-transmittance 权重合成：

```text
N(p) = normalize(Σ_i w_i(p)N_i)
```

注意这里的 rasterizer normal，和额外的 4D learnable normal-field feature 不是同一个东西：

- rasterizer normal：由 Gaussian 几何形状、旋转、尺度计算
- normal-field normal：由每 Gaussian 的 4D可学习特征转化而来，主要在 20,001 迭代后使用

在 Python 包装中，Ours 只有一个 median depth，所以：

```python
expected_depth = rendered_median_depth
median_depth   = rendered_median_depth
```

见 [ours.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/gaussian_renderer/ours.py:143)。

## 5. 16D 语义特征怎么渲染

初始化时，每个 Gaussian 会创建一个小随机向量：

```text
e_i ∈ R¹⁶
e_i ~ Normal(0, 0.01)
```

见 [gaussian_model.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/scene/gaussian_model.py:770)。

语义渲染并不是直接渲染类别 ID，也不是先对每个 Gaussian 做分类。它先像 RGB 一样混合 16D embedding：

```text
E(p) = Σ_i w_i(p)e_i
     = Σ_i T_i(p)α_i(p)e_i
```

RGB 和语义严格共享：

- Gaussian 投影位置
- 屏幕椭圆 footprint
- 深度顺序
- opacity
- 前向透射率
- early termination

CUDA 实现位于 [semantic.cu](/home/martin/code/gsagent/submodules/SemanticPriorField/submodules/diff-gaussian-rasterization-spf/cuda_rasterizer/semantic.cu:18)。

语义没有像 RGB 那样显式添加背景向量，因此没有 Gaussian 覆盖时：

```text
E(p) = 0
```

之后才将 `[16,H,W]` 特征图送入场景级分类头：

```text
logits(p) = W E(p) + b
```

这个分类头本质上是：

```python
Conv2d(16, num_classes, kernel_size=1)
```

所以它是逐像素线性分类器，不包含空间卷积。[head.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/head.py:9)

最后用 EntitySeg/Gaga mask 监督：

```text
L_sem = CE(logits(p), mask_label(p))
```

如果 mask 有 confidence 图，则每个像素的 CE 会乘相应置信度；ignore 像素不参与。损失还除以 `max(log(C),1)`，减小类别数不同造成的尺度差异。[losses.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/losses.py:11)

你的默认权重是：

```text
lambda_semantic = 1.0
balance_semantic = False
lambda_semantic_3d = 0
```

所以每轮基础总损失是：

```text
L = L_rgb + L_sem + 当前激活的几何/SPF损失
```

## 6. 最重要的梯度隔离策略

虽然 RGB 和语义共享同一套可见性权重，但当前实现刻意规定：

> 原始语义 CE 只更新 16D embedding 和分类头，不直接更新 Gaussian 的位置、尺度、旋转和 opacity。

语义 CUDA backward 为每个 Gaussian 计算：

```text
∂L_sem/∂e_i = Σ_p w_i(p) ∂L_sem/∂E(p)
```

CUDA 中：

```cpp
atomicAdd(
    &grad_semantic_features[i, channel],
    weight * pixel_gradient[channel]
);
```

见 [semantic.cu](/home/martin/code/gsagent/submodules/SemanticPriorField/submodules/diff-gaussian-rasterization-spf/cuda_rasterizer/semantic.cu:191)。

但该语义辅助 pass 不会计算：

```text
∂L_sem/∂xyz
∂L_sem/∂scale
∂L_sem/∂rotation
∂L_sem/∂opacity
```

因此梯度关系是：

| 损失 | 直接更新的主要参数 |
|---|---|
| RGB L1/SSIM | xyz、scale、rotation、opacity、SH、曝光参数 |
| 语义 CE | 16D semantic embedding、1×1 classifier |
| 深度-normal | xyz、scale、rotation、opacity |
| multiview | Gaussian 几何和外观 |
| SPF selective flatten | scale |
| SPF SH consistency/decay | 高阶 SH |
| SPF orientation | 4D normal-field feature |
| SPF budget/split/prune | 改变 Gaussian 数量和空间分布 |

所以这里的“联合训练”不是让语义 CE 无约束地拉动几何，而是：

```text
mask
  → 训练 embedding
  → embedding 形成稳定实例身份
  → SPF 将身份转成可解释的几何先验
  → 先验通过显式损失和 density control 修改几何
```

这种设计避免噪声 mask 直接通过 opacity/ellipse 梯度破坏重建几何。

## 7. SPF rasterizer 比 Ours-semantic 多了什么

SPF rasterizer 的前向结果与 Ours-semantic 相同。区别主要在语义 backward 中额外输出每个 Gaussian 的统计量。

对 Gaussian `i`，它累积：

```text
unsigned_grad_i =
    Σ_p w_i(p) ||∂L/∂E(p)||

contribution_i =
    Σ_p w_i(p)

signed_grad_i =
    ∂L/∂e_i
```

然后形成 conflict score：

```text
conflict_i =
    (unsigned_grad_i - ||signed_grad_i||)
    / contribution_i
```

直观上：

- 如果一个 Gaussian 覆盖的像素都要求它属于同一实例，语义梯度方向一致，conflict 低。
- 如果它跨越两个实例边界，一部分像素把 embedding 往类别 A 拉，另一部分往类别 B 拉。
- 有符号梯度会相互抵消，但 unsigned gradient 不会。
- 因而边界跨越 Gaussian 的 conflict 很高。

统计实现见 [scatter.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/scatter.py:32)。

这些统计在正常训练的 backward 中顺便计算，不需要为语义 splitting 单独再次渲染全部相机。

## 8. SPF 场如何从 embedding 产生

从 7000 次迭代开始，代码定期刷新 SPF。刷新过程在 `torch.no_grad()` 下执行：

```text
每 Gaussian 16D embedding
        │
        ▼
1×1 head 的线性权重
        │
        ▼
每 Gaussian 类别概率
        │
        ├─ top-1 → label
        └─ top1 - top2 → confidence
```

置信度低于 `0.3` 的 Gaussian 被标为 `-1`，表示身份不确定。

然后按实例收集 Gaussian 中心，拟合几何代理：

```text
thin heuristic
    ↓ 未命中
RANSAC plane
    ↓ 拟合差
quadric
    ↓ 拟合差
freeform
```

得到的 SPF 数据包括：

```text
labels[N]
label_confidence[N]
prior_type[N]
prior_normals[N,3]
prior_weight[N]
densify_multiplier[N]
```

SPF 自己不进入 rasterizer，也不作为神经网络参数更新；它只是当前 Gaussian 状态的一份可刷新解释。[prior_field.py](/home/martin/code/gsagent/submodules/SemanticPriorField/semantic_prior_field/semantic/prior_field.py:203)

## 9. 一次完整迭代的数据流

```text
随机训练相机
    │
    ▼
GaussianModel
 xyz / scale / rotation / opacity / SH / semantic[16]
    │
    ▼
diff_gaussian_rasterization_spf
    ├─ RGB [3,H,W]
    ├─ semantic field [16,H,W]
    ├─ alpha [1,H,W]
    ├─ median depth [1,H,W]
    ├─ normal [3,H,W]
    ├─ visibility/radii
    └─ backward semantic statistics
         │
         ├───────────────┐
         ▼               ▼
 RGB GT图像       Conv1×1(16,C)
         │               │
   L1 + SSIM       EntitySeg mask CE
         │               │
         └───────┬───────┘
                 ▼
          加上几何/SPF损失
                 │
                 ▼
             loss.backward()
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 RGB/几何参数更新       embedding/head更新
       │                    │
       └─────────┬──────────┘
                 ▼
       SPF刷新 / densify / split / prune
```

因此，准确描述这条命令的训练方式应该是：

> 使用 Ours median-depth CUDA rasterizer 的 SPF 语义扩展，在同一套 Gaussian 可见性和 alpha compositing 下同时渲染 RGB 与 16D实例特征；语义 CE 只训练 embedding 和分类头，SPF 再将语义身份显式转化为几何正则、密度预算、边界分裂和不稳定 Gaussian 剪枝。
