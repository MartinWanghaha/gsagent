# Gaussian Wrapping × Gaga 原生语义融合设计

## 1. 结论

本项目应以 `submodules/GaussianWrapping` 当前版本为唯一上游基线，建立一个保留上游提交历史的新 fork，例如：

```text
submodules/GaussianWrappingGaga/
```

不要以 `submodules/Gaga` 或现有 `submodules/SemanticGaussianWrapping` 为基线重写。正确的融合边界是：

1. Gaussian Wrapping 的训练、RaDeGS/Ours 光栅化、深度/法线、SDF/积分、网格提取、纹理、评测和 Blender 工具全部原样保留。
2. 在 Gaussian Wrapping 的 RaDeGS 与 Ours 两套现有 CUDA 光栅器内部，增加一个通用的 16 维 per-Gaussian auxiliary feature 通道。
3. 在高斯模型中增加独立的 16 维语义嵌入及其完整生命周期管理。
4. 复用 Gaga 的“关联掩码 → 16 维高斯嵌入 → 1×1 分类头 → 像素分类”方法，但不以 Gaga 的旧版 3DGS 光栅器替换 Gaussian Wrapping 光栅器。
5. 默认使用后训练 `semantic lift`：加载已经训练好的 Gaussian Wrapping，高斯几何和外观完全冻结，只优化语义嵌入与分类头。这是“继承全部功能”最强、最容易验证的模式。

语义融合不是一个临时分支或渲染后补丁，而是一套可扩展的“高斯辅助属性框架”。第一种属性是 Gaga 语义嵌入，后续还可挂接实例、材质、动态性或不确定性特征。

## 2. 代码审阅得到的事实

### 2.1 Gaussian Wrapping 不能被 Gaga 光栅器替换

Gaussian Wrapping 有两套主训练后端：

- `gaussian_wrapping/gaussian_renderer/radegs.py`
- `gaussian_wrapping/gaussian_renderer/ours.py`

RaDeGS 返回 RGB、期望/中值坐标、期望/中值深度、alpha 和法线，并提供点积分接口。Ours 返回 RGB、中值深度、alpha 和法线，也有自己的积分/SDF 路径。网格提取依赖这些专有输出。

Gaga 的 CUDA 光栅器来自更普通的 3DGS 分支，只在 RGB 合成旁增加 `NUM_OBJECTS=16` 的对象特征合成。若直接用它覆盖 Gaussian Wrapping 光栅器，将丢失上述深度、法线、坐标和积分能力。

因此必须：

- 从 Gaussian Wrapping 当前 RaDeGS/Ours CUDA 源码出发增加语义通道；
- 不能把 Gaga 的 rasterizer 目录整体复制覆盖；
- 不能用第二次 RGB 光栅化冒充语义光栅化，因为那会重复排序/混合、增加开销，并可能与主渲染贡献列表不一致。

### 2.2 Gaga 的核心语义机制

Gaga 为每个高斯维护 16 维 `_objects_dc`，光栅化时按与颜色相同的前向 alpha 合成：

```text
E(p) = Σ_i T_i(p) · α_i(p) · e_i
```

其中 `e_i ∈ R^16`。语义背景为零，不再加 `T_final × background_feature`。随后使用场景级 1×1 卷积：

```text
logits(p) = W · E(p) + b
```

并以关联掩码监督像素交叉熵。类别数是场景动态值，因此 CUDA 中只固定较小的 16 维嵌入，不应把类别数编译进 kernel。

Gaga 的 backward 同时把语义梯度累加给嵌入和 alpha/几何。但其 lift 训练冻结几何，所以实际只更新嵌入。新框架应把这个隐含行为升级为明确的梯度策略。

### 2.3 直接用 Gaga 加载/保存会破坏 Gaussian Wrapping 模型

实际产物已经显示：

- Gaussian Wrapping PLY 含 `filter_3D` 与 `gaussian_features_*`；
- 经现有 Gaga lift 保存的 PLY 含 `obj_dc_0..15`；
- 但 Gaga 保存结果丢失了 `filter_3D` 与 `gaussian_features_*`。

这意味着现有“先用 Gaussian Wrapping 训练，再交给 Gaga 模型重写 PLY”的流程不满足完整继承。语义字段必须由 Gaussian Wrapping 自己加载、保存、裁剪和致密化。

### 2.4 现有 SemanticGaussianWrapping 不适合作为基线

`submodules/SemanticGaussianWrapping` 是一套有价值的实验实现，包含属性注册、Gaga 观测适配和语义测试等思路；但它是平行重实现，并未继承 Gaussian Wrapping 的完整文件、API 和工具链。对外公开符号、原始 renderer、regularizer、mesh extraction、纹理与评测工具均存在较大缺口。

可选择性迁移它的以下思想：

- 高斯属性注册和原子生命周期变换；
- Gaga mask/confidence/boundary 观测适配；
- 语义梯度隔离测试。

不应覆盖或改造成目标基线，也不应覆盖该目录中现有的用户工作。

## 3. 总体架构

```text
Gaga 数据准备
  raw masks → cross-view association → associated masks + info.json
                                      │
                                      ▼
Gaussian WrappingGaga
  Camera/Observation Adapter ──→ semantic target [H,W]
                                      │
GaussianModel ── semantic embedding [N,16]
      │                               │
      ├── original GW fields          │
      │   xyz/SH/opacity/scale/...     │
      │   filter_3D/occupancy/normal   │
      │                               ▼
      └────────────────────→ RaDeGS-semantic CUDA
                         or Ours-semantic CUDA
                                      │
                     RGB/depth/normal/alpha unchanged
                     semantic_features [16,H,W]
                                      │
                                      ▼
                         scene semantic head 16→C
                                      │
                                      ▼
                              logits / labels / metrics
```

框架分为四层：

1. **数据层**：读取 Gaga 已关联掩码，确保其和 Gaussian Wrapping 最终相机分辨率严格对齐。
2. **属性层**：管理每个高斯的语义嵌入、优化器状态、PLY/checkpoint 和致密化生命周期。
3. **光栅层**：在两套原生 CUDA renderer 的同一次前后向传播中合成语义特征。
4. **任务层**：分类头、损失、训练模式、渲染导出和评测。

## 4. CUDA 设计

### 4.1 公共接口

两套语义扩展共享同一个逻辑契约：

```text
input:
  aux_features: contiguous float32 [N, D]
  D: build-time specialization, default 16
  aux_gradient_policy: EMBEDDING_ONLY | FULL

output:
  aux_image: float32 [D, H, W]

backward:
  grad_aux_features: float32 [N, D]
```

建议使用独立 Python/extension 包名，例如：

```text
diff_gaussian_rasterization_gw_semantic
diff_gaussian_rasterization_gw_ours_semantic
```

这样原始扩展可以共存，旧脚本不会因为 ABI 或返回 tuple 变化而意外加载新模块。

### 4.2 Forward

语义通道必须复用原 kernel 的：

- tile 分桶与排序结果；
- contributor 顺序；
- alpha 截断；
- early termination；
- `T` 的更新；
- 可见性和半径判定。

对每个通过原始可见性判断的高斯：

```cpp
for (int ch = 0; ch < AUX_DIM; ++ch) {
    aux_accum[ch] += aux_feature[gaussian_id * AUX_DIM + ch] * alpha * T;
}
```

最后直接写出 `aux_accum`。默认保持 Gaga 的 premultiplied 语义定义，不除以 `1 - T`。若研究上需要 normalized feature，应作为 Python 后处理：

```text
E_normalized = E / clamp(alpha_image, eps)
```

不要在 CUDA 默认路径中改变 Gaga 的数值语义。

### 4.3 Backward 与梯度路由

所有模式都计算：

```text
∂L/∂e_i += T_i · α_i · ∂L/∂E
```

梯度策略：

- `EMBEDDING_ONLY`：默认。语义 loss 的 adjoint 不加入 `dL/dalpha`，因此不会改变坐标、尺度、旋转、opacity 或深度排序相关参数。
- `FULL`：研究选项。按 Gaga backward 把语义项加入 `dL/dalpha`，允许语义监督影响几何。

这两者应由不同模板或编译分支实现，避免每个像素/高斯贡献处出现运行时分支。

### 4.4 兼容模式

语义关闭时，必须直接调用原始 extension 和原始 kernel，而不是给语义 kernel 传空张量。这样可以：

- 保留原始 tuple、ABI 和数值结果；
- 避免额外寄存器/共享内存降低原有性能；
- 把“完整继承”转化为可测试的严格边界。

### 4.5 性能约束

16 维 × 256 threads × float32 的共享特征缓存约增加 16 KiB。需分别检查 RaDeGS 和 Ours kernel 的：

- 每 block 共享内存；
- registers/thread；
- achieved occupancy；
- frame time；
- forward/backward 显存峰值。

默认固定 16 维是合理折中。不要在 CUDA 中直接合成数百个类别通道；类别解码放在 PyTorch 的 1×1 头中。

## 5. 高斯属性框架

### 5.1 独立语义字段

新增：

```text
_semantic_embedding: nn.Parameter [N,16]
semantic_dim: 16
```

它必须独立于 Gaussian Wrapping 现有 `_gaussian_features`。后者用于法线场等功能，维度和语义均不同，不能复用。

对外提供：

```text
get_semantic_embedding
has_semantics
initialize_semantics(...)
```

### 5.2 生命周期注册

Gaussian Wrapping 当前在多个方法中手工处理可选字段。新增一个轻量 registry，集中描述每个 per-Gaussian 属性：

```text
name
tensor accessor
optimizer group
PLY prefix
checkpoint serializer
on_prune(mask)
on_clone(parent_indices)
on_split(parent_indices, repeat)
```

第一阶段不重构已有字段，只让语义字段经过 registry，以降低对上游代码的侵入；待测试充分后，再考虑让既有可选字段逐步迁移。

语义字段生命周期规则：

- 创建：随机小值或零初始化；为避免所有高斯完全对称，推荐与 Gaga 一致使用可复现的小随机初始化。
- prune：使用与 `_xyz` 完全相同的保留 mask。
- clone：精确复制父高斯嵌入。
- split：默认精确复制父嵌入；仅在明确实验配置下加入零均值微扰。
- optimizer：扩展/裁剪 Adam 状态时与参数同步，保证 `exp_avg`、`exp_avg_sq` 第一维始终等于高斯数量。
- assert：每次拓扑变化后检查全部 registered 属性的第一维均为 `N`。

### 5.3 PLY 与 checkpoint

为兼容 Gaga，PLY 使用：

```text
obj_dc_0 ... obj_dc_15
```

内部仍命名为 `semantic_embedding`，避免把“object”概念写死在框架里。

保存时只能“追加”语义字段，必须保留：

- `filter_3D`
- `base_occupancy_*`
- `occupancy_shift_*`
- `gaussian_features_*`
- Gaussian Wrapping 的全部标准字段

分类头不是 per-Gaussian PLY 属性，应单独保存：

```text
semantic/semantic_head.pth
semantic/metadata.json
```

metadata 至少包含：

```text
format_version
semantic_dim
num_classes
class/instance label mapping
background_id
ignore_id
mask source and association config
renderer backend
compositing mode
gradient policy
```

Gaussian checkpoint 则保留原有格式，并在版本化尾部追加语义状态；旧 checkpoint 必须仍能加载。

## 6. 数据与相机对齐

Gaga 的 SAM/EntitySeg 掩码生成和跨视角关联保持为独立预处理，不把 detectron2、SAM 等重依赖引入 Gaussian Wrapping 主环境。

新增观测适配器：

```text
semantic/observations.py
```

职责：

- 从 associated masks 和 `info.json` 读取标签；
- 建立图像名到 mask 的确定性映射；
- 在 Camera 最终分辨率上使用 nearest-neighbor resize；
- 以 `0` 表示背景；
- 以 `-1` 表示缺失或明确忽略的像素；
- 可选读取 confidence/boundary，但不要让第一版训练依赖这些扩展字段。

RGB 继续走现有双线性/面积缩放路径，语义标签绝不能走双线性插值。

启动时进行严格校验：

- 相机必须能唯一匹配 mask；
- 标签必须位于 `[-1, C-1]`；
- 尺寸必须与最终渲染分辨率一致；
- `num_classes` 必须与 `info.json` 一致。

## 7. 渲染 API

保留现有 renderer 的所有参数和返回键。语义开启时仅追加：

```python
{
    # 原有键保持不变
    "semantic_features": Tensor[16, H, W],
}
```

统一的 Python 配置：

```python
SemanticRenderOptions(
    enabled=True,
    gradient_policy="embedding_only",
    normalized=False,
)
```

分类与可视化不进入 renderer：

```python
features = render_pkg["semantic_features"]
logits = semantic_head(features[None])[0]
labels = logits.argmax(dim=0)
```

应确保 `override_color`、Python SH 转换和 CUDA SH 转换三种颜色路径都独立设置 `aux_features`，避免 Gaga 当前 wrapper 中 `override_color` 分支可能遗漏 `sh_objs` 的问题。

## 8. 训练模式

### 8.1 `semantic=off`

完全原始的 Gaussian Wrapping：

- 不创建语义参数；
- 不加载语义扩展；
- 不改变 optimizer；
- 不改变 renderer 返回；
- 训练、渲染、网格和评测结果与上游基线一致。

### 8.2 `semantic=lift`（默认）

工作流：

1. 加载已训练的 Gaussian Wrapping checkpoint/PLY。
2. 冻结所有原始高斯参数、appearance、occupancy 和其他网络。
3. 禁止 densification/pruning。
4. 只训练 `[N,16]` 语义嵌入和 `16→C` 分类头。
5. 使用语义扩展 renderer 的 `EMBEDDING_ONLY` backward。

该模式不改变高斯几何、外观、mip filter、深度、法线和网格，因此最符合“在完全继承 Gaussian Wrapping 所有功能的基础上增加语义”。

建议使用独立入口：

```text
gaussian_wrapping/semantic_lift.py
```

不修改原 `train.py` 的默认行为。

### 8.3 `semantic=joint`（可选）

从头或中途联合 RGB/几何与语义训练。默认仍使用 `EMBEDDING_ONLY`，语义只学习嵌入和分类头；RGB/几何损失按原 Gaussian Wrapping 路径更新几何。

只有显式设置 `gradient_policy=full` 时，语义 loss 才能反向影响 opacity/geometry。该模式必须记录在 checkpoint metadata 中。

### 8.4 损失

第一版：

```text
L_sem = weighted_cross_entropy(logits, label, ignore_index=-1)
```

可选择按照 Gaga 方式除以 `log(C)`，但配置和日志必须明确。

可选 3D 一致性项：

```text
L_3d = neighbor KL / contrastive consistency
```

实现必须先采样再 KNN，并使用 `simple-knn`、`torch_cluster` 或 chunked KNN；不得对全部高斯构造完整 `N×N` 的 `torch.cdist`。

总损失：

```text
L = L_GW_original + λ_sem L_sem + λ_3d L_3d
```

在 lift 模式中只执行语义相关项，原始 Gaussian Wrapping loss 无需计算。

## 9. 建议目录

```text
GaussianWrappingGaga/
├── gaussian_wrapping/
│   ├── train.py                         # 默认行为不变
│   ├── semantic_lift.py                 # 新增
│   ├── semantic_render.py               # 新增
│   ├── semantic_eval.py                 # 新增
│   ├── semantic/
│   │   ├── config.py
│   │   ├── observations.py
│   │   ├── attributes.py
│   │   ├── head.py
│   │   ├── losses.py
│   │   └── checkpoint.py
│   ├── gaussian_renderer/
│   │   ├── radegs.py                    # 仅增加可选 semantic 分派
│   │   └── ours.py                      # 仅增加可选 semantic 分派
│   └── scene/
│       ├── gaussian_model.py            # 接入属性生命周期
│       ├── cameras.py                   # 可选 semantic observation
│       └── dataset_readers.py
└── submodules/
    ├── diff-gaussian-rasterization/     # 原 RaDeGS，保留
    ├── diff-gaussian-rasterization_ours/# 原 Ours，保留
    ├── diff-gaussian-rasterization_gw_semantic/
    └── diff-gaussian-rasterization_gw_ours_semantic/
```

SOF、MS、regularizers、mesh extraction、texture、benchmark 和 Blender 目录第一阶段不改。它们继续使用原有模型字段和原始 renderer。语义只作为附加能力存在，不成为这些功能的依赖。

## 10. 逐文件实施矩阵

| 区域 | 改动 | 不变量 |
|---|---|---|
| `scene/gaussian_model.py` | 增加语义属性、getter、初始化和 registry hook | 既有字段顺序、加载和优化逻辑不退化 |
| `scene/cameras.py` / `dataset_readers.py` | 可选挂载语义观测 | 无 mask 时相机行为完全不变 |
| `utils/camera_utils.py` | 标签 nearest resize 与校验 | RGB resize 不变 |
| `gaussian_renderer/radegs.py` | 根据 semantic option 分派到语义扩展，追加返回键 | 原返回键和 integration 不变 |
| `gaussian_renderer/ours.py` | 同上 | 原 SDF/integration 不变 |
| RaDeGS semantic extension | 在现有 forward/backward 添加 auxiliary accumulation | depth/normal/coord 原逻辑不改 |
| Ours semantic extension | 在现有 forward/backward 添加 auxiliary accumulation | median depth/normal 原逻辑不改 |
| `semantic_lift.py` | 冻结原模型并训练语义 | 原 `train.py` 不变 |
| `semantic_render.py` | 输出 feature/logit/label/可视化 | 原 `render.py` 不变 |
| `semantic_eval.py` | Hungarian IoU、mIoU、pixel accuracy | 原 RGB/几何评测不变 |

## 11. 验收与回归测试

### 11.1 上游功能零退化

对 RaDeGS 和 Ours 分别执行：

- semantic off 的 forward 输出与上游同提交对比；
- 同 seed 的短训练 loss、densification 数量和 checkpoint 对比；
- render 输出 RGB/depth/normal/alpha 对比；
- mesh extraction 顶点/面数及几何误差对比；
- 原有 texture、benchmark、SOF/MS smoke test。

语义关闭应优先要求 bitwise equality；受非确定 CUDA 影响的步骤则使用明确的数值容差。

### 11.2 CUDA 正确性

- 用小场景的纯 PyTorch alpha compositor 作为 16 维 forward oracle；
- 对 `aux_features` 做 finite-difference/双精度参考梯度检查；
- `EMBEDDING_ONLY` 下，只施加语义 loss 时验证 geometry/opacity/scale/rotation 梯度严格为零或 `None`；
- `FULL` 下验证相应几何梯度非零且与参考一致；
- 验证两套 renderer 在相同贡献序列下的语义合成定义一致；
- 覆盖空可见集、单高斯、early termination、alpha clamp、override color。

### 11.3 高斯生命周期

每次 clone/split/prune 后验证：

```text
len(xyz) == len(semantic_embedding) == 所有 per-Gaussian 字段长度
```

并检查：

- 语义值继承正确；
- optimizer moments 对齐；
- PLY round-trip 不丢 `filter_3D`、occupancy、`gaussian_features_*` 或 `obj_dc_*`；
- 旧 PLY/旧 checkpoint 无语义字段时仍可加载；
- Gaga `obj_dc_*` PLY 可以导入。

### 11.4 端到端验收

以仓库已有 Mip-NeRF 360 场景为基准：

1. 加载真实 Gaussian Wrapping PLY。
2. 执行 semantic lift。
3. 验证所有原字段数值未改变。
4. 对 lift 前后 RGB、depth、normal 和 mesh 做回归对比。
5. 计算 Hungarian-matched mIoU、mean class accuracy、pixel accuracy。
6. 报告 renderer forward/backward latency、峰值显存和 kernel occupancy。

不能只报告语义可视化；必须同时报告“原功能未退化”的证据。

## 12. 实施阶段

### Phase 0：建立可验证基线

- 从 Gaussian Wrapping 当前提交创建保留历史的 fork。
- 固定依赖与两套 renderer 的基线输出。
- 建立 semantic-off 回归测试。

### Phase 1：属性与数据层

- 实现语义属性 registry。
- 实现 PLY/checkpoint round-trip。
- 接入 Gaga associated masks。
- 完成 clone/split/prune 测试。

### Phase 2：RaDeGS CUDA

- 从 Gaussian Wrapping RaDeGS extension 复制出独立语义扩展。
- 增加 16 维 forward/backward。
- 完成 compositor 与梯度隔离测试。

### Phase 3：Ours CUDA

- 对 Ours extension 做同样扩展。
- 保持其 median depth、normal、SDF/integration 路径不变。
- 完成双后端一致性测试。

### Phase 4：Semantic lift

- 实现分类头、CE、可选 3D consistency。
- 实现语义训练、渲染、导出与 Hungarian IoU 评测。
- 跑真实场景端到端回归。

### Phase 5：可选联合训练

- 接入 densification/pruning 全周期。
- 增加 `EMBEDDING_ONLY` 与 `FULL` 实验开关。
- 评估语义对几何和 mesh 的收益/退化。

### Phase 6：工程化

- 补齐 launcher、文档、环境安装和模型格式迁移工具。
- 合并前跑完整 Gaussian Wrapping 功能矩阵。

## 13. 明确不采用的方案

1. **直接替换为 Gaga rasterizer**：会失去 Gaussian Wrapping 特有输出和网格路径。
2. **先训练 GW，再让 Gaga 加载并保存 PLY**：现有证据表明会丢 GW 专有字段。
3. **第二次 RGB raster pass 输出语义**：重复计算且可能与主贡献路径不一致。
4. **复用 `_gaussian_features` 作为语义**：与法线场语义冲突，破坏既有功能。
5. **把动态类别数直接放进 CUDA**：显存和编译组合不可控，跨场景不灵活。
6. **默认让语义改变几何**：无法保证“完整继承”和 lift 前后视觉/网格不变。
7. **以现有 SemanticGaussianWrapping 重实现为基线**：功能面与上游不等价。

## 14. 关键架构决策

最终建议冻结以下决策作为实现契约：

- 上游基线：Gaussian Wrapping 当前提交，保留 git 历史。
- 语义维度：16，类别解码在 PyTorch。
- 合成：与 Gaga 相同的 premultiplied alpha 合成。
- 主后端：RaDeGS 与 Ours 均原生支持语义。
- 默认训练：semantic lift。
- 默认梯度：embedding only。
- 默认兼容：semantic off 直接走原扩展。
- 模型格式：保留全部 GW PLY 字段并追加 `obj_dc_0..15`。
- 数据准备：Gaga 掩码生成/关联保持外置。
- 第一阶段不改：SOF/MS、mesh、texture、benchmark、Blender 和原 `train.py` 默认路径。

这套边界既保留 Gaussian Wrapping 的全部能力，又真正把语义合成放进其原生 CUDA 光栅化过程；同时把 Gaga 从一个独立旧版 3DGS 分支，收敛为可维护、可验证的语义能力模块。

## 15. 上游追踪与许可证

当前审阅基线：

```text
GaussianWrapping: f868c33
Gaga:             6944849
```

新项目应记录两个 upstream remote，并把语义改动拆成清晰提交：

```text
baseline import
attribute/data layer
RaDeGS semantic CUDA
Ours semantic CUDA
lift/render/eval
tests and launchers
```

Gaussian Wrapping 的上游许可包含非商业研究/评估限制，复制或派生时必须保留原许可证和 notices；Gaga 为 MIT，迁入的语义设计与代码也必须保留其版权和许可声明。最终仓库应明确标注哪些文件来自 Gaussian Wrapping、哪些修改参考 Gaga，以及新增代码采用何种许可。
