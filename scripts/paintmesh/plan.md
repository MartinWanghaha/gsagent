# PaintMesh normal 全流程与可配置虚拟视角渲染方案

实现状态：阶段 A 的双后端渲染及阶段 C 的自动 `removed normal → LaMa → completed normal` 已落地。渲染入口为 `scripts/paintmesh/render_virtual_views.py`，同级适配器集中在 `render_virtual_worker.py` 的 `native()` / `pgsr()`。normal 输入准备/总校验位于 `prepare_paintmesh_lama_data.py`；独立 dataset、向量处理位于 `tools/paintmesh_normal.py`；推理位于 `LaMa/bin/predict_normal.py`；`run_inpaint.sh` Stage 2/3 按上游 normal 自动执行。融合初始化及几何 finetune 仍是后续设计。

根据 [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md) 当前的 RGB/depth 流程，虚拟视角渲染增加两个同级、可配置的后端：`inpaint360gs` 和 `edgs-pgsr`。默认使用 `inpaint360gs`，显式选择 `edgs-pgsr` 时同步输出：

```text
RGB + PGSR plane-depth + PGSR normal + alpha
```

关键原则是：

1. remove 阶段只删除 3DGS Gaussian，不单独“删除法线图”；
2. full 和 removed 3DGS 使用同一组虚拟相机、同一个选定后端；PGSR 分支同步渲染 RGB、depth、normal、alpha；
3. normal completion 固定为 `removed normal → LaMa → completed normal`，与 depth completion 平级；只要上游提供完整的 removed normal 就自动执行，不增加 normal completion 开关或模式选择；
4. 3DGS finetune 后，再用 PGSR normal/depth loss 做局部几何优化；
5. 最终 mesh 仍然由 PGSR depth + RGB 做 TSDF，法线用于监督和质量检查。

`VIRTUAL_RENDERER` 只选择虚拟视角渲染后端，不改变基础训练、对象删除、3DGS finetune 或最终 TSDF 的后端配置。下面的 normal 全流程图描述 `edgs-pgsr` 分支；`inpaint360gs` 分支保留现有 RGB/depth 补全能力。normal 是否执行由经过验证的上游模态决定，不根据后端名字硬编码，也不由用户额外启用。completed depth 派生的法线只可用于后续质量诊断，不替换 LaMa completed normal。

---

## 一、当前代码中的缺口

当前实现还不能直接完成这条流程：

- 虚拟视角双后端与 PGSR raw normal 导出已经实现；`virtual_pose.py --poses-only` 负责相机生成，`render_virtual_worker.py::pgsr` 负责四模态同步渲染；
- [`PGSRRenderer.render()`]( /home/martin/code/gsagent/submodules/EDGS/source/renderers/pgsr.py:146) 已经返回：
  - `render`
  - `plane_depth`
  - `rendered_normal`
  - `rendered_alpha`
  - 可选的 `depth_normal`
- [`EDGS/render.py`]( /home/martin/code/gsagent/submodules/EDGS/render.py:378) 的最终真实视角输出仍需扩展 raw normal；这与已完成的虚拟视角 raw normal 导出是不同入口；
- `prepare_paintmesh_lama_data.py`、`run_inpaint.sh` Stage 3 和 `predict_normal.py` 已实现自动 normal 输入准备、LaMa 推理和完成校验；尚未将 completed normal 接入融合/初始化和 loss；
- [`edit_object_inpaint.py`]( /home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_inpaint.py:251) 当前 finetune 只有 RGB loss，没有 depth/normal loss；
- [`compose_utils.py`]( /home/martin/code/gsagent/submodules/Inpaint360GS/utils/compose_utils.py:90) 新 Gaussian 默认单位旋转、三轴相同尺度，PLY 中的 `nx,ny,nz` 目前也不会被实际用于初始化。

---

## 二、目标完整流程

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
        └───────────┴───────────────────────┘
                                │
                                ▼
                  RGB-D-N support point fusion
                                │
                                ▼
                 normal-aware Gaussian initialization
                                │
                                ▼
                     RGB 3DGS finetune
                                │
                                ▼
                PGSR depth/normal geometry refinement
                                │
                                ▼
                    final EDGS-PGSR render
                                │
                                ├── final RGB
                                ├── final depth
                                ├── final normal
                                └── RGB-D TSDF mesh
```

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

旧 run 没有模态声明却存在 normal 时，要求先通过虚拟渲染入口生成有效 manifest；不能仅凭目录存在就信任来源。不新增 normal 开关，也不要求单独指定 normal checkpoint，首版复用现有 `LAMA_MODEL_PATH`。

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

首版复用 RGB Big-LaMa 权重，是“编码法线作为三通道图像补全”的基线，不把它标成专门训练的法线模型。单位化与朝向校正并不保证 depth-normal 一致性或多视角一致性。completed depth 派生法线只能用于角度误差诊断，不能覆盖此分支输出；后续融合/训练使用的置信度另行评估，不把 valid 或 removed alpha 当作洞内预测置信度。

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

## 七、阶段 D：扩展 RGB-D-N 点云融合

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

法线变换时只能使用旋转，不能加平移：

```text
n_world = R_camera_to_world · n_camera
```

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
5. 使用 alpha、normal confidence 和视角夹角过滤异常点。

---

## 八、阶段 E：使用 normal 初始化新 Gaussian

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

## 九、阶段 F：3DGS finetune 增加几何子阶段

现有 [`edit_object_inpaint.py`]( /home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_inpaint.py:321) 的 5000 次迭代本质上是 RGB finetune。建议不要一开始直接把 EDGS 和 Inpaint 两套 renderer 强行混在同一进程里。

推荐拆成：

```text
Stage 5a: 原有 RGB 3DGS finetune
Stage 5b: 独立 PGSR geometry refinement
```

新增：

```text
submodules/EDGS/tools/finetune_pgsr_geometry.py
```

输入：

```text
RGB finetune 后的临时 PLY
completed depth
completed normal
normal confidence
camera manifest
hole masks
```

只优化：

- 新增 support Gaussians；
- hole 区域附近的局部 Gaussian；
- 必要时允许少量 opacity/scale 调整。

背景 Gaussian 冻结或使用很小学习率，并继续使用空间 gate 防止已知区域漂移。

PGSR 几何损失可以定义为：

```text
L = L_rgb
  + λ_depth L_depth
  + λ_normal L_normal
  + λ_consistency L_consistency
```

其中：

```text
L_depth =
    SmoothL1(log(D_pred), log(D_target))

L_normal =
    1 - dot(N_pred, N_target)

L_consistency =
    1 - abs(dot(N_pred, N_depth(D_pred)))
```

预测法线：

```python
N_pred = normalize(
    rendered_normal / rendered_alpha.clamp_min(1e-6)
)
```

有效区域：

```python
valid =
    hole_mask
    & (alpha > alpha_threshold)
    & (normal_confidence > confidence_threshold)
    & finite(depth)
    & (depth > 0)
```

建议几何阶段从 500～1000 步开始，验证稳定后再增加到 1500～2000 步。没有必要盲目再跑一个完整的 5000 步全场景 RGB 训练。

---

## 十、阶段 G：最终 PGSR render 和 mesh

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
4a. 生成或验证 virtual_cameras.json
4b. 解析 VIRTUAL_RENDERER，检查模型与请求模态是否受支持
4c. 统一入口调用所选后端，渲染 full 和 removed，提交 render manifest
4d. 用所选后端的 removed RGB 打包 tracker archive
4e. 建立与 render artifact 绑定的 tracking session
```

### `run_inpaint.sh`

normal completion 根据上游数据自动执行，无需额外配置。默认原生分支没有 normal 时继续现有 RGB/depth 路径；PGSR 分支有 normal 时必须在同一个 Stage 3 完成 normal LaMa。虚拟视角 backend 不隐式选择 finetune 或最终 mesh renderer。下面的 RGB-D-N 融合和几何训练仍是后续独立工作，不是完成 normal LaMa 的前置条件。

```text
Stage 2:
    准备 RGB/depth 输入；自动检测并准备 normal/valid/推理 mask

Stage 3:
    RGB LaMa + depth LaMa + normal LaMa（有 normal 就必须执行）
    按 required modalities 统一验证并提交 completion manifest

Stage 4:
    RGB-D-N support fusion

Stage 5:
    现有 RGB 3DGS finetune

Stage 5b:
    PGSR depth/normal geometry refinement

Stage 6:
    发布 inpainted EDGS model

Stage 7:
    PGSR render + TSDF + raw normal 输出

Stage 8:
    semantic lifting 和最终提交
```

与本批 completion 相关的既有配置保持不变：

```bash
VIRTUAL_RENDERER=inpaint360gs
NORMAL_ALPHA_MIN=0.01
```

`VIRTUAL_RENDERER` 是已有渲染后端选择，`NORMAL_ALPHA_MIN` 是已有渲染有效性阈值，都不是 normal completion 开关。不新增 `NORMAL_PIPELINE`、`NORMAL_COMPLETION_MODE` 或 `ENABLE_NORMAL`。全视角融合与几何 finetune 的启用策略在后续实现时单独确定，不影响本批自动 LaMa completion。

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

复用并扩展现有完成链（不新增平行的 normal-only 完成标记）：

```text
render_manifest.json                # 已实现，full/removed 各一份
lama_input_manifest.json            # 扩展 normal 输入与 required modalities
lama_completion_manifest.json       # RGB/depth/normal 一起验收和提交
fusion_manifest.json                # 后续 RGB-D-N 融合扩展
```

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

- 恰好 30 个 frame；
- RGB、depth、normal、alpha、mask 尺寸一致；
- 没有 NaN/Inf；
- 有效 normal 的模长接近 1；
- 无效 normal 等于 `[0,0,0]`；
- mask 外 normal 与 removed normal 完全一致；
- camera manifest ID 一致；
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

### P1：改善补全密度和法线质量

```text
30 视角 RGB-D-N 融合
+ normal-aware Gaussian rotation/scale 初始化
+ PGSR geometry refinement
```

这是最值得优先做的阶段，也能同时改善你之前遇到的“补全区域点云稀疏”和“补全法线不理想”。

### P2：提升 LaMa normal 预测质量

```text
专门 normal LaMa checkpoint
或 RGB-D-N joint completion 网络
```

首版直接复用当前 Big-LaMa 权重；后续若质量诊断不达标，再评估专用 normal 训练或 joint 网络，不把训练专用权重作为自动 normal 分支落地的前置条件。是否采用新网络属于后续方案评审，不增加本批模式选择。

本文记录设计要求与实现顺序；所列新增入口、配置和 worker 在落地并通过对应验收前，均不代表已有运行能力。
