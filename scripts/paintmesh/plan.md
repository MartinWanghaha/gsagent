# PaintMesh normal 全流程与独立 EDGS-PGSR 局部几何优化方案

实现状态：阶段 A 的双后端渲染及阶段 C 的自动 `removed normal → LaMa → completed normal` 已落地。渲染入口为 `scripts/paintmesh/render_virtual_views.py`，同级适配器集中在 `render_virtual_worker.py` 的 `native()` / `pgsr()`。normal 输入准备/总校验位于 `prepare_paintmesh_lama_data.py`；独立 dataset、向量处理位于 `tools/paintmesh_normal.py`；推理位于 `LaMa/bin/predict_normal.py`；`run_inpaint.sh` Stage 2/3 按上游 normal 自动执行。

阶段 F 已接入：保持已有 RGB-D 反投影、单 seed Gaussian 初始化和 RGB finetune，在其后运行**独立、默认关闭的 EDGS-PGSR 局部几何优化**，新增 LaMa normal 监督及局部权重渐增调度。阶段 D/E 的 RGB-D-N 融合和 normal-aware 初始化延期，不作为阶段 F 的依赖。局部入口、开关、配置、编辑范围、checkpoint 与发布链已实现；通过小型合成 30 帧 CUDA 链路测试，不代表真实 kitchen 完整训练的质量已验证。

新增下一轮设计：**D0 周边密度匹配的深度支持点重采样（待实现）**，在 Stage 4 后、Stage 5a 初始化前执行，不改变 Stage 5b 的固定点数契约。优先解决 seed 采样密度与周边背景不匹配，不提前启用 D/E 的全视角 RGB-D-N 融合或 normal-aware 初始化。本次仅更新文档；下文原流程图和“反投影不变”指已实现基线，D0 小节单独定义后续改动。

根据 [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md) 当前的 RGB/depth 流程，虚拟视角渲染增加两个同级、可配置的后端：`inpaint360gs` 和 `edgs-pgsr`。默认使用 `inpaint360gs`，显式选择 `edgs-pgsr` 时同步输出：

```text
RGB + PGSR plane-depth + PGSR normal + alpha
```

关键原则是：

1. remove 阶段只删除 3DGS Gaussian，不单独“删除法线图”；
2. full 和 removed 3DGS 使用同一组虚拟相机、同一个选定后端；PGSR 分支同步渲染 RGB、depth、normal、alpha；
3. normal completion 固定为 `removed normal → LaMa → completed normal`，与 depth completion 平级；只要上游提供完整的 removed normal 就自动执行，不增加 normal completion 开关或模式选择；
4. normal 不参与本轮反投影或初始化；仅在显式开启局部优化后，用 completed normal/depth 对 RGB finetune 结果做 PGSR 几何监督；
5. 最终 mesh 仍然由 PGSR depth + RGB 做 TSDF，法线用于监督和质量检查。

局部优化开关独立于自动 normal completion，也独立于 `VIRTUAL_RENDERER`。关闭时完整保留原流程；开启时只影响当前 inpaint run。禁止更改 `run_seg.sh` 的全局训练、`configs/gs/pgsr.yaml` 的默认几何启用时间，或给全局 loss 隐式注入 LaMa 目标。

`VIRTUAL_RENDERER` 只选择虚拟视角渲染后端，不改变基础训练、对象删除、3DGS finetune 或最终 TSDF 的后端配置。下面的 normal 全流程图描述 `edgs-pgsr` 分支；`inpaint360gs` 分支保留现有 RGB/depth 补全能力。normal 是否执行由经过验证的上游模态决定，不根据后端名字硬编码，也不由用户额外启用。completed depth 派生的法线只可用于后续质量诊断，不替换 LaMa completed normal。

---

## 一、当前代码中的缺口

当前实现状态及仍然延期的缺口：

- 虚拟视角双后端与 PGSR raw normal 导出已经实现；`virtual_pose.py --poses-only` 负责相机生成，`render_virtual_worker.py::pgsr` 负责四模态同步渲染；
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

---

## 二、本轮目标流程

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

## 七、阶段 D：支持点密度与后续 RGB-D-N 融合

### D0. 周边密度匹配的深度支持点重采样（待实现）

#### D0.1 范围与代码依据

目标是在 EDGS-PGSR 局部优化**之前**，使补全初始化点与相邻正常背景具有相近的局部表面密度。保持 completed depth 的 z-depth 定义、精确相机、normal 自动 completion 和 Stage 5b normal loss；不反投影 LaMa normal，不修改全局 `run_seg`，不把几何损失当作增密机制。

当前可确认的行为：

- `point_utils.py::create_point_cloud()` 为每个像素生成一个点，`ply_color_fusion()` 以 mask 筛选，没有 stride/voxel 下采样；
- `run_inpaint.sh` 从 30 个 PLY 选 `FUSION_SEED_FRAME=4`，不是 30 帧联合初始化；
- `GaussianModel.inpaint_setup()` 用 `nb_neighbors=5, std_ratio=4.0` 做统计离群过滤；
- `compose_utils.py::create_from_pcd_our()` 用 `sqrt(distCUDA2)` 初始化等向 scale，平方距离下限 `1e-7`；
- 5a 的 densify/prune 是 RGB 梯度驱动，最后还有 seed 空间 gate；5b 不增加点。

所以稀疏可能产生于像素采样上限、单 seed 覆盖、离群过滤、5a 剪枝或 gate；目前没有实测定位 kitchen 的主因。先报告每阶段点数、密度与拒绝原因，不能预设只需增加采样倍率。

#### D0.2 新的局部流程与边界

```text
Stage 4a：保留原 RGB-D 反投影 -> fused/mask/00000..00029.ply
Stage 4b（新增可选）：边界密度估计 -> seed 表面采样 -> init_support.ply
Stage 5a：用 init_support 初始化；用原 seed support 做 gate；RGB 训练不变
Stage 5b：可选独立 PGSR 几何优化，仍固定点数/行序
```

Stage 4a/4b 同属 shell Stage 4，不增加顶层 stage 编号。首版采用 **seed 表面自适应重采样 + 30 视角一致性检查**，不直接合并 30 帧。这样保留既有 seed gate 的空间范围，避免直接拼接不同 LaMa 深度造成多层表面；缺少 seed 可见性的区域明确标为未覆盖，后续扩大覆盖需要另行设计多视角 gate。

拟新增 `SUPPORT_DENSITY_MODE=legacy|boundary_adaptive`（默认 `legacy` 保持旧 run），以及 `SUPPORT_DENSITY_CONFIG`。这是待实现接口，不是当前可用命令。其开关与 `LOCAL_GEOMETRY_REFINE`、normal completion 独立；首版 adaptive 要求 manifest 明确为 PGSR plane z-depth，不静默把原生 renderer 的 depth 当成同一物理量。legacy 不新增 PGSR 依赖。

#### D0.3 周边参考密度：同表面、可见、实际保留

1. 从与 Stage 5a **完全相同的删除选择**得到保留行集合。不能只按目标 ID 的反集近似 classifier/凸包规则，不能使用已删除对象的中心作为参考。建议抽取只读 retained-row 选择函数，两个阶段共用并绑定选择结果 hash；禁止重构时改变旧分支行为。
2. 各帧构造 `dilate(M, outer) - dilate(M, inner)` 的已知环带，按图像尺寸缩放环带宽度；建议初值 outer 为短边 3%、inner 为短边 0.5%。不足时仅在有上限的同表面范围内扩张，不扩到全场景。
3. 投影实际保留的 Gaussian 中心，以深度正值、视锥、removed alpha、与 removed depth 的残差和局部面法向距离筛选可见背景。Gaussian 中心不是精确表面点，深度容差需兼顾局部 spacing/scale 并设上限。过滤低 opacity、孤立异常点；不按纹理丰富度挑参考。
4. 以深度连通性/局部平面划分背景 patch，分开墙、地面、台面以及深度断层两侧；跨视角按原 Gaussian 行号去重。切平面只从可靠深度/邻域 PCA 得到；LaMa normal 不作为参考真值。
5. 在同 patch 切平面估计 `rho_ref = median(k / (π*r_k²))`，初值 `k=8`，排除自身/重复点/跨层点；查询邻域不能被细环带裁断，可从更宽同表面集合查询并仅保留中心落入环带的样本，或使用有效面积校正。记录样本量、分位数、有效面积和稳定性。拒绝非表面型或边界偏差严重的估计。
6. 沿 seed 深度的连通表面向洞内传播目标密度，不跨断层传播、不对所有区域使用同一个均值。`rho_target = density_ratio * rho_ref`，初值 ratio=1。间距 `h_target = c / sqrt(rho_target)`，常数 c 与采样规则有关，用合成平面标定，不直接把 `r_k` 当最近邻间隔。

可信参考不足或表面归属不明时，输出未解决区域与原因。adaptive 模式默认拒绝提交“密度匹配完成”，可由用户显式选 legacy 新 run，不能静默伪装成功。removed alpha 仅用于已知区筛选，洞内接近零不能拒绝新补全几何。

#### D0.4 受控深度表面采样与去重

1. 对 seed 正且有限的 completed depth 按原相机公式反投影。以像素网格构建局部三角片，只连接 mask 内的同表面有效邻点；深度跳变、退化片、跨断层三角形剔除。mask 边缘按有效区域裁剪，不能为凑点跨 hole 外界；记下因此损失的窄边覆盖。
2. 按世界坐标三角形面积积分目标密度确定预算：`N_target ≈ ∫ rho_target dA`，并减去该片内可信的现存同表面背景点贡献。保留点只参与统计/排斥，不复制进新点 PLY，也不删除背景。
3. 稀疏 patch 内进行三角片细分/面积加权候选采样，再用变半径表面距离抑制（Poisson-disk 类规则）调整间距；过密处下采样。局部间距缓变、分 patch 执行，不能让隔墙/薄表面另一侧的点互相排斥。固定随机种子和排序确保可重现。
4. 新点严格留在深度三角面上；回投 seed 在同一有效像素单元内插值 completed RGB。禁止复制 XYZ、随机三维 jitter、跨深度边缘双线性插值或仅增大 scale。细分增加的是采样点，不是恢复额外的真实几何细节。
5. 重投影到其他相机，在已知区对比有效 removed depth，在 hole 内对比 completed depth；以相对深度误差和局部间距对应的绝对误差联合设阈值。先分类为可检验一致、可检验冲突、被遮挡、出视野/无效；后两者不当作冲突。要求支持视角有足够视差，连续近重复帧不当成独立强证据。记录有效视角数、支持数与冲突率，不用固定“30 帧都支持”的规则。
6. 首版区分 multi-view confirmed 与 seed-only uncertain：无第二个可靠观测的点不获得高置信标记；可保留为不确定候选单独统计，不计入高置信覆盖验收。其他视角的 LaMa 完成值只能提供一致性证据，不是真值。明显冲突候选剔除，不在不同表面之间平均 XYZ。
7. 对现存背景与新增候选做同表面、局部半径去重并重新测量密度，有限次补足缺点 patch；所有补采样仍经过相同质量过滤。建议总点数上限 500k、原 seed 有效点数倍率上限 8、最多 3 次补足，均为待基准测试的初值。预算不足记录 density deficit，停止并报告，而非无限加点或放松几何阈值。

不使用全局固定 voxel size 作为密度目标；空间哈希仅用于加速邻域查询。LaMa normal 首版仍只用于既有 5b loss 和可选角度诊断，不新增 normal 硬过滤、normal-aware 初始化或 normal 推导 depth。

#### D0.5 初始化过滤与 gate 必须解耦

- `init_support.ply` 只负责新 Gaussian 初始化；原 `fused/mask/<seed>.ply` 继续作为 `gate_support`。建议新增明确的 `--init_support_ply` / `--gate_support_ply`，legacy 两者指向原 support；不能让现有 `args.supp_ply` 同时承担两种不同身份。
- 原 gate 的 seed camera、mask 膨胀规则、距离阈值与 support hash 不因重采样改变。候选须在原 gate 内；gate 限制导致的密度缺口必须报告，不扩大 gate。重采样后的点分布不能重新计算一个更大的 gate 距离阈值。
- adaptive 在密度验收前执行并记录原 statistical outlier removal；初始化消费已验证点，避免重复过滤造成第二次不透明损失。legacy 继续原过滤路径。若需补足，补足后重新过滤与验收，并保证交付给 initializer 的点与 manifest 完全一致。
- 首版保留单位 quaternion、opacity=0.1、SH/语义 KNN 初始化，scale 仍由最终点集的 `distCUDA2` 计算。记录线性 scale/h_target 分布与 `1e-7` 平方距离 floor 命中率；floor 限制目标时显式报告，不贸然更改全局数值下限。normal-aware 各向异性初始化继续延期。
- 5a 仍可 densify/prune；需要输出新点数量及局部密度在初始化、训练结束、gate 提交后的变化，避免“seed 已达标但最后仍稀疏”。首版不承诺强制最终点数；若稀疏主因在 5a，后续另立密度保持增密策略，不能靠不断增加 seed 掩盖。
- `rgb_context`、5a receipt、5b request/checkpoint、发布和最终提交同时绑定 init support、原 gate support、density artifact。现有只绑定单 support 的接口必须升级；5b 实际可编辑行仍在 5a 最终行顺序确定后生成，不能使用初始化索引。

#### D0.6 文件落点、数据与恢复契约

以下均为拟新增/拟修改，尚未编码：

| 文件 | 责任 |
|---|---|
| `scripts/paintmesh/prepare_density_support.py` | CPU 调度入口：验证来源、加载参考、生成/检查支持点、提交 manifest |
| `scripts/paintmesh/support_density.py` | 表面分组、密度估计、目标场传播、采样、去重、重投影检查；避免依赖全局 trainer |
| `scripts/paintmesh/configs/support_density.yaml` | 独立采样参数、环带、有效性阈值、预算、随机种子与 debug 配置 |
| `run_inpaint.sh` | Stage 4b 调度、legacy/adaptive 路由、Stage 5a 两种 support 参数 |
| `scene/gaussian_model.py` / `edit_object_inpaint.py` | 共用保留行选择、消费已过滤初始化点、原 gate support 独立传递、阶段计数 |
| `local_geometry_io.py` / 发布与最终校验工具 | 两种 support 与新 artifact 绑定，5b gate 保持原空间契约 |
| `scripts/paintmesh/tests/test_support_density.py` | CPU 几何/密度/身份测试；小型 GPU 初始化及 Stage 5a/5b 集成测试另行标记 |

```text
fused/mask/<frame>.ply                      # 原 RGB-D 输出，保留不覆盖
fused/density/init_support.ply              # RGB 点云，新增 Gaussian 初始化输入
fused/density/support.npz                   # XYZ/RGB、seed UV、patch、目标间距、来源与支持视角统计
fused/density/reference.npz                 # 参考 Gaussian 行号、patch 与密度统计
fused/density/debug/                        # 密度热图、间距分布、覆盖/拒绝原因图
manifests/support_density_manifest.json     # 与原 fusion manifest 并列
```

新 manifest 绑定实际保留集合、源 PLY、fusion/LaMa/render artifact、30 帧相机与 mask、seed、两种 support、配置、算法版本、随机种子和输出 hash。不要向旧 fusion manifest 填入含义不同的输出后仍保留原 identity。结果逐个校验后原子提交 complete；统计文件和失败诊断可留存但不代表完成。变更模式/参数/数据须新 run 或拒绝旧产物；Stage 5/6 续跑时验证密度依赖，legacy 不要求该文件。启用 adaptive 后缺数据/身份不匹配不得静默退回旧 seed。

debug 建议 adaptive 默认开启，属于 Stage 4b 诊断，独立于 Stage 5b 的训练 debug。每个 patch 输出 `rho_ref/rho_init`、间距分位数、目标/预算/实际点数、过滤损失、未覆盖面积、支持视角和不确定比例；同时保存固定 seed 的前后投影，避免仅凭查看器蓝点数量判断。

#### D0.7 实施顺序与验收

1. **先做只读审计**：同一 kitchen run 测量 raw seed、过滤后 seed、5a gate 前后和 5b 的密度、scale、alpha；缺少中间产物的项标 unavailable，不根据最终 PLY 猜行来源。确认稀疏主要在哪一步产生，再决定采样预算。
2. 实现合成平面上的参考估计、seed 自适应采样与预算控制；CPU 单测通过后，输出候选 PLY/debug，暂不接训练。
3. 分离 init/gate support，接 Stage 4b 与身份链；验证关闭时 legacy 输出/参数行为不变，再接初始化和 GPU smoke test。
4. 真实场景固定 mask、LaMa、相机和随机种子，比较 legacy/adaptive × 关闭/开启 Stage 5b 四组；区别采样收益与 normal loss 收益，不宣称仅凭训练 loss 证明几何改进。

建议验收目标（工程初值，非现有实测保证）：可靠同表面 patch 的初始化密度比 `rho_init/rho_ref` 在 `0.75～1.33`，同时报告每 patch 和面积加权分布、kNN 间距分位数。定义分母为全部待补表面面积，分别报告总覆盖、高置信覆盖和未覆盖比例，不能只挑有点的 patch 计算通过率。5a/5b 后重测相同指标；未达标明确区分初始化失败或训练后密度退化。

测试覆盖正面/倾斜平面、两相邻密度、前后双层与遮挡、深度跳变、窄 mask/边界、无参考、NaN/零深度、重复/极密点、尺度整体缩放、分辨率改变、点预算耗尽、确定性和 manifest 失配。检查新点不越原 gate、背景未改、采样不生成跨层桥面，低纹理区域不依赖 RGB 梯度也可获得采样。密度达标还须检查渲染 alpha、洞内 depth 残差、边界接缝、洞外 RGB 和显存/耗时；completed depth 是伪监督，真实几何改善仍需独立视角/参考或人工检查。

### D1. 扩展 RGB-D-N 点云融合（仍延期，不是 D0 前置条件）

现有实现保持 `edit_object_removal_plyfusion.py`、`point_utils.py` 的 RGB-D 反投影算法、30 个支持 PLY、`FUSION_SEED_FRAME` 和既有 fusion manifest 不变。下面 D1 仅保留未来 RGB-D-N 联合融合设计，不实施这些参数或本节的 normal support 契约，也不以完成 D1 作为局部优化或 D0 的前置条件。D0 的 `fused/density/support.npz` 是独立的密度采样 sidecar，不等于此处的 RGB-D-N support。

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

## 八、阶段 E：使用 normal 初始化新 Gaussian（延期，不在本轮范围）

本轮保留现有单位旋转、等向尺度初始化，不修改 `compose_utils.py`。本节是未来候选优化；Stage 5b 从 RGB finetune 后的已有参数开始，不能为了接入 normal loss 重做 seed 初始化。

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

首版不启用局部 densify/prune、opacity reset、SH degree 递增、全局 scale loss 或多视角 LNCC/reprojection loss。这些均不是新增 LaMa normal loss 的前置条件；未来若加入，另行定义局部调度和验收。

### F3. 输入与局部编辑范围

输入严格绑定当前 inpaint run：

- Stage 5a 最终提交的含语义 PLY、其 hash、行数与 SH degree；不是最初的 full/removed PLY；
- 30 个精确虚拟相机，以及 RGB/depth/normal 共用的 hole masks；不能重新生成轨迹或以原始真实 RGB 监督洞区域；
- 已验证的 completed RGB、completed plane z-depth、float32 HWC completed normal、normal validity 和 LaMa completion manifest；depth 必须与当前模型同场景尺度，normal 为相机坐标、朝向相机；
- 与 Stage 5a 最终 PLY 行顺序绑定的 `editable_mask.npy` 和 `rgb_finetune_manifest.json`。

Stage 5a 的训练、反投影和 gate 算法不变，仅在输出处记录既有空间 gate 确认的可编辑行，包含 gate 内新增点与允许局部提交的原有点。sidecar 必须在最终行筛选/顺序确定后生成，并绑定 seed、mask、相机和 PLY hash；不能只保存训练前的索引或靠 `obj_dc` 猜测新增点。`surrounding_ids` 等最终处理必须同步映射 sidecar。

局部 optimizer 只持有可编辑行的 XYZ、rotation、scale。SH、opacity、语义 embedding、classifier 及其他所有行均冻结。将可训练局部张量与常量背景按原行顺序组合后做**完整场景渲染**，保留正确遮挡；不能只渲染局部子模型。使用新的 optimizer，不继承全局 Adam 动量；仅置零背景梯度不作为冻结保证。

首版保持点数、行序和语义字段不变；写出时在输入 PLY 的副本上只更新许可字段，避免 EDGS 普通 PLY writer 丢掉 `obj_dc_*` 或额外字段。最终检查编辑行仍处于已记录的 seed 空间 gate；越界行回退到 Stage 5a 参数并重新计算验收指标。参数级冻结不能阻止局部 splat 对 mask 外像素产生影响，因此仍需已知区域外观约束与渲染检查。

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

首版在有效目标内使用统一权重，并以较小的 `lama_normal` 权重表达伪监督的不确定性；`normal_valid` 不是置信度，不要求不存在的 confidence 文件。记录 completed depth 派生法线与 LaMa normal 的角度差供诊断，不能覆盖 LaMa 输出；后续引入置信度时再显式版本化其计算方法。

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

当 `t <= s` 时几何项权重为零；在 `s < t < s+r` 线性增加；`t >= s+r` 为完整目标权重。示例 `s=100,r=400`：`t=100/300/500` 对应 `0/0.5/1`。RGB 与 alpha 保持项从 `t=0` 生效。首版要求 `T>0, 0<=s<T-1, r>0, s+r<=T-1`，各权重/学习率有限非负；错误配置在 GPU 优化前拒绝。希望 5000 步局部优化可独立设置 `T=5000`，不会改变原 RGB 的 5000 步或全局调度。

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

阶段 F 本身不承诺修复点云稀疏：normal loss 不会凭空增加点。下一轮按 D0 单独改进初始化支持点密度，不自动启用延期的 D1/E，也不改变 F 的固定点数契约。

已执行局部 loss/config/gate/冻结字段测试，并用真实 PGSR CUDA 执行小型合成 30 帧的基线和最终渲染、3 步优化、中途 checkpoint 中断恢复、完成后复用与最后 checkpoint 恢复；验证输入变化会拒绝复用。PaintMesh/Inpaint360GS 回归 109 项通过、3 项跳过，另有全局 PGSR 相关 28 项通过。真实场景完整训练、质量消融与恢复的长期数值表现仍需进一步评估，不能把上述链路测试当作几何质量验收。

---

## 十、阶段 G：最终 PGSR render 和 mesh

Stage 7 始终使用 Stage 6 实际发布的 PLY：局部优化关闭时是 5a，开启时是完成验证的 5b。TSDF 算法与真实相机选择不变。以下最终真实视角 raw 导出仍为后续扩展，不是 Stage 5b 训练的前置条件；局部优化自己的诊断可直接保存 PGSR 返回的 raw 张量。

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

normal completion 根据上游数据自动执行，无需额外配置。默认原生分支没有 normal 时继续现有 RGB/depth 路径；PGSR 分支有 normal 时必须在同一个 Stage 3 完成 normal LaMa。虚拟视角 backend 不隐式启用局部优化。下面 Stage 5b 及其发布绑定已接入，Stage 4 与 seed 初始化保持原样。

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

复用并扩展现有完成链（不新增平行的 normal-only 完成标记）：

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

### P2：周边密度匹配的支持点重采样（下一轮，设计完成、待实现）

按 D0 先审计 raw seed、初始化和 5a gate 前后的密度；再做周边同表面密度估计、seed 深度表面自适应采样、多视角检查和 init/gate support 分离。该模块位于 Stage 4/5a 交界，与已实现的 P1 独立，不更改深度定义或 normal completion。D1 全视角 RGB-D-N 融合、E normal-aware 初始化以及训练期密度保持仍为后续候选，不作为 D0 的隐式依赖。

### P3：提升 LaMa normal 预测质量

```text
专门 normal LaMa checkpoint
或 RGB-D-N joint completion 网络
```

首版直接复用当前 Big-LaMa 权重；后续若质量诊断不达标，再评估专用 normal 训练或 joint 网络，不把训练专用权重作为自动 normal 分支落地的前置条件。是否采用新网络属于后续方案评审，不增加本批模式选择。

本文记录设计要求与实现顺序；所列新增入口、配置和 worker 在落地并通过对应验收前，均不代表已有运行能力。
