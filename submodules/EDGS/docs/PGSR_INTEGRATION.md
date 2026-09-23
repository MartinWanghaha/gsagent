# EDGS × PGSR 混合训练使用说明

本集成只替换 EDGS 的渲染后端，并在 EDGS 光度损失上增加 PGSR 的尺度、单视图法线一致性和多视图几何/NCC 损失。场景读取、相机、GaussianModel、RoMa 对应点初始化、优化器、densification、checkpoint 与导出仍由 EDGS 管理。

默认配置 `gs=base` 继续使用原生 EDGS renderer 且不启用 PGSR loss；只有显式指定 `gs=pgsr` 才会进入混合模式。

## 1. 代码与子模块

本集成使用 PGSR：

```text
https://github.com/zju3dv/PGSR.git
commit de24f1a38b350387e8d8fe381b2cd70c1ae946e7
```

在 EDGS 根目录检查子模块：

```bash
cd /path/to/EDGS
git submodule sync --recursive
git submodule update --init --recursive
git -C submodules/PGSR rev-parse HEAD
```

最后一条命令应输出上面的 commit。不要把 `submodules/PGSR` 加入 `PYTHONPATH`：PGSR 与 gaussian-splatting 含有同名的 `scene`、`utils` 和 `gaussian_renderer`，全局暴露 PGSR 根目录可能让 Python 静默导入错误模块。本项目只直接导入 `diff_plane_rasterization`。

## 2. 环境与 CUDA ABI 检查

所有命令均使用现有的 `paintmesh` Conda 环境：

```bash
conda activate paintmesh
cd /path/to/EDGS
```

在编译或重编 CUDA 扩展前检查 PyTorch、CUDA toolkit 和驱动：

```bash
python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("CUDA_HOME:", os.environ.get("CUDA_HOME", "<unset>"))
PY

nvcc --version
nvidia-smi
```

`torch.version.cuda` 与 `CUDA_HOME/bin/nvcc` 应来自兼容的 CUDA 版本；建议二者主、次版本一致，至少主版本必须一致。`nvidia-smi` 显示的是驱动可支持的最高 CUDA 版本，不等同于实际用于编译扩展的 toolkit 版本。若切换 PyTorch、CUDA toolkit 或 Python 版本，需要在 `paintmesh` 环境中重新编译两个 rasterizer。

## 3. 安装

`install.sh` 默认使用环境名 `paintmesh`。若该环境已经包含 PyTorch，脚本会保留现有 PyTorch/CUDA 版本，再在同一 ABI 下编译 EDGS 的两个扩展与 PGSR plane rasterizer；若环境不存在，则创建 Python 3.10、PyTorch 2.0.0、CUDA 11.8 的环境。它不会安装 PGSR 自带的 `simple-knn`，也不会安装 PyTorch3D；混合模式复用 EDGS/gaussian-splatting 的 `simple_knn._C`。

完整安装或重编全部扩展：

```bash
cd /path/to/EDGS
bash install.sh
```

训练所需依赖是默认安装范围。若还要使用 EDGS 的 Gradio demo、Open3D 和 Jupyter，可显式执行 `EDGS_INSTALL_DEMO=1 bash install.sh`。

如果 `paintmesh` 已经配置好，只需确认扩展可导入；无需重复执行完整安装脚本。仅需手动重建扩展时执行：

```bash
conda activate paintmesh
cd /path/to/EDGS

python -m pip install -e \
  submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install -e \
  submodules/gaussian-splatting/submodules/simple-knn
python -m pip install -e \
  submodules/PGSR/submodules/diff-plane-rasterization
```

当前 `diff-plane-rasterization` 已在 `paintmesh` 中编译安装时，直接进入下一节即可。

## 4. Smoke test

先验证两个 CUDA rasterizer 与 EDGS 的 `simple_knn` 均来自当前环境：

```bash
conda activate paintmesh
cd /path/to/EDGS

python - <<'PY'
import torch
import diff_gaussian_rasterization
import diff_plane_rasterization
from simple_knn import _C as simple_knn_cuda

assert torch.cuda.is_available(), "需要可用的 CUDA GPU"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("diff-gaussian:", diff_gaussian_rasterization.__file__)
print("diff-plane:", diff_plane_rasterization.__file__)
print("simple-knn:", simple_knn_cuda.__file__)
PY
```

再检查 Hydra profile，确保默认模式没有被改变：

```bash
python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir

config_dir = str(Path("configs").resolve())
with initialize_config_dir(version_base="1.2", config_dir=config_dir):
    native = compose(config_name="train")
    pgsr = compose(config_name="train", overrides=["gs=pgsr"])

assert native.gs.renderer.backend == "native"
assert native.gs.opt.pgsr_loss.enabled is False
assert pgsr.gs.renderer.backend == "pgsr"
assert pgsr.gs.opt.pgsr_loss.enabled is True
print("Hydra profiles: OK")
PY
```

给定一个有效的 COLMAP scene 后，分别跑一轮 native 回归 smoke 和 PGSR forward/backward smoke：

```bash
SCENE=/absolute/path/to/scene
NATIVE_OUT=/absolute/path/to/smoke_native
PGSR_OUT=/absolute/path/to/smoke_pgsr

CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=base \
  train.gs_epochs=1 \
  train.no_densify=true \
  init_wC.use=false \
  wandb.mode=disabled \
  gs.dataset.source_path="$SCENE" \
  gs.dataset.model_path="$NATIVE_OUT"

CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=pgsr \
  train.gs_epochs=1 \
  train.no_densify=true \
  init_wC.use=false \
  wandb.mode=disabled \
  gs.opt.pgsr_loss.single_view_from_iter=999999 \
  gs.opt.pgsr_loss.multi_view_from_iter=999999 \
  gs.dataset.source_path="$SCENE" \
  gs.dataset.model_path="$PGSR_OUT"
```

第二条命令会验证 plane rasterizer、RGB/尺度损失和反向传播。要用较低显存验证完整几何损失，可额外运行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=pgsr \
  train.gs_epochs=2 \
  train.no_densify=true \
  init_wC.use=false \
  wandb.mode=disabled \
  gs.opt.pgsr_loss.single_view_from_iter=0 \
  gs.opt.pgsr_loss.multi_view_from_iter=0 \
  gs.opt.pgsr_loss.sample_num=1024 \
  gs.dataset.source_path="$SCENE" \
  gs.dataset.model_path=/absolute/path/to/smoke_pgsr_full
```

## 5. 正式训练

典型训练命令：

```bash
conda activate paintmesh
cd /path/to/EDGS

CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=pgsr \
  train.gs_epochs=30000 \
  train.no_densify=false \
  gs.dataset.source_path=/absolute/path/to/scene \
  gs.dataset.model_path=/absolute/path/to/output \
  init_wC.use=true \
  init_wC.matches_per_ref=20000 \
  init_wC.nns_per_ref=3 \
  init_wC.num_refs=180 \
  wandb.mode=disabled
```

`init_wC.use=true` 首次运行时会下载 RoMa 与其 DINOv2 backbone 权重，需要可访问模型下载地址的网络；Torch 默认缓存位于 `~/.cache/torch/hub`。离线机器应提前在联网环境完成下载并复制该缓存，或先用 `init_wC.use=false` 验证训练链路。

`train.no_densify=false` 使用 EDGS 当前的 densification/pruning 路径；改成 `true` 则使用 EDGS 的 no-densify 训练方式。两种方式都不会切换到 PGSR 自带的 GaussianModel 或 densification 实现。

PGSR profile 的默认损失为：

```text
L = (1 - 0.2) * L1 + 0.2 * (1 - SSIM)
    + 100.0 * Lscale
    + I(step > 7000) * (
          0.015 * Lnormal
        + 0.03  * Lgeometry
        + 0.15  * LNCC
      )
```

其中单视图 normal loss 默认使用图像梯度权重；多视图默认最多选择 8 个候选邻居，角度不超过 30 度，归一化相机距离位于 `[0.01, 1.5]`。所有参数都可以用 Hydra 覆盖，例如：

```bash
python train.py gs=pgsr \
  gs.opt.pgsr_loss.single_view_from_iter=5000 \
  gs.opt.pgsr_loss.multi_view_from_iter=10000 \
  gs.opt.pgsr_loss.sample_num=32768 \
  gs.dataset.source_path=/absolute/path/to/scene \
  gs.dataset.model_path=/absolute/path/to/output \
  train.gs_epochs=30000
```

主要输出位于 `gs.dataset.model_path`：

- `cfg_args`：数据集与渲染参数快照；
- `config.yaml`：完整、已解析的 Hydra 配置，恢复自定义 renderer/loss 覆盖时应保留；
- `chkpnt<step>.pth`：Gaussian 参数和迭代数；
- `point_cloud/iteration_<step>/point_cloud.ply`：对应迭代的 Gaussian 点云；
- W&B 日志：仅当 `wandb.mode=online` 或 `offline` 时产生。

### 可选：PGSR 训练过程可视化

PGSR profile 提供默认关闭的训练期诊断图。开启方式：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=pgsr \
  gs.opt.pgsr_debug.enabled=true \
  gs.opt.pgsr_debug.interval=200 \
  train.gs_epochs=30000 \
  gs.dataset.source_path=/absolute/path/to/scene \
  gs.dataset.model_path=/absolute/path/to/output \
  wandb.mode=disabled
```

输出位于 `<output>/debug/%05d_<camera_name>.jpg`，每张是与 PGSR 源码一致的
2×4 拼图：

```text
GT | 渲染 RGB | 渲染法线 | Gaussian 平面距离
重投影权重 | 平面深度 | 深度法线 | 图像梯度权重
```

默认 `from_iter=null`，会跟随
`pgsr_loss.multi_view_from_iter=7000`；判断是严格大于阈值且满足
`step % interval == 0`，因此默认首张为第 7200 步。可用
`gs.opt.pgsr_debug.from_iter=10000` 延后开始，用
`gs.opt.pgsr_debug.output_dir=debug/pgsr` 修改模型目录内的相对输出路径，用
`gs.opt.pgsr_debug.jpeg_quality=90` 修改 JPEG 质量。

诊断图复用当前训练视角以及同一轮 loss 已选择的 neighbor 和重投影权重，不会
额外渲染参考视角，也不会改变邻视图随机选择。某一轮没有有效 neighbor 时不写
伪造面板。该功能与 `gs.pipe.debug` 完全不同：后者是 CUDA rasterizer 崩溃快照
开关，本集成仍明确禁止开启。配置会写入 `config.yaml`，恢复训练后调度继续使用
绝对 `gs_step`；但已经结束且未开启可视化的训练无法事后还原随机训练视角。

## 6. 恢复训练

`load.gs` 传输出目录，`load.gs_step` 传 checkpoint 中的迭代数；`train.gs_epochs` 是本次还要追加的步数。恢复已有 Gaussian 时必须关闭 RoMa 再初始化：

```bash
conda activate paintmesh
cd /path/to/EDGS

CUDA_VISIBLE_DEVICES=0 python train.py \
  gs=pgsr \
  gs.dataset.source_path=/absolute/path/to/scene \
  gs.dataset.model_path=/absolute/path/to/output \
  load.gs=/absolute/path/to/output \
  load.gs_step=7000 \
  train.gs_epochs=23000 \
  init_wC.use=false \
  wandb.mode=disabled
```

恢复时继续使用 `gs=pgsr`，保证 renderer 和 loss 配置与原训练一致。若之前覆盖过 loss 参数，也应在恢复命令中使用相同覆盖值。

## 7. 渲染、TSDF 网格与指标

根目录的 `render.py` 和 `metrics.py` 直接消费 `train.py` 的输出。渲染器以
`config.yaml` 为权威配置，因此会复用训练时真实的 renderer、图像目录和
`gs.dataset.resolution`；它不会使用旧版 `cfg_args` 中可能存在的占位值。

只渲染最新保存迭代的测试集并计算指标：

```bash
conda activate paintmesh
cd /path/to/EDGS

python render.py \
  -m /absolute/path/to/output \
  --iteration -1 \
  --skip-train

python metrics.py -m /absolute/path/to/output
```

`--iteration -1` 会从实际存在的
`point_cloud/iteration_<N>/point_cloud.ply` 中选择最大迭代。若数据集已移动，
使用 `--source-path /new/path/to/scene`；也可用 `--images` 或 `--resolution`
显式覆盖保存配置。渲染仍需要原数据集来重建相机和读取 GT。

完整 PGSR 渲染（训练集、测试集、深度、法线和训练视角 TSDF 网格）：

```bash
python render.py \
  -m /absolute/path/to/output \
  --iteration 30000 \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1 \
  --use-depth-filter
```

`--extract-mesh` 是显式开启的可选步骤，并会延迟导入 Open3D。若
`paintmesh` 中未安装 Open3D，可执行 `EDGS_INSTALL_DEMO=1 bash install.sh`；
不传 `--extract-mesh` 时仍会正常生成逐视图结果。使用 native renderer 时没有
可用于 TSDF 的 PGSR 平面深度；如需网格，可通过
`--renderer pgsr --extract-mesh` 显式使用 PGSR renderer。

输出结构如下：

```text
<output>/
├── train/ours_<iteration>/
│   ├── renders/
│   ├── gt/
│   ├── renders_depth/
│   ├── renders_normal/
│   └── render_manifest.json
├── test/ours_<iteration>/
│   ├── renders/
│   ├── gt/
│   ├── renders_depth/
│   ├── renders_normal/
│   └── render_manifest.json
├── mesh/ours_<iteration>/
│   ├── tsdf_fusion.ply
│   └── tsdf_fusion_post.ply
├── results.json
└── per_view.json
```

`renders_depth` 是逐视图归一化的彩色可视化，并不是可恢复的米制深度文件。
`metrics.py` 会严格配对 `test/ours_<iteration>/{renders,gt}` 的同名图片，流式
计算 PSNR、SSIM 和 PGSR/3DGS VGG LPIPS，再写场景均值和逐视图 JSON。可一次
评测多个输出：

```bash
python metrics.py -m /path/to/scene_a_output /path/to/scene_b_output
```

首次在新环境计算 LPIPS 时可能下载 VGG16 与 LPIPS 权重，离线机器应提前准备
Torch 缓存。

## 8. Renderer 输出契约

两个 backend 都提供 EDGS 训练所需的公共字段：

| 字段 | 含义 |
| --- | --- |
| `render` | RGB，形状 `[3, H, W]` |
| `viewspace_points` | 可求导的屏幕空间 Gaussian 中心 |
| `visibility_filter` | 兼容上游的可见性表示；native 为索引，PGSR 为布尔 mask |
| `visible_mask` | 两个 backend 均提供的、形状 `[N]` 的规范化布尔 mask |
| `radii` | 屏幕空间半径 |

PGSR backend 还会按请求返回：

| 字段 | 含义 |
| --- | --- |
| `viewspace_points_abs` | PGSR absolute-gradient 屏幕空间占位张量 |
| `out_observe` | rasterizer 的观测计数/权重输出 |
| `rendered_normal` | rasterize 得到的相机空间法线 `[3, H, W]` |
| `rendered_alpha` | plane map 的累积 alpha `[1, H, W]` |
| `rendered_distance` | Gaussian 局部平面距离 `[1, H, W]` |
| `plane_depth` | PGSR 平面几何深度 `[1, H, W]` |
| `depth_normal` | 从 `plane_depth` 重建的法线 `[3, H, W]` |

`rendered_normal`、`rendered_alpha`、`rendered_distance` 和 `plane_depth` 仅在 `return_plane=true` 时存在；`depth_normal` 还要求 `return_depth_normal=true`。EDGS native renderer 同时保留上游 `depth` 并提供语义更明确的 `inverse_depth` 别名；它们都是 inverse depth，而 PGSR 的 `plane_depth` 是几何深度，二者不能混用。

## 9. 已知限制

- 这是“EDGS 部件 + PGSR renderer/loss”的混合实现，不是完整 PGSR 复现。它保留 EDGS densification，不包含 PGSR 的 absolute-gradient densification 与 multi-view trim。
- 当前相机投影假设主点位于 `(W/2, H/2)`。EDGS 现有相机路径满足这一假设；非中心主点数据需要同时扩展 Camera 适配和 plane rasterizer CUDA 接口。
- 当前 `virtual_camera=false`，只使用真实训练相机邻居。虚拟相机分支尚未开放。
- PGSR plane rasterizer 不支持 EDGS 的 antialiasing，且所固定版本的 debug wrapper 不安全；混合模式要求 `pipe.antialiasing=false`、`pipe.debug=false`，代码会对错误配置显式报错。
- PGSR 的许可仅允许教育、研究和非营利用途；基于 PGSR 的修改要求开源，商业使用需要联系原作者。发布或部署前请阅读 `submodules/PGSR/LICENSE.md`。
- CUDA 扩展与编译它们时的 PyTorch、Python、CUDA ABI 绑定；更换任一版本后应重新编译。

## 10. 故障排查

### `ModuleNotFoundError: diff_plane_rasterization`

确认当前解释器属于 `paintmesh`，然后在同一环境重装扩展：

```bash
which python
python -m pip --version
python -m pip install --force-reinstall --no-deps -e \
  submodules/PGSR/submodules/diff-plane-rasterization
```

### `_C.so: undefined symbol`、`invalid device function` 或编译版本不匹配

通常是 PyTorch/CUDA ABI 或 GPU 架构发生变化。重新执行第 2 节检查，确保 `CUDA_HOME` 指向预期 toolkit，再在 `paintmesh` 中重编扩展。不要复制其他环境生成的 `.so` 文件。

### 导入了错误的 `scene` 或 `gaussian_renderer`

移除指向 `submodules/PGSR` 根目录的 `PYTHONPATH`/`sys.path` 修改。EDGS 只需要 `diff_plane_rasterization` 可导入，不应直接导入 PGSR 顶层 Python 包。

### 多视图损失始终为零

先确认训练集至少有两个有效视角。随后查看邻居筛选阈值；相机稀疏时可适当增大 `max_angle_deg`/`max_distance` 或降低 `min_distance`。没有合法邻居的相机会安全跳过 geometry 与 NCC 项。

### 显存不足

优先降低 `gs.opt.pgsr_loss.sample_num`，其次降低输入分辨率或将 `multi_view_from_iter` 延后。`patch_size=3` 表示 `7×7` patch；进一步增大该值会显著增加 NCC 内存。

### 出现 NaN/Inf

先运行单步 renderer smoke，并将单视图、多视图启用迭代设得很大以隔离问题；再分别开启 normal、geometry 和 NCC。确认 COLMAP 相机外参、scene scale 和深度均有效，并避免将 native inverse depth 传给 PGSR 几何函数。

### Resume 后 Gaussian 数量异常增加

恢复命令必须包含 `init_wC.use=false`。否则 checkpoint 加载后会再次执行 RoMa 对应点初始化。
