# PaintMesh：语义重建、目标移除与补全

[English](README.md) | **简体中文** | [实现原理](PRINCIPLES.zh-CN.md)

本目录提供三个稳定入口，把修改后的 EDGS-PGSR 与 Inpaint360GS 串成一条可恢复、带产物契约的流水线：

```text
run_seg
  EDGS-PGSR 3DGS + 基础 TSDF mesh
       -> 跨视角实例分割
       -> 语义 3DGS + 语义 mesh

run_remove
  语义 3DGS/mesh
       -> 删除指定实例
       -> removed 3DGS + 重建后的 removed mesh
       -> 30 个虚拟视角 + 交互跟踪 mask

run_inpaint
  removed 结果 + 30 个跟踪 mask
       -> LaMa RGB/depth 补全
       -> inpainted 3DGS + 重建后的 inpainted mesh
```

CropFormer/Inpaint360GS 生成的是当前场景内的实例 ID，而不是 `chair`、`table` 等类别名。ID `0` 是背景；mesh 语义中的 `65535` 默认表示支撑不足或预测歧义的未知区域。

## 1. 稳定入口与快速开始

请从仓库根目录运行无扩展名的入口；它们分别转发到同名的 `.sh` 实现文件：

```bash
cd /home/martin/code/gsagent

scripts/paintmesh/run_seg --help
scripts/paintmesh/run_remove --help
scripts/paintmesh/run_inpaint --help
```

以 `mip-nerf/360_v2/kitchen`、分辨率倍率 `8`、目标实例 `14` 为例，完整流水线是：

```bash
# 1. 语义重建：Stage 1..6
PAINTMESH_ENV=paintmesh GPU=0 \
  scripts/paintmesh/run_seg \
  "mip-nerf/360_v2" kitchen 8 1

# 2. 移除并启动交互 tracker：Stage 1..5
PAINTMESH_ENV=paintmesh GPU=0 END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1

# 3. RGB/depth/3DGS/mesh 补全：Stage 1..8
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

三个入口的依赖关系不可颠倒：`run_remove` 读取 `run_seg` 的冻结语义结果，`run_inpaint` 读取已经完成 Stage 4–5 的 `run_remove` 结果。

## 2. 环境、数据与 checkpoint

### 2.1 Conda 环境

EDGS、Inpaint360GS、PGSR mesh 重建、语义提升和产物校验默认都在 `paintmesh` 环境运行。LaMa 的 pinned Lightning/WebDataset 依赖默认保留在独立的 `lama` 环境：

```bash
PAINTMESH_ENV=paintmesh
LAMA_ENV=lama
```

如果已经把 LaMa 所需依赖完整安装并验证到 `paintmesh`，可设置 `LAMA_ENV=paintmesh`。脚本会为 EDGS 和 Inpaint360GS 子进程分别设置隔离的 `PYTHONPATH`，避免两个仓库中同名的 `scene`、`utils`、`gaussian_renderer` 相互污染。

### 2.2 数据目录

默认数据根目录是仓库下的 `data/`。一个场景至少应包含：

```text
data/<dataset>/<scene>/
├── images/
├── images_2/、images_4/ 或 images_8/   # 选择对应倍率时必须存在
└── sparse/0/                            # COLMAP 模型
```

`resolution` 只能是 `1`、`2`、`4`、`8`：

- `1`：2D 分割读取 `images/`；
- `2/4/8`：2D 分割读取 `images_<resolution>/`；
- EDGS 和 Inpaint360GS 同时接收 `--resolution`/`gs.dataset.resolution`，因此三个脚本必须使用同一个倍率。

### 2.3 Checkpoint

默认从仓库级 `ckpt/` 读取：

```text
ckpt/
├── CropFormer_hornet_3x_03823a.pth
├── sam_vit_b_01ec64.pth
├── R50_DeAOTL_PRE_YTB_DAV.pth
├── groundingdino_swint_ogc.pth
└── big-lama/
    ├── config.yaml
    └── models/best.ckpt
```

- `run_seg` Stage 2 使用 CropFormer checkpoint；
- `run_remove` Stage 5 使用 SAM、DeAOT 和 GroundingDINO checkpoint；
- `run_inpaint` Stage 3 使用 `big-lama`。

脚本会在子模块期望的位置创建受控软链接，但源 checkpoint 必须存在。可用 `CKPT_ROOT=/absolute/path` 或 `LAMA_MODEL_PATH=/absolute/path/to/big-lama` 覆盖默认位置。

## 3. 参数、恢复与单阶段执行语义

### 3.1 `run_seg`

```text
scripts/paintmesh/run_seg dataset_name scene resolution start_stage
```

| 位置 | 参数 | 默认值 | 说明 |
|---:|---|---|---|
| 1 | `dataset_name` | `inpaint360` | `DATA_ROOT` 下的相对数据集路径，可包含 `/` |
| 2 | `scene` | `doppelherz` | 单个场景目录名 |
| 3 | `resolution` | `2` | `1/2/4/8` |
| 4 | `start_stage` | `1` | 首个执行阶段，范围 `1..6` |

`run_seg` 没有 `END_STAGE`：`start_stage=N` 会验证前置产物，然后执行 `N..6`。例如从语义提升恢复：

```bash
scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 6
```

如需只调试其中一个阶段，请使用后文列出的底层命令；日常运行优先使用入口脚本，因为它还会执行哈希、点数、shape 和 manifest 校验。

四个位置参数也分别可由 `DATASET_NAME`、`SCENE`、`RESOLUTION`、`START_STAGE` 环境变量提供；显式位置参数优先。

### 3.2 `run_remove`

```text
scripts/paintmesh/run_remove \
    dataset_name scene resolution target_ids surrounding_ids start_stage
```

| 位置 | 参数 | 默认值 | 说明 |
|---:|---|---|---|
| 1 | `dataset_name` | `inpaint360` | 数据集相对路径 |
| 2 | `scene` | `doppelherz` | 场景名 |
| 3 | `resolution` | `2` | 必须与 `run_seg` 一致 |
| 4 | `target_ids` | 无，必填 | 正整数或逗号分隔列表，如 `14`、`14,21` |
| 5 | `surrounding_ids` | `none` | 临时移除、最终恢复的遮挡物 ID 列表 |
| 6 | `start_stage` | `1` | 首个执行阶段，范围 `1..5` |

目标和 surrounding 集合必须互斥，且不能包含背景 `0`。ID 会排序并规范化；默认目录名是 `target_<ids>` 或 `target_<ids>__surrounding_<ids>`。

六个位置参数也可分别由 `DATASET_NAME`、`SCENE`、`RESOLUTION`、`TARGET_IDS`、`SURROUNDING_IDS`、`START_STAGE` 环境变量提供；显式位置参数优先。

`START_STAGE` 由第 6 个位置参数给出，`END_STAGE` 由环境变量给出，脚本只执行闭区间 `[start_stage, END_STAGE]`，同时验证需要复用的前置产物：

```bash
# 只执行/恢复 Stage 3
END_STAGE=3 LAUNCH_REFINER=false \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 3

# 只生成 removed 3DGS 和 mesh，不启动 tracker
END_STAGE=3 LAUNCH_REFINER=false \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

### 3.3 `run_inpaint`

```text
scripts/paintmesh/run_inpaint \
    dataset_name scene resolution target_ids surrounding_ids start_stage
```

六个位置参数与 `run_remove` 含义一致，`start_stage` 范围为 `1..8`。默认 `END_STAGE=8`，执行区间也是 `[start_stage, END_STAGE]`。

```bash
# 从 RGB-D 融合恢复并执行 Stage 4..8
END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 4

# 只重新执行/验证 Stage 6
END_STAGE=6 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 6
```

即使 `start_stage>1`，Stage 1 的 workspace 准备器仍会以幂等方式运行，用来确认 removal、相机、mask 和当前 inpaint workspace 属于同一组输入。Stage 2/3 在后续阶段需要时也会复核对应 manifest；“跳过”不等于跳过校验。

## 4. 环境变量参考

路径型变量若使用相对路径，会相对于调用入口脚本时的当前目录解析；可复用任务建议传绝对路径。布尔值接受 `true/false`、`1/0`、`yes/no`、`on/off`。

### 4.1 三个脚本通用

| 变量 | 默认值 | 作用 |
|---|---|---|
| `PAINTMESH_ENV` | `paintmesh` | 主 Conda 环境 |
| `GPU` | 空 | 非空时传给 `CUDA_VISIBLE_DEVICES`，如 `GPU=0` |
| `CONDA_BIN` | `conda` | Conda 可执行文件 |
| `DATA_ROOT` | `<repo>/data` | 数据根目录 |
| `CKPT_ROOT` | `<repo>/ckpt` | checkpoint 根目录 |
| `OUTPUT_ROOT` | `<repo>/output` | 输出根目录 |
| `PIPELINE_ROOT` | `OUTPUT_ROOT/paintmesh/<dataset>/<scene>` | 当前场景流水线根目录 |
| `EDGS_IMAGES` | `images` | EDGS/Inpaint360GS 的原始图像参数 |
| `DISTILL_ITERATION` | `2000` | 语义蒸馏/移除源迭代 |
| `MAX_DEPTH` | `5.0` | TSDF 融合最大深度 |
| `VOXEL_SIZE` | `0.002` | TSDF voxel 大小 |
| `NUM_CLUSTERS` | `1` | mesh 连通簇数 |
| `USE_DEPTH_FILTER` | `false` | PGSR mesh 深度过滤 |
| `WRITE_COLORED_MESH` | `true` | 额外写出带实例色的完整 mesh |
| `MESH_NEIGHBORS` | `8` | 每个 mesh 顶点查询的高斯邻居数 |
| `MESH_CHUNK_SIZE` | `32768` | 语义提升 chunk 大小 |
| `MESH_WORKERS` | `-1` | worker 数，`-1` 为工具默认策略 |
| `MESH_OPACITY_MIN` | `0.01` | 参与提升的高斯最低 opacity |
| `MESH_SUPPORT_SIGMA` | `3.0` | 空间支撑尺度系数 |
| `MESH_NORMAL_POWER` | `2.0` | 法向一致性权重指数 |
| `MESH_MIN_CONFIDENCE` | `0.10` | 最低语义置信度 |
| `MESH_MIN_MARGIN` | `0.02` | 第一、第二类别的最低 margin |
| `MESH_UNKNOWN_ID` | `65535` | 未知语义 ID |

### 4.2 `run_seg` 专用

| 变量 | 默认值 | 作用 |
|---|---|---|
| `BASE_ITERATION` | `30000` | EDGS-PGSR 训练/mesh 迭代 |
| `NO_DENSIFY` | `true` | 传给 `train.no_densify` |
| `SEG_THRESHOLD` | `0.5` | CropFormer mask 阈值 |
| `ASSOCIATION_PATCH` | `16` | 跨视角关联 patch 大小 |
| `RENDER_VIDEO` | `false` | Stage 5 额外渲染视频 |
| `PGSR_DEBUG` | `false` | 保存 PGSR 2×4 训练诊断图 |
| `PGSR_DEBUG_INTERVAL` | `200` | 诊断图间隔 |
| `PGSR_DEBUG_FROM_ITER` | `auto` | `auto` 跟随多视图 loss 起点 |
| `PGSR_DEBUG_OUTPUT_DIR` | `debug` | 相对 `EDGS_MODEL_ROOT` 的目录 |
| `PGSR_DEBUG_JPEG_QUALITY` | `95` | JPEG 质量 `1..100` |
| `EDGS_MODEL_ROOT` | `PIPELINE_ROOT/edgs` | EDGS 输出或已有 EDGS run |
| `EDGS_BRIDGE_ROOT` | `PIPELINE_ROOT/edgs_bridge` | 只读 EDGS bridge |
| `SEMANTIC_GS_ROOT` | `PIPELINE_ROOT/semantic_3dgs` | 语义 3DGS 输出 |
| `SEMANTIC_MESH_ROOT` | `PIPELINE_ROOT/semantic_mesh` | 语义 mesh 输出 |
| `DISTILL_CONFIG` | `Inpaint360GS/config/object_distill/train_distill.json` | 蒸馏配置 |

### 4.3 `run_remove` 专用

| 变量 | 默认值 | 作用 |
|---|---|---|
| `END_STAGE` | `5` | 最后执行阶段 |
| `REMOVAL_THRESHOLD` | `0.7` | 目标高斯移除阈值 |
| `RUN_NAME` | 由 ID 生成 | removal 子目录名 |
| `REMOVAL_ROOT` | `PIPELINE_ROOT/removal/<RUN_NAME>` | removal run 根目录 |
| `EDGS_MODEL_ROOT` | `PIPELINE_ROOT/edgs` | 原始 EDGS 模型 |
| `EDGS_BRIDGE_ROOT` | `PIPELINE_ROOT/edgs_bridge` | EDGS bridge |
| `SEMANTIC_GS_ROOT` | `PIPELINE_ROOT/semantic_3dgs` | 输入语义 3DGS |
| `SEMANTIC_MESH_ROOT` | `PIPELINE_ROOT/semantic_mesh` | 输入语义 mesh |
| `RENDER_VIDEO` | `false` | Stage 2 removal 视频 |
| `RENDER_OBJECT_VIDEOS` | `false` | Stage 2 各对象视频 |
| `RENDER_REMOVAL_TRAIN` | `false` | Stage 2 train 诊断渲染 |
| `RENDER_REMOVAL_TEST` | `false` | Stage 2 test 诊断渲染 |
| `RENDER_EDGS_TEST` | `false` | Stage 3 是否渲染 EDGS test views |
| `WRITE_DEBUG_PLY` | `false` | Stage 2 上游大体积诊断 PLY |
| `LAUNCH_REFINER` | `true` | Stage 5 是否启动 Gradio tracker |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Stage 5 监听地址 |
| `GRADIO_SERVER_PORT` | `7860` | Stage 5 监听端口 |
| `GRADIO_SHARE` | `false` | 是否让 Gradio 创建公网分享链接 |

### 4.4 `run_inpaint` 专用

| 变量 | 默认值 | 作用 |
|---|---|---|
| `LAMA_ENV` | `lama` | LaMa Conda 环境 |
| `LAMA_MODEL_PATH` | `CKPT_ROOT/big-lama` | LaMa 模型目录 |
| `END_STAGE` | `8` | 最后执行阶段 |
| `FINETUNE_ITERATION` | 空 | 可选一致性断言；真正值读取 run-local inpaint config，默认通常为 `5000` |
| `FUSION_SEED_FRAME` | `4` | 训练 support PLY 帧，范围 `0..29` |
| `MASK_MIN_AREA` | `50` | 清理小 mask 的面积阈值 |
| `MASK_DILATION` | `10` | mask 膨胀像素 |
| `RECURSIVE_GUIDE` | `false` | LaMa color recursive guide |
| `RENDER_INPAINT_VIDEO` | `false` | Stage 5 诊断视频 |
| `RENDER_INPAINT_TRAIN` | `false` | Stage 5 train 诊断渲染 |
| `RENDER_INPAINT_TEST` | `false` | Stage 5 test 诊断渲染 |
| `RENDER_EDGS_TEST` | `false` | Stage 7 是否渲染 test views |
| `WRITE_HOLE_PLY` | `false` | Stage 4 是否写 30 个额外 hole PLY |
| `INPAINT_RUN_NAME` | `default` | inpaint 变体名 |
| `REMOVAL_ROOT` | 当前 ID 对应的 removal 目录 | 输入 removal run |
| `INPAINT_RUN_ROOT` | `REMOVAL_ROOT/inpaint/INPAINT_RUN_NAME` | inpaint run 根目录 |

## 5. 底层命令的公共上下文

下面各 Stage 的命令用于排错、复现实验和理解数据流。正常使用时仍建议执行三个入口脚本，因为入口还包含不可变性校验、原子 manifest、软链接边界检查和自动复用。

以下上下文对应文档示例 `target=14`、`surrounding=none`。先在仓库根目录执行：

```bash
export REPO_ROOT=/home/martin/code/gsagent
export DATASET_NAME="mip-nerf/360_v2"
export SCENE=kitchen
export RESOLUTION=8
export TARGET_IDS=14
export SURROUNDING_IDS=none
export GPU=0

export CONDA_BIN=conda
export PAINTMESH_ENV=paintmesh
export LAMA_ENV=lama
export DATA_ROOT="$REPO_ROOT/data"
export CKPT_ROOT="$REPO_ROOT/ckpt"
export OUTPUT_ROOT="$REPO_ROOT/output"
export EDGS_ROOT="$REPO_ROOT/submodules/EDGS"
export INPAINT_ROOT="$REPO_ROOT/submodules/Inpaint360GS"
export LAMA_ROOT="$INPAINT_ROOT/LaMa"

export BASE_ITERATION=30000
export DISTILL_ITERATION=2000
export FINETUNE_ITERATION=5000
export FUSION_SEED_FRAME=4
printf -v FUSION_SEED_NAME '%05d' "$FUSION_SEED_FRAME"

export SCENE_ROOT="$DATA_ROOT/$DATASET_NAME/$SCENE"
export PIPELINE_ROOT="$OUTPUT_ROOT/paintmesh/$DATASET_NAME/$SCENE"
export EDGS_MODEL_ROOT="$PIPELINE_ROOT/edgs"
export EDGS_BRIDGE_ROOT="$PIPELINE_ROOT/edgs_bridge"
export SEMANTIC_GS_ROOT="$PIPELINE_ROOT/semantic_3dgs"
export SEMANTIC_MESH_ROOT="$PIPELINE_ROOT/semantic_mesh"

export RUN_NAME=target_14
export REMOVAL_ROOT="$PIPELINE_ROOT/removal/$RUN_NAME"
export CONFIG_ROOT="$REMOVAL_ROOT/config"
export WORK_MODEL_ROOT="$REMOVAL_ROOT/work_model"
export REMOVED_GS_ROOT="$REMOVAL_ROOT/removed_3dgs"
export REMOVED_MESH_ROOT="$REMOVAL_ROOT/removed_mesh"
export TRACKER_ROOT="$REMOVAL_ROOT/tracker"
export REMOVAL_CONFIG="$CONFIG_ROOT/object_removal/$DATASET_NAME/$SCENE.json"
export INPAINT_CONFIG="$CONFIG_ROOT/object_inpaint/$DATASET_NAME/$SCENE.json"

export INPAINT_RUN_NAME=default
export INPAINT_RUN_ROOT="$REMOVAL_ROOT/inpaint/$INPAINT_RUN_NAME"
export INPAINT_WORK_MODEL="$INPAINT_RUN_ROOT/work_model"
export MANIFEST_ROOT="$INPAINT_RUN_ROOT/manifests"
export LAMA_INPUT_ROOT="$INPAINT_RUN_ROOT/lama/input"
export LAMA_OUTPUT_ROOT="$INPAINT_RUN_ROOT/lama/output"
export FUSED_ROOT="$INPAINT_RUN_ROOT/fused/mask"
export HOLE_ROOT="$INPAINT_RUN_ROOT/fused/hole"
export INPAINTED_GS_ROOT="$INPAINT_RUN_ROOT/inpainted_3dgs"
export INPAINTED_MESH_ROOT="$INPAINT_RUN_ROOT/inpainted_mesh"
export LAMA_MODEL_PATH="$CKPT_ROOT/big-lama"
```

定义与脚本一致的 Python 启动函数：

```bash
run_pm_python() {
  local workdir="$1"
  local pythonpath="$2"
  local env_name="$3"
  shift 3
  local -a process_env=("PYTHONPATH=$pythonpath")
  [[ -z "${GPU:-}" ]] || process_env+=("CUDA_VISIBLE_DEVICES=$GPU")
  (
    cd "$workdir"
    env "${process_env[@]}" \
      "$CONDA_BIN" run --no-capture-output -n "$env_name" python "$@"
  )
}

run_edgs_py() {
  run_pm_python "$EDGS_ROOT" "$EDGS_ROOT" "$PAINTMESH_ENV" "$@"
}

run_inpaint_py() {
  run_pm_python \
    "$INPAINT_ROOT" \
    "$INPAINT_ROOT:$INPAINT_ROOT/seg/detectron2" \
    "$PAINTMESH_ENV" "$@"
}

run_lama_py() {
  local lama_prefix
  lama_prefix="$("$CONDA_BIN" run --no-capture-output -n "$LAMA_ENV" \
    python -c 'import sys; print(sys.prefix)')"
  local -a process_env=(
    "PYTHONPATH=$LAMA_ROOT"
    "TORCH_HOME=$LAMA_ROOT"
    "LD_LIBRARY_PATH=$lama_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  )
  [[ -z "${GPU:-}" ]] || process_env+=("CUDA_VISIBLE_DEVICES=$GPU")
  (
    cd "$LAMA_ROOT"
    env "${process_env[@]}" \
      "$CONDA_BIN" run --no-capture-output -n "$LAMA_ENV" python "$@"
  )
}
```

## 6. `run_seg`：每个 Stage 的命令、输入和输出

完整命令：

```bash
PAINTMESH_ENV=paintmesh GPU=0 \
  scripts/paintmesh/run_seg \
  "mip-nerf/360_v2" kitchen 8 1
```

### Stage 1：EDGS-PGSR 重建、基础 TSDF mesh 与 bridge

默认训练命令：

```bash
run_edgs_py train.py \
  gs=pgsr \
  "train.gs_epochs=$BASE_ITERATION" \
  train.no_densify=true \
  "gs.opt.iterations=$BASE_ITERATION" \
  "gs.opt.position_lr_max_steps=$BASE_ITERATION" \
  gs.opt.pgsr_debug.enabled=false \
  gs.opt.pgsr_debug.interval=200 \
  gs.opt.pgsr_debug.output_dir=debug \
  gs.opt.pgsr_debug.jpeg_quality=95 \
  "gs.dataset.source_path=$SCENE_ROOT" \
  "gs.dataset.model_path=$EDGS_MODEL_ROOT" \
  gs.dataset.images=images \
  "gs.dataset.resolution=$RESOLUTION" \
  gs.dataset.eval=true \
  "gs.opt.save_iterations=[$BASE_ITERATION]" \
  init_wC.use=true \
  wandb.mode=disabled
```

启用上游 PGSR 风格 2×4 诊断图时，把 `gs.opt.pgsr_debug.enabled` 设为 `true`；若 `PGSR_DEBUG_FROM_ITER` 不是 `auto`，再追加 `gs.opt.pgsr_debug.from_iter=<N>`。入口脚本的等价用法是：

```bash
PGSR_DEBUG=true PGSR_DEBUG_INTERVAL=200 GPU=0 \
  scripts/paintmesh/run_seg \
  "mip-nerf/360_v2" kitchen 8 1
```

提取基础 mesh：

```bash
run_edgs_py render.py \
  -m "$EDGS_MODEL_ROOT" \
  --iteration "$BASE_ITERATION" \
  --renderer pgsr \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1
```

创建 Inpaint360GS 可读取、但不污染原 EDGS 输出的 bridge：

```bash
run_inpaint_py tools/build_edgs_bridge.py \
  --edgs-model "$EDGS_MODEL_ROOT" \
  --iteration "$BASE_ITERATION" \
  --output "$EDGS_BRIDGE_ROOT"
```

输入：`$SCENE_ROOT/images`、`$SCENE_ROOT/sparse/0`、EDGS/PGSR 配置。  
输出：

```text
edgs/config.yaml
edgs/point_cloud/iteration_30000/point_cloud.ply
edgs/mesh/ours_30000/tsdf_fusion_post.ply
edgs/debug/*.jpg                         # 仅 PGSR_DEBUG=true
edgs_bridge/bridge_manifest.json
```

若最终 checkpoint 已存在，脚本会独立复用训练结果；mesh 缺失时只重跑 mesh。已经完成的训练无法事后补生成随机训练视角的 debug 历史，需要换新的 `EDGS_MODEL_ROOT` 重训。

### Stage 2：逐视角 CropFormer mask

```bash
SEGMENTATION_IMAGE_FOLDER="images_$RESOLUTION"  # RESOLUTION=1 时改为 images

run_inpaint_py seg/raw_mask_sam.py \
  --dataset_path "$DATA_ROOT/$DATASET_NAME" \
  --scene_name "$SCENE" \
  --image_folder "$SEGMENTATION_IMAGE_FOLDER" \
  --method hqsam \
  --threshold 0.5
```

输入：对应分辨率图像、`CropFormer_hornet_3x_03823a.pth`。  
输出：`$SCENE_ROOT/raw_hqsam/*.png` 和 `$SCENE_ROOT/raw_hqsam_color/*.png`。

### Stage 3：跨视角实例关联与编号预览

```bash
run_inpaint_py seg/mask_associate.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$EDGS_BRIDGE_ROOT" \
  --iteration "$BASE_ITERATION" \
  --images images \
  --resolution "$RESOLUTION" \
  --mask_generator hqsam \
  --patch 16 \
  --eval

run_inpaint_py tools/add_label_num_hqsam.py \
  --source_path "$SCENE_ROOT" \
  --resolution "$RESOLUTION" \
  --mask_generator hqsam
```

输入：Stage 2 的 `raw_hqsam`、EDGS bridge、COLMAP 相机和原始图像。  
输出：

```text
<scene>/associated_hqsam/*.png
<scene>/associated_hqsam_color/*.png
<scene>/associated_hqsam/scene.json
<scene>/images_8_num/*                    # 示例 resolution=8
```

`images_<resolution>_num/` 是选择 `target_ids` 最直观的编号预览。

### Stage 4：对象特征蒸馏到高斯

```bash
run_inpaint_py seg/distillation.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$SEMANTIC_GS_ROOT" \
  --vanilla_3dgs_path "$EDGS_BRIDGE_ROOT" \
  --images images \
  --resolution "$RESOLUTION" \
  --object_path associated_hqsam \
  --config_file "$INPAINT_ROOT/config/object_distill/train_distill.json" \
  --test_iterations "$DISTILL_ITERATION" \
  --save_iterations "$DISTILL_ITERATION" \
  --checkpoint_iterations "$DISTILL_ITERATION" \
  --eval
```

输入：冻结的 EDGS bridge、关联 mask、`scene.json`、蒸馏配置。  
输出：

```text
semantic_3dgs/cfg_args
semantic_3dgs/point_cloud/iteration_2000/point_cloud.ply
semantic_3dgs/point_cloud/iteration_2000/classifier.pth
```

`classifier.pth` 把 16 维对象特征映射为当前场景的实例 ID，必须与该 PLY 和 `scene.json` 成组保存。

### Stage 5：渲染语义 3DGS

```bash
run_inpaint_py render.py \
  --model_path "$SEMANTIC_GS_ROOT" \
  --iteration "$DISTILL_ITERATION" \
  --skip_fused_ply
```

设置 `RENDER_VIDEO=true` 时追加 `--render_video`。  
输入：Stage 4 的语义 PLY、classifier、相机与图像。  
输出：`semantic_3dgs/{train,test}/ours_2000/{renders,gt,gt_objects_color,objects_pred,objects_pred_color,depth}/`，以及可选 `video/ours_2000/`。

### Stage 6：把高斯实例语义提升到基础 mesh

```bash
run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply "$SEMANTIC_GS_ROOT/point_cloud/iteration_$DISTILL_ITERATION/point_cloud.ply" \
  --classifier "$SEMANTIC_GS_ROOT/point_cloud/iteration_$DISTILL_ITERATION/classifier.pth" \
  --scene-info "$SCENE_ROOT/associated_hqsam/scene.json" \
  --mesh "$EDGS_MODEL_ROOT/mesh/ours_$BASE_ITERATION/tsdf_fusion_post.ply" \
  --output-dir "$SEMANTIC_MESH_ROOT" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply
```

输入：语义 PLY、classifier、`scene.json`、基础 TSDF mesh。  
输出：

```text
semantic_mesh/
├── semantic_mesh.ply
├── gaussian_instance_id.npy
├── gaussian_confidence.npy
├── vertex_instance_id.npy
├── vertex_confidence.npy
├── face_instance_id.npy
├── face_confidence.npy
├── palette.json
└── semantic_manifest.json
```

`.npy` 与 `semantic_manifest.json` 是权威语义结果；`semantic_mesh.ply` 只是 `WRITE_COLORED_MESH=true` 时生成的可视化副本。

## 7. `run_remove`：每个 Stage 的命令、输入和输出

完整命令：

```bash
PAINTMESH_ENV=paintmesh GPU=0 END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

带临时遮挡物的例子：

```bash
GPU=0 scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 "10,24" 1
```

### Stage 1：run-local 配置和隔离 workspace

```bash
run_inpaint_py tools/init_configs.py \
  --dataset_name "$DATASET_NAME" \
  --scene "$SCENE" \
  --target_id "$TARGET_IDS" \
  --target_surronding_id "$SURROUNDING_IDS" \
  --removal_thresh 0.7 \
  --output_root "$CONFIG_ROOT"

run_inpaint_py tools/prepare_removal_workspace.py \
  --semantic-model "$SEMANTIC_GS_ROOT" \
  --iteration "$DISTILL_ITERATION" \
  --bridge-manifest "$EDGS_BRIDGE_ROOT/bridge_manifest.json" \
  --semantic-manifest "$SEMANTIC_MESH_ROOT/semantic_manifest.json" \
  --output "$WORK_MODEL_ROOT"
```

注意：`--target_surronding_id` 的 `surronding` 拼写来自现有工具 API，命令中必须保持原样。workspace 准备器在恢复运行时也会执行，用于验证绑定关系。

输入：`semantic_3dgs`、EDGS bridge manifest、基础 semantic mesh manifest、目标/遮挡实例 ID。  
输出：

```text
removal/target_14/config/object_removal/.../kitchen.json
removal/target_14/config/object_inpaint/.../kitchen.json
removal/target_14/work_model/workspace_manifest.json
removal/target_14/work_model/point_cloud -> semantic_3dgs/point_cloud
```

### Stage 2：移除高斯并发布 removed 3DGS

默认不写诊断 train/test/video/大 PLY：

```bash
run_inpaint_py edit_object_removal.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$WORK_MODEL_ROOT" \
  --reference_model_path "$SEMANTIC_GS_ROOT" \
  --iteration "$DISTILL_ITERATION" \
  --resolution "$RESOLUTION" \
  --config_file "$REMOVAL_CONFIG" \
  --skip_train \
  --skip_test \
  --skip_debug_ply
```

没有 surrounding ID 时，发布输入为：

```bash
FINAL_REMOVED_PLY="$WORK_MODEL_ROOT/point_cloud_object_removal/iteration_$DISTILL_ITERATION/point_cloud.ply"
```

存在 surrounding ID 时，surrounding 对象在最终模型中恢复，发布输入改为 `iteration_${DISTILL_ITERATION}_removal_target/point_cloud.ply`。发布命令：

```bash
run_inpaint_py tools/publish_removed_edgs_model.py \
  --removed-ply "$FINAL_REMOVED_PLY" \
  --classifier "$SEMANTIC_GS_ROOT/point_cloud/iteration_$DISTILL_ITERATION/classifier.pth" \
  --edgs-config "$EDGS_MODEL_ROOT/config.yaml" \
  --cfg-args "$SEMANTIC_GS_ROOT/cfg_args" \
  --iteration "$DISTILL_ITERATION" \
  --target-ids "$TARGET_IDS" \
  --surrounding-ids "$SURROUNDING_IDS" \
  --bridge-manifest "$EDGS_BRIDGE_ROOT/bridge_manifest.json" \
  --semantic-manifest "$SEMANTIC_MESH_ROOT/semantic_manifest.json" \
  --removal-threshold 0.7 \
  --output "$REMOVED_GS_ROOT"
```

输入：Stage 1 workspace、语义 PLY/classifier、removal config。  
输出：

```text
work_model/point_cloud_object_removal/iteration_2000*/point_cloud.ply
removed_3dgs/config.yaml
removed_3dgs/cfg_args
removed_3dgs/model_manifest.json
removed_3dgs/point_cloud/iteration_2000/{point_cloud.ply,classifier.pth}
```

### Stage 3：从 removed 3DGS 重建并标注 mesh

```bash
run_edgs_py render.py \
  --model-path "$REMOVED_GS_ROOT" \
  --iteration "$DISTILL_ITERATION" \
  --renderer pgsr \
  --source-path "$SCENE_ROOT" \
  --images images \
  --resolution "$RESOLUTION" \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1 \
  --skip-test

run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply "$REMOVED_GS_ROOT/point_cloud/iteration_$DISTILL_ITERATION/point_cloud.ply" \
  --classifier "$REMOVED_GS_ROOT/point_cloud/iteration_$DISTILL_ITERATION/classifier.pth" \
  --scene-info "$SCENE_ROOT/associated_hqsam/scene.json" \
  --mesh "$REMOVED_GS_ROOT/mesh/ours_$DISTILL_ITERATION/tsdf_fusion_post.ply" \
  --output-dir "$REMOVED_MESH_ROOT" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply
```

输入：发布后的 removed 3DGS、原场景相机/图像、classifier、`scene.json`。  
输出：

```text
removed_3dgs/train/ours_2000/render_manifest.json
removed_3dgs/mesh/ours_2000/tsdf_fusion_post.ply
removed_3dgs/mesh/ours_2000/mesh_manifest.json
removed_mesh/geometry.ply -> 上述 TSDF mesh
removed_mesh/{gaussian,vertex,face}_{instance_id,confidence}.npy
removed_mesh/semantic_mesh.ply
removed_mesh/semantic_manifest.json
removal_manifest.json
```

`mesh_manifest.json`、`geometry.ply` 和最终 `removal_manifest.json` 由入口脚本的内置契约步骤写入/链接，不是额外的独立 CLI。因此手动运行以上两个 Python 命令后，仍应执行一次 `END_STAGE=3 ... start_stage=3` 来提交完整产物。

### Stage 4：生成 30 个虚拟视角和 tracker archive

```bash
run_inpaint_py tools/virtual_pose.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$WORK_MODEL_ROOT" \
  --iteration "$DISTILL_ITERATION" \
  --resolution "$RESOLUTION" \
  --config_file "$REMOVAL_CONFIG" \
  --tracker_archive "$TRACKER_ROOT/images.zip" \
  --camera_manifest "$TRACKER_ROOT/virtual_cameras.json"
```

输入：removal workspace/config、原始相机、目标移除前后的模型。  
输出：

```text
work_model/virtual/ours_2000/{renders,depth}/
work_model/virtual/ours_object_removal/iteration_2000/{renders,depth}/
tracker/images.zip                        # 严格 00000.png..00029.png
tracker/virtual_cameras.json              # 全精度相机姿态
tracker/tracking_session.json             # 初始为 in_progress
```

### Stage 5：交互式 Segment-and-Track-Anything

入口脚本实际使用下列进程环境启动 Gradio：

```bash
(
  cd "$INPAINT_ROOT/Segment-and-Track-Anything"
  env \
    "PYTHONPATH=$INPAINT_ROOT/Segment-and-Track-Anything" \
    "TRACKER_IMAGE_SEQUENCE=$TRACKER_ROOT/images.zip" \
    "TRACKER_ASSETS_ROOT=$TRACKER_ROOT/assets" \
    "TRACKING_RESULTS_ROOT=$TRACKER_ROOT/results" \
    GRADIO_SERVER_NAME=127.0.0.1 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=false \
    PYTHONUNBUFFERED=1 \
    "CUDA_VISIBLE_DEVICES=$GPU" \
    "$CONDA_BIN" run --no-capture-output \
      -n "$PAINTMESH_ENV" python -u app.py
)
```

输入：`tracker/images.zip` 和三个 tracker checkpoint。  
输出：

```text
tracker/results/images/images_masks/00000.png ... 00029.png
tracker/results/images/images_seg.mp4
tracker/results/images/images_seg.gif
tracker/tracking_session.json             # 校验后为 complete
```

在浏览器中：

1. 打开终端打印的 `http://127.0.0.1:7860`。
2. 进入 **Image-Seq type input**，选择/点击当前 run 的 `images.zip`，然后执行 extract。
3. 在首帧 `00000` 上标记目标并初始化 tracker。
4. 点击 **Start Tracking**，等待终端显示处理到 frame `29`，并确认 30 个基础 mask 都已写出。
5. 回到启动服务器的终端按 `Ctrl+C`。

`conda run` 有时会把正常的 `Ctrl+C` 翻译成退出状态 `1`。入口脚本只在精确的 `00000.png..00029.png`、尺寸和 archive 绑定关系全部通过后接受状态 `0/1/130`；若缺少任何 mask，仍会失败。`*_new.png` 是诊断文件，不计入最终 30 帧。

若界面已经导出全部 mask、但上次在提交 session 前中断，可执行：

```bash
END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 5
```

脚本会先验证并提交现有 mask；有效时不会重复打开界面。

## 8. `run_inpaint`：每个 Stage 的命令、输入和输出

完整命令：

```bash
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

### Stage 1：验证 removal/tracker 并准备 inpaint workspace

```bash
run_inpaint_py tools/prepare_inpaint_workspace.py \
  --removal-workspace "$WORK_MODEL_ROOT" \
  --removal-manifest "$REMOVAL_ROOT/removal_manifest.json" \
  --tracking-session "$TRACKER_ROOT/tracking_session.json" \
  --camera-manifest "$TRACKER_ROOT/virtual_cameras.json" \
  --tracking-masks "$TRACKER_ROOT/results/images/images_masks" \
  --source-iteration "$DISTILL_ITERATION" \
  --output "$INPAINT_WORK_MODEL"
```

输入：完整 removal manifest、完整 tracking session、30 个 mask、同一 archive 的相机 manifest。  
输出：`work_model/workspace_manifest.json`，以及指向 removal workspace/相机/mask 的受控相对链接。

### Stage 2：准备 run-local LaMa RGB/depth/mask

```bash
run_inpaint_py tools/prepare_paintmesh_lama_data.py prepare \
  --tracking-masks "$INPAINT_WORK_MODEL/tracking_masks" \
  --removed-rgb "$INPAINT_WORK_MODEL/virtual/ours_object_removal/iteration_$DISTILL_ITERATION/renders" \
  --removed-depth "$INPAINT_WORK_MODEL/virtual/ours_object_removal/iteration_$DISTILL_ITERATION/depth" \
  --reference-depth "$INPAINT_WORK_MODEL/virtual/ours_$DISTILL_ITERATION/depth" \
  --color-input "$LAMA_INPUT_ROOT/color" \
  --depth-input "$LAMA_INPUT_ROOT/depth" \
  --manifest "$MANIFEST_ROOT/lama_input_manifest.json" \
  --frames 30 \
  --min-area 50 \
  --dilation 10
```

输入：30 个跟踪 label mask、removed RGB/depth、移除前 reference depth。  
输出：

```text
lama/input/color/                         # RGB + LaMa mask
lama/input/depth/                         # depth + LaMa mask
manifests/lama_input_manifest.json
```

准备器会把索引 label 转为目标二值 mask，清理小连通域并执行膨胀；不会写入原始数据集。

### Stage 3：LaMa 完成 RGB 和 depth，并验证 30 帧

```bash
run_lama_py bin/predict_color.py \
  --input-dir "$LAMA_INPUT_ROOT/color" \
  --output-dir "$LAMA_OUTPUT_ROOT/color" \
  --model-path "$LAMA_MODEL_PATH"

run_lama_py bin/predict_depth.py \
  --input-dir "$LAMA_INPUT_ROOT/depth" \
  --output-dir "$LAMA_OUTPUT_ROOT/depth" \
  --model-path "$LAMA_MODEL_PATH"

run_inpaint_py tools/prepare_paintmesh_lama_data.py validate-output \
  --color-input "$LAMA_INPUT_ROOT/color" \
  --depth-input "$LAMA_INPUT_ROOT/depth" \
  --color-output "$LAMA_OUTPUT_ROOT/color" \
  --depth-output "$LAMA_OUTPUT_ROOT/depth" \
  --model-path "$LAMA_MODEL_PATH" \
  --input-manifest "$MANIFEST_ROOT/lama_input_manifest.json" \
  --manifest "$MANIFEST_ROOT/lama_completion_manifest.json" \
  --frames 30
```

`RECURSIVE_GUIDE=true` 时，color 预测和验证命令都追加 `--recursive-guide`。  
输入：Stage 2 的 LaMa 输入、`big-lama` checkpoint。  
输出：`lama/output/{color,depth}/` 和 `manifests/lama_completion_manifest.json`。验证器会检查帧集合、shape、有限深度、输入外像素保持和哈希。

### Stage 4：把已补全 RGB-D 反投影为 support PLY

```bash
run_inpaint_py edit_object_removal_plyfusion.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$INPAINT_WORK_MODEL" \
  --resolution "$RESOLUTION" \
  --iteration "$DISTILL_ITERATION" \
  --source_iteration "$DISTILL_ITERATION" \
  --config_file "$INPAINT_CONFIG" \
  --inpainted_color_dir "$LAMA_OUTPUT_ROOT/color" \
  --inpaint_mask_dir "$LAMA_INPUT_ROOT/color" \
  --completed_depth_dir "$LAMA_OUTPUT_ROOT/depth" \
  --fused_output_dir "$FUSED_ROOT" \
  --hole_output_dir "$HOLE_ROOT" \
  --camera_manifest "$TRACKER_ROOT/virtual_cameras.json" \
  --lama_manifest "$MANIFEST_ROOT/lama_completion_manifest.json" \
  --manifest "$MANIFEST_ROOT/fusion_manifest.json" \
  --skip_hole_ply
```

输入：补全 RGB/depth、对应 mask、全精度虚拟相机、LaMa completion manifest。  
输出：`fused/mask/00000.ply..00029.ply` 和 `manifests/fusion_manifest.json`。默认 `WRITE_HOLE_PLY=false`，所以追加 `--skip_hole_ply`；训练所需 support PLY 始终生成。

### Stage 5：优化补全后的对象感知 3DGS

```bash
INPAINTED_WORK_PLY="$INPAINT_WORK_MODEL/point_cloud_object_inpaint_virtual/iteration_$FINETUNE_ITERATION/point_cloud.ply"

run_inpaint_py edit_object_inpaint.py \
  --source_path "$SCENE_ROOT" \
  --model_path "$INPAINT_WORK_MODEL" \
  --images images \
  --resolution "$RESOLUTION" \
  --iteration "$DISTILL_ITERATION" \
  --source_iteration "$DISTILL_ITERATION" \
  --config_file "$INPAINT_CONFIG" \
  --supp_ply "$FUSED_ROOT/$FUSION_SEED_NAME.ply" \
  --fusion_dir "$FUSED_ROOT" \
  --fusion_seed_frame "$FUSION_SEED_FRAME" \
  --inpaint_output_ply "$INPAINTED_WORK_PLY" \
  --inpainted_color_dir "$LAMA_OUTPUT_ROOT/color" \
  --inpaint_mask_dir "$LAMA_INPUT_ROOT/color" \
  --camera_manifest "$TRACKER_ROOT/virtual_cameras.json" \
  --removal_source_iteration "$DISTILL_ITERATION" \
  --skip_surrounding_filter \
  --skip_train \
  --skip_test
```

设置 `RENDER_INPAINT_VIDEO=true` 时追加 `--render_video`；启用 train/test 诊断时分别去掉 `--skip_train`/`--skip_test`。  
输入：inpaint workspace、30 个 support PLY、seed PLY、LaMa color/mask、相机和 inpaint config。  
输出：`work_model/point_cloud_object_inpaint_virtual/iteration_5000/point_cloud.ply` 及可选诊断渲染。

### Stage 6：发布 EDGS 可加载的 inpainted 3DGS

```bash
run_inpaint_py tools/publish_inpainted_edgs_model.py \
  --inpainted-ply "$INPAINTED_WORK_PLY" \
  --classifier "$INPAINT_WORK_MODEL/point_cloud/iteration_$DISTILL_ITERATION/classifier.pth" \
  --edgs-config "$EDGS_MODEL_ROOT/config.yaml" \
  --cfg-args "$INPAINT_WORK_MODEL/cfg_args" \
  --source-iteration "$DISTILL_ITERATION" \
  --output-iteration "$FINETUNE_ITERATION" \
  --target-ids "$TARGET_IDS" \
  --surrounding-ids "$SURROUNDING_IDS" \
  --removed-model-manifest "$REMOVED_GS_ROOT/model_manifest.json" \
  --removal-manifest "$REMOVAL_ROOT/removal_manifest.json" \
  --workspace-manifest "$INPAINT_WORK_MODEL/workspace_manifest.json" \
  --tracking-session "$TRACKER_ROOT/tracking_session.json" \
  --lama-manifest "$MANIFEST_ROOT/lama_completion_manifest.json" \
  --fusion-manifest "$MANIFEST_ROOT/fusion_manifest.json" \
  --fusion-seed-frame "$FUSION_SEED_FRAME" \
  --inpaint-config "$INPAINT_CONFIG" \
  --output "$INPAINTED_GS_ROOT"
```

实际脚本中的 `--edgs-config` 是从 removed model manifest 的 `inputs.edgs_config.path` 解析并严格校验的；默认流水线下等价于上面的 `$EDGS_MODEL_ROOT/config.yaml`。

输入：Stage 5 PLY、classifier、EDGS 配置，以及 removal/workspace/tracking/LaMa/fusion 全部上游 manifest。  
输出：

```text
inpainted_3dgs/config.yaml
inpainted_3dgs/cfg_args
inpainted_3dgs/model_manifest.json
inpainted_3dgs/point_cloud/iteration_5000/point_cloud.ply
inpainted_3dgs/point_cloud/iteration_5000/classifier.pth
```

重要：发布后的 `point_cloud.ply` 是对 Stage 5 工作 PLY 的**独立普通文件复制**，不是符号链接。发布器使用临时文件、`fsync` 和原子替换，并在 `model_manifest.json` 记录 `point_cloud_storage.type=regular_copy` 与 SHA-256。入口脚本还用 `require_regular_file` 拒绝 symlink。`classifier.pth` 仍可作为受控相对链接复用，因为它是冻结的语义分类器。

### Stage 7：PGSR 渲染与 inpainted TSDF mesh

```bash
run_edgs_py render.py \
  --model-path "$INPAINTED_GS_ROOT" \
  --source-path "$SCENE_ROOT" \
  --images images \
  --resolution "$RESOLUTION" \
  --iteration "$FINETUNE_ITERATION" \
  --renderer pgsr \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1 \
  --skip-test

run_inpaint_py tools/publish_inpaint_mesh.py \
  --model-manifest "$INPAINTED_GS_ROOT/model_manifest.json" \
  --gaussian-ply "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/point_cloud.ply" \
  --mesh "$INPAINTED_GS_ROOT/mesh/ours_$FINETUNE_ITERATION/tsdf_fusion_post.ply" \
  --train-render-manifest "$INPAINTED_GS_ROOT/train/ours_$FINETUNE_ITERATION/render_manifest.json" \
  --output "$INPAINTED_GS_ROOT/mesh/ours_$FINETUNE_ITERATION/mesh_manifest.json" \
  --iteration "$FINETUNE_ITERATION" \
  --source-path "$SCENE_ROOT" \
  --images images \
  --resolution "$RESOLUTION" \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1
```

`RENDER_EDGS_TEST=true` 时去掉 render 命令的 `--skip-test`，并给 publisher 追加 `--test-render-manifest <.../test/.../render_manifest.json>`；`USE_DEPTH_FILTER=true` 时两个命令都追加 `--use-depth-filter`。  
输入：Stage 6 发布模型、原场景相机/图像。  
输出：train/test render、`mesh/ours_5000/tsdf_fusion_post.ply` 和 `mesh_manifest.json`。

### Stage 8：语义提升并提交最终 inpaint 结果

```bash
run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/point_cloud.ply" \
  --classifier "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/classifier.pth" \
  --scene-info "$SCENE_ROOT/associated_hqsam/scene.json" \
  --mesh "$INPAINTED_GS_ROOT/mesh/ours_$FINETUNE_ITERATION/tsdf_fusion_post.ply" \
  --output-dir "$INPAINTED_MESH_ROOT" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply

run_inpaint_py tools/finalize_inpaint_result.py \
  --model-manifest "$INPAINTED_GS_ROOT/model_manifest.json" \
  --mesh-manifest "$INPAINTED_GS_ROOT/mesh/ours_$FINETUNE_ITERATION/mesh_manifest.json" \
  --semantic-manifest "$INPAINTED_MESH_ROOT/semantic_manifest.json" \
  --removal-manifest "$REMOVAL_ROOT/removal_manifest.json" \
  --workspace-manifest "$INPAINT_WORK_MODEL/workspace_manifest.json" \
  --lama-manifest "$MANIFEST_ROOT/lama_completion_manifest.json" \
  --fusion-manifest "$MANIFEST_ROOT/fusion_manifest.json" \
  --gaussian-ply "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/point_cloud.ply" \
  --mesh "$INPAINTED_GS_ROOT/mesh/ours_$FINETUNE_ITERATION/tsdf_fusion_post.ply" \
  --geometry "$INPAINTED_MESH_ROOT/geometry.ply" \
  --output "$INPAINT_RUN_ROOT/inpaint_manifest.json"
```

入口脚本会在 finalize 前创建：

```text
inpainted_mesh/geometry.ply -> ../inpainted_3dgs/mesh/ours_5000/tsdf_fusion_post.ply
```

输入：发布后的 inpainted PLY/classifier、新 TSDF mesh、所有上游 manifest。  
输出：`inpainted_mesh` 全套语义数组/可视化 mesh，以及最终原子完成标记 `inpaint_manifest.json`。

## 9. 默认输出树

### 9.1 `run_seg`

```text
output/paintmesh/<dataset>/<scene>/
├── edgs/
│   ├── config.yaml
│   ├── point_cloud/iteration_<B>/point_cloud.ply
│   └── mesh/ours_<B>/tsdf_fusion_post.ply
├── edgs_bridge/
│   └── bridge_manifest.json
├── semantic_3dgs/
│   ├── cfg_args
│   └── point_cloud/iteration_<D>/
│       ├── point_cloud.ply
│       └── classifier.pth
└── semantic_mesh/
    ├── semantic_mesh.ply
    ├── gaussian_instance_id.npy
    ├── gaussian_confidence.npy
    ├── vertex_instance_id.npy
    ├── vertex_confidence.npy
    ├── face_instance_id.npy
    ├── face_confidence.npy
    ├── palette.json
    └── semantic_manifest.json
```

### 9.2 `run_remove`

```text
output/paintmesh/<dataset>/<scene>/removal/target_<ids>/
├── config/
├── work_model/
│   ├── workspace_manifest.json
│   ├── point_cloud -> ../../../semantic_3dgs/point_cloud
│   ├── point_cloud_object_removal/
│   └── virtual/
├── removed_3dgs/
│   ├── config.yaml
│   ├── cfg_args
│   ├── model_manifest.json
│   ├── point_cloud/iteration_<D>/
│   │   ├── point_cloud.ply
│   │   └── classifier.pth
│   └── mesh/ours_<D>/
│       ├── tsdf_fusion_post.ply
│       └── mesh_manifest.json
├── removed_mesh/
│   ├── geometry.ply -> ../removed_3dgs/mesh/ours_<D>/tsdf_fusion_post.ply
│   ├── semantic_mesh.ply
│   ├── gaussian_instance_id.npy
│   ├── vertex_instance_id.npy
│   ├── face_instance_id.npy
│   └── semantic_manifest.json
├── tracker/
│   ├── images.zip
│   ├── virtual_cameras.json
│   ├── assets/
│   ├── results/images/images_masks/
│   └── tracking_session.json
└── removal_manifest.json
```

### 9.3 `run_inpaint`

```text
output/paintmesh/<dataset>/<scene>/removal/target_<ids>/inpaint/default/
├── work_model/
│   ├── workspace_manifest.json
│   ├── point_cloud -> removal 的语义 checkpoint
│   ├── point_cloud_object_removal -> removal workspace
│   └── virtual/cameras.json -> tracker/virtual_cameras.json
├── lama/
│   ├── input/{color,depth}/
│   └── output/{color,depth}/
├── fused/
│   └── mask/00000.ply ... 00029.ply
├── inpainted_3dgs/
│   ├── config.yaml
│   ├── cfg_args
│   ├── model_manifest.json
│   ├── point_cloud/iteration_<F>/
│   │   ├── point_cloud.ply              # 独立普通文件复制，不是 symlink
│   │   └── classifier.pth
│   └── mesh/ours_<F>/
│       ├── tsdf_fusion_post.ply
│       └── mesh_manifest.json
├── inpainted_mesh/
│   ├── geometry.ply -> ../inpainted_3dgs/mesh/ours_<F>/tsdf_fusion_post.ply
│   ├── semantic_mesh.ply
│   ├── gaussian_instance_id.npy
│   ├── gaussian_confidence.npy
│   ├── vertex_instance_id.npy
│   ├── vertex_confidence.npy
│   ├── face_instance_id.npy
│   ├── face_confidence.npy
│   ├── palette.json
│   └── semantic_manifest.json
├── manifests/
│   ├── lama_input_manifest.json
│   ├── lama_completion_manifest.json
│   └── fusion_manifest.json
└── inpaint_manifest.json
```

`<B>` 是 `BASE_ITERATION`，`<D>` 是 `DISTILL_ITERATION`，`<F>` 来自 run-local `object_inpaint` 配置中的 `finetune_iteration`。

## 10. 复用、不可变性和完全重跑

### 10.1 正常恢复

- 使用同一输出目录、同一输入和同一参数时，完整 manifest 允许安全复用。
- `start_stage` 只决定首个计算阶段；前置文件仍会被校验。
- `run_remove`、`run_inpaint` 用 `END_STAGE` 限制最后阶段；必须满足 `start_stage <= END_STAGE`。
- 不要只复制一个 PLY 然后搭配另一 run 的 `classifier.pth`。实例通道是场景内局部语义，manifest 会绑定 PLY、分类器、相机、ID、参数和上游 artifact ID。

### 10.2 参数变体

已发布目录对身份参数和输入是不可变的。改变 removal 阈值、目标 ID、mask 膨胀、融合 seed、渲染诊断开关或 mesh 提升参数时，优先创建新 run：

```bash
# removal 参数变体
RUN_NAME=target_14_t080 REMOVAL_THRESHOLD=0.8 END_STAGE=3 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1

# inpaint 参数变体
INPAINT_RUN_NAME=dilate_16 MASK_DILATION=16 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

如果输出目录提示 `belongs to different parameters or inputs`，含义是该目录已经绑定另一组哈希/参数，而不是普通的“文件存在”。不要只删除 `model_manifest.json`：其余阶段 manifest 和受控链接仍属于旧 run。

### 10.3 完全重新运行并使用原目录名

三个入口没有隐式覆盖现有不可变产物的 `--force`。最安全做法是先把完整 run 目录移动为备份，再从 Stage 1 运行；这样新结果仍写到原目录名，同时旧结果可恢复。

完全重跑 inpaint：

```bash
INPAINT_DIR="$REPO_ROOT/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14/inpaint/default"
BACKUP_DIR="${INPAINT_DIR}.backup-$(date +%Y%m%d-%H%M%S)"
mv -- "$INPAINT_DIR" "$BACKUP_DIR"

PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

完全重跑 removal（会重新生成虚拟视角和 tracker session）：

```bash
REMOVAL_DIR="$REPO_ROOT/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14"
BACKUP_DIR="${REMOVAL_DIR}.backup-$(date +%Y%m%d-%H%M%S)"
mv -- "$REMOVAL_DIR" "$BACKUP_DIR"

PAINTMESH_ENV=paintmesh GPU=0 END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

完全重跑整个场景（包括 EDGS 训练）时，对完整场景 pipeline 做备份：

```bash
SCENE_PIPELINE="$REPO_ROOT/output/paintmesh/mip-nerf/360_v2/kitchen"
BACKUP_DIR="${SCENE_PIPELINE}.backup-$(date +%Y%m%d-%H%M%S)"
mv -- "$SCENE_PIPELINE" "$BACKUP_DIR"

PAINTMESH_ENV=paintmesh GPU=0 \
  scripts/paintmesh/run_seg \
  "mip-nerf/360_v2" kitchen 8 1
```

移动目录前确保没有对应训练、渲染或 Gradio 进程仍在写入。若确认不需要备份，可自行删除精确的 run 目录后重跑，但不要对仓库根、`output/` 根或含未解析变量的路径执行递归删除。

## 11. 推荐的日常检查

```bash
# 发布后的 inpaint PLY 必须是普通文件，而非软链接
test -f "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/point_cloud.ply"
test ! -L "$INPAINTED_GS_ROOT/point_cloud/iteration_$FINETUNE_ITERATION/point_cloud.ply"

# 查看三个最终完成标记
python -m json.tool "$SEMANTIC_MESH_ROOT/semantic_manifest.json" >/dev/null
python -m json.tool "$REMOVAL_ROOT/removal_manifest.json" >/dev/null
python -m json.tool "$INPAINT_RUN_ROOT/inpaint_manifest.json" >/dev/null

# 查看可用空间；彩色 TSDF mesh 可能很大
df -h "$OUTPUT_ROOT"
```

当 `WRITE_COLORED_MESH=true` 时会额外写一个完整 mesh。高分辨率场景可能占用数百 MB 到数 GB；用于程序读取时，以语义 `.npy` 和 manifest 为准。
