# Inpaint360GS 表 1 的实验与评估

论文及补充材料：https://arxiv.org/html/2511.06457v1

官方指标实现：https://github.com/dfki-av/Inpaint360GS/blob/main/tools/metrics_fid_masked.py

## 协议

- 表 1 使用主数据集的 11 个场景：bag、car、cone_red、cone_yellow、cube、doppelherz、fruits、garden_toys、redbull、toys、truck。`others/` 中的场景不加入该表。
- 论文 §4.1 使用 1/4 分辨率。补充材料 B 的重建设置使用非 `test_*` 图像的 7/8 训练，1/8 间隔留出。
- `test_*` 是移除目标后的真实照片，仅用于最终修复评估；它们不是上述 1/8 重建留出图。训练、分割、LaMa 修复与初始化不能使用其 RGB。评估使用它们的 COLMAP 相机。
- 使用作者提供的 `unseen_mask/test_*.png`，白色为待评估的 NBS 区域。不能替换为模型预测 mask 或虚拟视图的 tracker mask。
- 必须移除与真实 GT 对应的全部目标物体；多物体场景不能只删除其中一个。对象 ID 来自每次训练的分割结果，不能跨模型复用 ID。
- 使用官方已经清理测试区域的稀疏点云，不重新把 object-free GT 的点云加入初始化。
- 对每个方法固定相同测试视图、分辨率、官方 mask 与指标实现。调参使用训练/验证数据，不依据 object-free GT 分数选择超参数。

作者发布代码的 masked 指标定义为：将预测和 GT 的 mask 外像素同时置零，然后在整张画布上调用 PSNR、SSIM 和 LPIPS。它不是按 mask 内有效像素数归一化的误差，也不裁剪 mask 外接框。小 mask 会使 masked PSNR/SSIM 很高，因此必须同时看 full-image 指标。

LPIPS 使用作者的 `lpipsPyTorch`、VGG、v0.1，输入保持其代码的 [0,1] 约定。FID 使用 `pytorch_fid` 的 2048 维 Inception 特征，对完整渲染与完整 GT 图像集合计算。默认不额外裁剪/缩放；mask 用最近邻缩放到渲染尺寸。

作者代码对 PSNR/SSIM/LPIPS 汇总所有测试帧，对 FID 先逐场景计算再取平均。评估入口同时输出这个汇总和场景等权平均，便于核对。FID 不是逐图计算，也不能只用一张图评估。

## 评估单个 bag 模型

当前这批模型以分辨率 4、`EVAL=false` 训练，removal run 为 `table1_hemi90`。训练与评估分辨率均为 4，但训练使用全部非 `test_*` 图像，仍需标注其与论文 7/8 训练划分的差异。

```bash
cd /home/member/wyc/gsagent

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/eval_inpaint360.py run \
  --model-path output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP/inpainted_3dgs \
  --source-path data/Inpaint360GS/inpaint360/bag \
  --resolution 4 --iteration 5000 --batch-size 8
```

输出位于模型的 `evaluation/inpaint360_r4/`：`renders/`、`gt/`、`masks/`、`render_manifest.json`、`results.json`。只渲染真实 `test_*` 相机，模型及其 config/训练 split 不被修改。不读取 90 个虚拟视图作为 GT。

`results.json` 包含七项平均值、每视图指标、训练/评估分辨率、训练划分设置、checkpoint/config/GT/mask 的 SHA256。已有匹配渲染会复用；`--metrics-only` 仅重新计算指标，`--render-only` 仅渲染。修改到 `--resolution 2` 会创建独立的 `evaluation/inpaint360_r2/`。

## 按论文的数据划分重跑

分辨率改变会重写 `raw_hqsam` 和 `associated_hqsam`，所以使用独立的数据工作目录。下例仅链接原始图像、COLMAP 与官方 GT mask，生成的分割 mask 位于新目录。

首次建立工作目录（不在这里自动训练）：

```bash
cd /home/member/wyc/gsagent
export BENCHMARK_DATA="$PWD/data_processed/table1_r4"
export BENCHMARK_OUTPUT="$PWD/output/table1_r4_eval"

for source_scene in "$PWD"/data/Inpaint360GS/inpaint360/*; do
  [[ -d "$source_scene/images" ]] || continue
  scene_name="${source_scene##*/}"
  scene_root="$BENCHMARK_DATA/Inpaint360GS/inpaint360/$scene_name"
  mkdir -p "$scene_root"
  for directory in images images_4 sparse unseen_mask; do
    [[ -d "$source_scene/$directory" ]] || { echo "Missing $source_scene/$directory"; break; }
    [[ -e "$scene_root/$directory" ]] || ln -s "$source_scene/$directory" "$scene_root/$directory"
  done
done

DATA_ROOT="$BENCHMARK_DATA" OUTPUT_ROOT="$BENCHMARK_OUTPUT" \
  PAINTMESH_ENV=paintmesh GPU=1 EVAL=true \
  scripts/paintmesh/run_seg Inpaint360GS/inpaint360 all 4 1
```

每个场景从新目录的 `images_4_num/` 选择对应 GT 中被移除的目标 ID，再执行 removal 与 inpaint。下面 `target_ids` 必须改成新模型的真实 ID，不能直接使用旧模型的 36。

```bash
scene=bag
read -r -p "新模型中要移除的目标 ID（多个用逗号分隔）：" target_ids
surrounding_ids=none
removal_run=table1_hemi90

DATA_ROOT="$BENCHMARK_DATA" OUTPUT_ROOT="$BENCHMARK_OUTPUT" \
  PAINTMESH_ENV=paintmesh GPU=1 \
  VIRTUAL_RENDERER=edgs-pgsr VIRTUAL_CAMERA_PATH=hemisphere \
  VIRTUAL_CAMERA_COUNT=90 VIRTUAL_HEMISPHERE_MAX_ELEVATION_DEG=85 \
  RUN_NAME="$removal_run" END_STAGE=5 \
  bash scripts/paintmesh/run_remove.sh \
  Inpaint360GS/inpaint360 "$scene" 4 "$target_ids" "$surrounding_ids" 1

# 完成 mask 跟踪并点击 Finish 后执行。
removal_root="$BENCHMARK_OUTPUT/paintmesh/Inpaint360GS/inpaint360/$scene/removal/$removal_run"

DATA_ROOT="$BENCHMARK_DATA" OUTPUT_ROOT="$BENCHMARK_OUTPUT" \
  PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=1 \
  REMOVAL_ROOT="$removal_root" \
  INPAINT_RUN_ROOT="$removal_root/inpaint/edgs_rgbdn_joint_woDP" \
  INPAINT_RUN_NAME=edgs_rgbdn_joint_woDP INPAINT_PIPELINE=edgs-pgsr \
  VIRTUAL_CAMERA_COUNT=90 \
  EDGS_INIT_USE_DEPTH=false EDGS_INIT_USE_NORMAL=false \
  EDGS_TRAIN_USE_DEPTH=true EDGS_TRAIN_USE_NORMAL=true \
  EDGS_INPAINT_ITERATIONS=5000 EDGS_GEOMETRY_FROM_ITER=100 EDGS_GEOMETRY_RAMP_ITERS=2000 \
  LOCAL_GEOMETRY_REFINE=false SUPPORT_DENSITY_MODE=legacy END_STAGE=8 \
  bash scripts/paintmesh/run_inpaint.sh \
  Inpaint360GS/inpaint360 "$scene" 4 "$target_ids" "$surrounding_ids" 1

CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/eval_inpaint360.py run \
  --model-path "$removal_root/inpaint/edgs_rgbdn_joint_woDP/inpainted_3dgs" \
  --source-path "$BENCHMARK_DATA/Inpaint360GS/inpaint360/$scene" \
  --resolution 4 --iteration 5000
```

这组实验应标为我们自己的 EDGS-PGSR 方法：与论文原方法的基础重建、分割器、半球视角、法线监督和训练预算不同。论文原方法的补充材料使用圆形虚拟相机、深度颜色融合初始化、2000 步 3D 修复；原方法 baseline 要使用作者的实现或公开结果，不用我们的 EDGS 路线冒充。

## 11 场景汇总与其他方法

当前已完成的八场景实验（`bag`、`car`、`cone_red`、`cone_yellow`、`cube`、`doppelherz`、`redbull`、`truck`）可统一评估：

```bash
cd /home/member/wyc/gsagent
PYTHONNOUSERSITE=1 conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/eval_inpaint360_batch.py --gpu 0
```

该入口依次评估各场景的 `removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP/inpainted_3dgs`，使用 iteration 5000、评估分辨率 4。每场景输出仍位于模型的 `evaluation/inpaint360_r4/`。逐场景日志写入 `logs/eval_inpaint360_gpu0_<时间>_<PID>/`；全部完成后，汇总的 `summary.json`、`summary.csv`、`summary.md` 写入 `output/evaluation/inpaint360_table1_hemi90_edgs_rgbdn_joint_woDP_r4_8scenes/`。`--dry-run` 只检查输入并显示计划。

八场景汇总明确记录未包含 `fruits`、`garden_toys`、`toys`，不能当作完整的 11 场景表 1。实际训练划分会从模型配置读取并记录；分辨率 4、`EVAL=false` 的训练结果应标注其与论文 7/8 划分的差异。

每个方法/变体单独汇总，每个场景只选一个最终模型。以下示例适用于上面的统一新路径：

```bash
conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/eval_inpaint360.py aggregate \
  --results "$BENCHMARK_OUTPUT"/paintmesh/Inpaint360GS/inpaint360/*/removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP/inpainted_3dgs/evaluation/inpaint360_r4/results.json \
  --output "$BENCHMARK_OUTPUT/table1_edgs_rgbdn_joint_woDP.json"
```

正式表 1 对比必须确认 `scene_count=11`。单场景 bag 不能与论文的 11 场景平均值直接比较，也不加入 `others/`。未完成场景要单独记录，不能悄悄漏掉。

其他方法已经导出的图像也可用同一套指标计算，render/GT/官方 mask 的文件名 stem 必须一致、保留 `test_` 前缀：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/eval_inpaint360.py metrics \
  --renders-dir /path/to/baseline/renders \
  --gt-dir /path/to/baseline/gt \
  --mask-dir data/Inpaint360GS/inpaint360/bag/unseen_mask \
  --output /path/to/baseline/bag_results.json
```

该入口不会猜测别人导出的图像分辨率；必须自行确认与主实验同尺寸。最稳妥的比较是在本地用相同 evaluator 重新评估所有 baseline，并保留官方原始数值作为参考列。

## LaMa mask 扩张对照：bag、truck

检查当前 `table1_hemi90` 的修复伪影时，可单独比较 RGB 补全：

```bash
cd /home/member/wyc/gsagent
PYTHONNOUSERSITE=1 conda run --no-capture-output -n paintmesh \
  python scripts/paintmesh/compare_lama_dilation.py --gpu 0
```

固定原来的分辨率 4、90 个虚拟相机、移除后 RGB、Big-LaMa 权重、细化参数和 `RECURSIVE_GUIDE=false`，只比较 `MASK_DILATION=10,20,30`。三组均重新推理；使用现有 PaintMesh 的椭圆膨胀核，不同时改变核形状。原始跟踪 mask 按索引非零读取，避免调色板 PNG 的灰度转换改变目标区域。

默认结果写入 `output/evaluation/lama_dilation_bag_truck_<时间>/`，各组日志写入 `logs/<同名实验>/`。`--output-root` 可以指定一个尚不存在的目录，`--scenes bag` 可只跑单场景，`--dry-run` 检查全部原始图像/mask 哈希以及 dilation=10 与原流水线的一致性，不写结果。

输出包括 `comparison_overview.png`、包含全部 180 个视角的 `index.html`、逐帧 `mask_coverage.csv`、`report.md` 和记录文件哈希及进度的 `experiment.json`。每场景每组保留 `lama/input/color` 和 `lama/output/color`。完成前后检查输出帧名、尺寸、mask 外像素保持及三组 mask 的包含关系。

这是独立的 RGB 诊断实验，不修改已有修复结果，不重跑深度、法线或 3D 训练，也不生成可冒充完整流水线产物的 completion manifest。阴影无人工标注，因此 mask 面积不能解释为阴影覆盖率；通过带三组轮廓的输入图检查残影、阴影是否仍延伸到 mask 外。虚拟相机没有配对 GT，不用它们报告表 1 的 PSNR/SSIM。

选定扩张参数后，覆盖八场景原 inpaint 路径、重跑 Stage 1–8：

```bash
GPU=1 MASK_DILATION=30 RECURSIVE_GUIDE=false OVERWRITE_INPAINT=true \
  bash scripts/paintmesh/run_inpaint_table1_batch.sh all
```

该命令在每个场景开始前删除其 `removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP` 并重新生成 RGB/深度/法线补全、初始化、联合训练与最终模型；不删除 removal、跟踪或语义重建。每场景轮到时才替换，失败会停止后续场景。`OVERWRITE_INPAINT` 默认关闭；加 `--dry-run` 可先查看八场景计划。改变 mask 后不能沿用旧 completion/joint manifest。原模型被覆盖后，之前导出的指标也需要重新评估。

### 按场景记录的扩张参数运行

`MASK_DILATION_CONFIG` 可读取 `paintmesh-lama-dilation-policy` JSON，其中 `scenes.<场景名>.mask_dilation` 给出每场景固定半径，优先于全局 `MASK_DILATION`。记录必须包含全部八场景、分辨率 4、数据集 `Inpaint360GS/inpaint360` 和 `recursive_guide=false`；读取和检查发生在任何覆盖操作之前。

```bash
GPU=1 RECURSIVE_GUIDE=false OVERWRITE_INPAINT=true \
  MASK_DILATION_CONFIG="$PWD/scripts/paintmesh/configs/lama_dilation_by_scene.json" \
  bash scripts/paintmesh/run_inpaint_table1_batch.sh all
```

此参数是逐场景视觉选择，不是逐帧改变半径。记录同时保留选参理由、检查的虚拟视角帧号和对照图路径；只检查训练用虚拟视图的暗边、mask 外残留、纹理连续性及邻近结构，不以评估 GT 选参。可用 `compare_lama_dilation.py --scenes car cone_red ... --sample-count 13 --gpu 1` 生成覆盖不同方位的候选对照；`--dilations 5 10 15` 可以比较另一组三个递增半径。未指定采样数量时比较全部 90 视角。
