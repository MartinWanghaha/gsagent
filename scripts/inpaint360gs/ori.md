先给结论：Inpaint360GS 不是单一网络，而是一条由“COLMAP/3DGS 重建 → 2D 实例分割 → 跨视角对象关联 → 对象特征蒸馏 → 3D 对象删除 → 虚拟视角 2D 修补 → RGB-D 反投影 → 3DGS 再优化”组成的多阶段视觉流水线。

按你的要求，本次只使用了 `senior-computer-vision` skill，没有调用其他 skill、没有联网，也没有修改代码。下面是基于当前 checkout 的源码级静态审阅。

“每一个模块”这里按以下口径说明：

- 项目自有的入口、`seg/`、`scene/`、`tools/`、`utils/`、renderer 等，逐模块说明。
- 内嵌的 Detectron2、Entity、LaMa、Segment-and-Track-Anything、vanilla 3DGS 合计数千个第三方文件，按实际调用的功能边界分组说明，而不是逐个枚举 GLM、测试、训练器等上游文件。

## 一、整体数据流

```text
多视角照片
  │
  ├─ COLMAP
  │    └─ 相机内外参 + sparse points3D
  │
  ├─ Vanilla 3D Gaussian Splatting
  │    └─ 普通 RGB Gaussian 场景
  │
  └─ CropFormer Entity Segmentation
       └─ 每张图独立的实例 ID 图
              │
              └─ 将 Gaussian 中心投影到各视图
                    └─ 跨视角关联实例 ID
                           │
                           └─ 蒸馏 16-D Gaussian 对象特征
                                  └─ Object-aware 3DGS
                                         │
                                         ├─ 用户选择 target ID
                                         ├─ 删除对应 Gaussian
                                         └─ 生成 30 个虚拟视角
                                                │
                                                ├─ SAM 首帧交互选区
                                                └─ DeAOT 跨视角传播 mask
                                                       │
                                                       ├─ LaMa 补 RGB
                                                       ├─ LaMa 补 depth
                                                       └─ RGB-D 反投影成新点
                                                              │
                                                              └─ 初始化新 Gaussians
                                                                     └─ 5000 步 3DGS 微调
                                                                            └─ 最终补全场景
```

三个主入口分别是：

- 对象感知 3DGS：[run_seg.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/run_seg.sh:18)
- 删除与交互掩码：[run_remove.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/run_remove.sh:27)
- 2D/3D 补全：[run_inpaint.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/run_inpaint.sh:18)

## 二、核心数据契约

| 数据 | 形状/格式 | 含义 |
|---|---:|---|
| RGB 图像 | `[3,H,W] float32` | PyTorch 中通常为 `[0,1]`；OpenCV 读入时是 `[H,W,3] BGR uint8` |
| 实例标签 | `[H,W] uint8` | `0` 为背景，`1..K-1` 为实例 ID |
| 相机旋转/平移 | `R [3,3]`、`T [3]` | 最终组成 world-to-camera 和 projection 矩阵 |
| Gaussian 坐标 | `_xyz [N,3]` | 世界坐标下的 Gaussian 中心 |
| RGB SH | `_features_dc [N,1,3]`、`_features_rest [N,15,3]` | 默认三阶 SH，共 16 个系数 |
| Gaussian 尺度 | `_scaling [N,3]` | 存储 log-scale，使用时取 `exp` |
| Gaussian 旋转 | `_rotation [N,4]` | `wxyz` 四元数，使用时归一化 |
| Gaussian opacity | `_opacity [N,1]` | 存储 logit，使用时取 sigmoid |
| 对象特征 | `_objects_dc [N,1,16]` | 每个 Gaussian 的 16 维可学习 embedding，不是 one-hot 类别 |
| 对象分类器 | `Conv2d(16,K,1)` | 可把 16-D Gaussian/像素特征映射为 K 个场景级对象 |
| 渲染 RGB | `[3,H,W]` | alpha compositing 后的颜色 |
| 渲染对象特征 | `[16,H,W]` | 同一 rasterizer 对对象 embedding 做 alpha compositing |
| 渲染深度 | `[1,H,W]` | `Σ αᵢTᵢzᵢ`，未除以累计 alpha |

完整 Gaussian PLY 顶点字段为：

```text
x y z nx ny nz
f_dc_0..2
f_rest_0..44
opacity
scale_0..2
rot_0..3
obj_dc_0..15
```

定义和读写位于 [scene/gaussian_model.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/gaussian_model.py:45)。

一个容易误解的点是：16 维 embedding 不代表只能分 16 类。分类器是 `16 → num_classes`，真正限制来自标签 PNG 的 `uint8`，场景 ID 最多可靠表达到 255。

## 三、阶段 0：数据下载与准备

### `scripts/download_inpaint360gs_dataset.sh`

输入：

- 两个固定 Google Drive 文件 ID。
- 本机 `gdown`、`unzip`。

输出：

```text
data/inpaint360/
data/others/
```

它会无条件下载两个数据集并删除 ZIP；没有校验和或 `set -e`。见 [下载脚本](/home/martin/code/gsagent/submodules/Inpaint360GS/scripts/download_inpaint360gs_dataset.sh:1)。

### `scripts/download_inpaint360gs_result.sh`

输入固定 Google Drive 文件夹，输出 `inpaint360gs_result/`，解压其中所有 ZIP 后删除压缩包。见 [结果下载脚本](/home/martin/code/gsagent/submodules/Inpaint360GS/scripts/download_inpaint360gs_result.sh:4)。

### `scripts/run_data_prepare.sh`

当前硬编码：

```bash
scene="doppelherz"
data/inpaint360/${scene}/train_and_test/
```

依次调用 `convert.py` 和 `separate_train_test_ply.py`。见 [run_data_prepare.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/scripts/run_data_prepare.sh:3)。

### `convert.py`

输入目录：

```text
data/inpaint360/<scene>/train_and_test/input/*
```

处理：

1. COLMAP SIFT feature extraction。
2. exhaustive matching。
3. sparse mapper。
4. image undistortion。
5. 可选 2、4、8 倍降采样。

输出：

```text
train_and_test/
├── distorted/database.db
├── distorted/sparse/0/
├── images/
├── images_2/
├── images_4/
├── images_8/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

它强制 `ImageReader.single_camera=1`，即共享相机内参，但每张图仍有不同外参。`--magick_executable` 虽然被解析，实际 resize 固定调用 `mogrify`。见 [convert.py](/home/martin/code/gsagent/submodules/Inpaint360GS/convert.py:18)。

### `tools/separate_train_test_ply.py`

输入：

- `train_and_test/sparse/0/*.bin`
- `train_and_test/images*`
- 文件名不含 `test` 的图像被视为训练图。

处理：

- 只保留至少被一个训练相机观察到的 points3D。
- 裁剪每个点的 observation track。
- 将图像和 COLMAP 模型提升到正式场景根目录。

输出：

```text
data/inpaint360/<scene>/
├── images/
├── images_2/
├── images_4/
├── images_8/
└── sparse/0/
```

注意它先写过滤后的 `images.bin`，随后又复制完整 `images.bin` 覆盖回来。因此最终是“训练过滤 points3D + 完整相机列表”，方便项目读取 `test_*` 相机，但不属于严格自洽的 train-only COLMAP 模型。见 [separate_train_test_ply.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/separate_train_test_ply.py:27)。

## 四、阶段 1：对象感知 Gaussian 训练

### 1. Vanilla 3DGS

[run_seg.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/run_seg.sh:18) 首先调用：

```text
gaussian_splatting/train.py
```

输入：

- `images[_2|_4|_8]`
- `sparse/0/{cameras,images,points3D}.bin`
- 稀疏 COLMAP 点作为初始 Gaussian。

训练目标主要为：

```text
(1 - λ) · L1(render, GT)
+ λ · (1 - SSIM(render, GT))
```

训练期间进行 Gaussian clone、split、prune 和 opacity reset。

默认输出：

```text
output/<dataset>/<scene>/3dgs_output/
├── point_cloud/iteration_30000/point_cloud.ply
├── cfg_args
├── cameras.json
└── exposure.json
```

这个模型只有 RGB/几何属性，没有有效对象特征。

### 2. 每视角实例分割

入口是 [seg/raw_mask_sam.py](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/raw_mask_sam.py:68)。

虽然参数名和输出目录叫 `hqsam`，实际加载的是：

```text
HorNet-L
  → Mask2Former MS-Deformable-Attention pixel decoder
  → CropFormer crop-shared transformer decoder
```

配置证据见 [seg_config.json](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/seg_config.json:1)。

输入：

```text
data/<dataset>/<scene>/images_<resolution>/*
H×W×3 BGR uint8
```

CropFormer 会：

- 将完整图和 4 个局部 crop 一起推理。
- 使用 200 个 object queries。
- 输出 class-agnostic entity masks 和 scores。
- 保留 `score >= 0.5` 的实例。
- 低分 mask 先写，高分 mask 后写，因此重叠区域由高分实例覆盖。

输出：

```text
raw_hqsam/<stem>.png
raw_hqsam_color/<stem>.png
```

`raw_hqsam/*.png` 是 `[H,W] uint8`：

- 0：背景。
- 1..M：只在当前图内有效的局部实例 ID。
- ID 本质上来自置信度排序，不跨视角稳定，也不是语义类别。

标准 `--method sam` 分支当前不可用：

- 调用处把 SAM generator 的 `list[dict]` 当成字典使用。
- 本地 `automatic_mask_generator.py` 又注释掉了 RLE 生成，但后续仍访问 `rles`。
- BGR/RGB 约定也不一致。

见 [automatic_mask_generator.py](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/automatic_mask_generator.py:140)。

### 3. 跨视角 3D 关联

入口：[seg/mask_associate.py](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/mask_associate.py:34)。

输入：

- Vanilla 3DGS 的 `_xyz [N,3]`。
- COLMAP 相机。
- `raw_hqsam/*.png`。
- 默认 `16×16` patch 网格。

每一视图的处理：

1. 把所有 Gaussian 中心投影到像素平面。
2. 对每个 2D 实例和活跃 patch，收集中心投影落入该 mask 的 Gaussian。
3. 按相机深度做 `KMeans(k=2)`。
4. 选择平均深度更近的簇。
5. 再只保留该簇最近的约 30%，作为当前 2D 实例的“可见 Gaussian 集合”。
6. 与已有场景对象集合比较并映射为全局 ID。

关联分数实际为：

```text
intersection / (current_size + intersection + ε)
```

它不是标准 IoU；完全相同的集合分数也接近 0.5。阈值固定为 0.1。

输出：

```text
associated_hqsam/
├── <stem>.png
└── scene.json

associated_hqsam_color/
```

`scene.json` 保存：

- `num_classes`，包括背景。
- raw/associated 路径。
- patch 数量。

这一步只投影 Gaussian 中心，没有使用 Gaussian footprint、opacity、alpha 或真正的 z-buffer，因此遮挡判断只是“投影 + 深度聚类”近似。

### 4. 标签编号预览

[tools/add_label_num_hqsam.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/add_label_num_hqsam.py:13)：

输入：

- `images_<resolution>/`
- `associated_hqsam/*.png`

输出：

```text
images_<resolution>_num/*
```

它在每个实例轮廓的矩中心绘制红色 ID，供用户在第二阶段选择 `target_id`。它不修改训练标签。

### 5. 对象特征蒸馏

入口：[seg/distillation.py](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/distillation.py:33)。

输入：

- Vanilla 3DGS PLY。
- `associated_hqsam/*.png`。
- `scene.json` 中的类别数。
- 默认 2000 次迭代配置。

训练边界：

```text
Gaussian object embedding: [N,1,16]
        ↓ 自定义 differentiable rasterizer
render_object: [16,H,W]
        ↓ Conv2d(16,K,1)
logits: [K,H,W]
        ↓ CrossEntropy
GT instance IDs: [H,W]
```

只优化：

- `_objects_dc`
- 1×1 classifier

冻结：

- xyz
- SH/RGB
- scale
- rotation
- opacity

每 50 步还计算一次 3D 邻域正则。函数同时返回 KL 和 cosine loss，但调用处丢弃了 KL，实际加入训练的只有很小权重的对象 embedding 余弦一致性。

输出：

```text
output/<dataset>/<scene>/
├── point_cloud/iteration_2000/
│   ├── point_cloud.ply
│   └── classifier.pth
├── chkpnt2000.pth
└── cfg_args
```

### 6. 渲染检查

[render.py](/home/martin/code/gsagent/submodules/Inpaint360GS/render.py:61) 输入对象感知 PLY、classifier 和相机，输出：

```text
train/ours_2000/
test/ours_2000/
├── renders/
├── gt/
├── gt_objects_color/
├── objects_pred/
├── objects_pred_color/
├── depth/
└── fused_full_col_dep_ply/

video/ours_2000/final_video.mp4
```

每视图还会用 GT RGB 和渲染深度反投影一份完整 RGB-D 点云 PLY。

## 五、阶段 2：对象删除与虚拟相机

### `tools/init_configs.py`

输入：

- `target_id`：永久删除对象。
- `target_surronding_id`：临时移除的遮挡对象。
- dataset、scene。

输出两个场景配置：

```text
config/object_removal/<dataset>/<scene>.json
config/object_inpaint/<dataset>/<scene>.json
```

并写入：

```json
{
  "target_id": [...],
  "surrounding_ids": [...],
  "select_obj_id": ["target 和 surrounding 的组合"]
}
```

参数名源码中确实拼作 `surronding`。见 [init_configs.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/init_configs.py:70)。

### `edit_object_removal.py`

入口：[edit_object_removal.py](/home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_removal.py:154)。

输入：

- 对象感知 Gaussian PLY。
- `classifier.pth`。
- `select_obj_id`。
- 默认概率阈值 `0.7`。

处理：

1. 对每个 Gaussian 的 16-D embedding 分类。
2. 取目标类别概率大于阈值的种子 Gaussian。
3. 对种子点逐轴做 IQR 离群过滤。
4. 用 SciPy Delaunay 构造 3D 凸包。
5. 将凸包内的其他 Gaussian 也并入删除区域。
6. 把每个对象拆成独立 `GaussianModel`，主模型保留补集。

输出：

```text
point_cloud_object_removal/iteration_2000/
├── point_cloud.ply
├── point_cloud_<target-id>.ply
└── point_cloud_<surrounding-id>.ply

point_cloud_vis/
train/ours_object_removal/iteration_2000/
test/ours_object_removal/iteration_2000/
```

同时把目标的物理半径写回 removal/inpaint JSON，供虚拟相机轨迹估算。

### `tools/virtual_pose.py`

入口：[virtual_pose.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/virtual_pose.py:102)。

输入：

- 训练相机轨迹。
- 目标物理半径。
- 完整模型和删除模型。
- classifier。

处理：

- 对相机做 PCA/focus 分析。
- 结合目标半径和 FoV 估算圆周半径。
- 固定生成 30 个虚拟相机。
- 用第一个训练 Camera 作为模板替换外参。

输出：

```text
virtual/ours_2000/                         # 完整场景
virtual/ours_object_removal/iteration_2000/ # 删除场景
├── renders/
├── depth/
├── objects_pred/
└── fused_vanilla_col_dep_ply/
```

随后把删除场景 RGB 打包为：

```text
Segment-and-Track-Anything/assets/images.zip
```

### Segment-and-Track-Anything

这是独立的交互阶段，不是 `run_seg.sh` 的分割器。

真实链路：

```text
用户在首帧点击/框选/文本选取
  ↓
SAM 生成首帧 mask
  ↓
DeAOT 在 30 个虚拟视角间传播
  ↓
逐帧实例标签 PNG
```

关键模块：

| 模块 | 输入 | 输出 |
|---|---|---|
| [app.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/app.py:104) | 视频或 ZIP、点击/框/文本 | Gradio UI、跟踪任务、回滚与细化 |
| [tool/segmentor.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/tool/segmentor.py:21) | RGB、point/box/mask prompt | `[H,W]` SAM 二值 mask |
| [tool/detector.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/tool/detector.py:57) | RGB、文本 | GroundingDINO boxes |
| [SegTracker.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/SegTracker.py:37) | RGB、SAM mask、跟踪状态 | 实例 ID 图及新对象发现 |
| [aot_tracker.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/aot_tracker.py:52) | RGB、reference ID mask | DeAOT logits/ID mask |
| [seg_track_anything.py](/home/martin/code/gsagent/submodules/Inpaint360GS/Segment-and-Track-Anything/seg_track_anything.py:263) | 图像序列、首帧 mask | palette PNG、预览帧、MP4、GIF、ZIP |
| `model_args.py` | checkpoint/阈值配置 | SAM ViT-B、R50-DeAOTL、sam-gap=10 等设置 |

默认输出：

```text
Segment-and-Track-Anything/tracking_results/images/
├── images_masks/*.png
├── images_masked_frames/*.png
├── images_seg.mp4
├── images_seg.gif
└── images_pred_mask.zip
```

SAM 和 DeAOT 的职责不同：

- SAM：产生或修正某一帧的目标区域。
- DeAOT：在虚拟相机序列中传播对象 identity。
- 每 10 帧可再次运行 SAM everything，尝试发现新区域。

源码没有把 CropFormer/associated mask 自动传入 SAM，也没有从 UI 自动复制结果到后续主工程；这里必须人工操作。

## 六、阶段 3：2D 与 3D 补全

### `tools/prepare_lama_data.py`

入口：[prepare_lama_data.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/prepare_lama_data.py:13)。

`--inpaint2lama` 模式输入：

```text
tracking_results/images/images_masks/*.png
virtual/ours_2000/depth/*
virtual/ours_object_removal/iteration_2000/renders/*
virtual/ours_object_removal/iteration_2000/depth/*
```

掩码处理：

- 灰度阈值 128。
- 移除面积小于 50 的连通域；若全部太小则保留最大域。
- 使用 21×21 MaxFilter，默认向外膨胀约 10 像素。

输出：

```text
data/<scene>/inpaint_2d_unseen_mask_virtual/

LaMa/data/color/360_<scene>_virtual/
├── 00000.png
└── 00000_mask.png

LaMa/data/depth/360_<scene>_virtual/
├── 00000.npy
├── 00000_mask.png
└── depth_original/00000.npy
```

反向模式把 LaMa 输出复制回：

```text
data/<scene>/images_inpaint_unseen_virtual/
virtual/ours_object_removal/iteration_2000/depth_completed/
```

它只是把 `.png` 重命名为 `.JPG`，没有重新 JPEG 编码。

### `LaMa/bin/predict_color.py`

输入：

- RGB PNG `[H,W,3]`
- 对应 `_mask.png`
- `LaMa/big-lama/config.yaml`
- checkpoint。

输出：

```text
LaMa/output/color/360_<scene>_virtual/*.png
```

默认启用 refinement。可选 `--recursive_guide` 会把前一帧结果和当前帧水平拼接，作为时序上下文。见 [predict_color.py](/home/martin/code/gsagent/submodules/Inpaint360GS/LaMa/bin/predict_color.py:50)。

### `LaMa/bin/predict_depth.py`

输入：

- 删除场景深度 `.npy`
- 二值 mask
- 完整场景对应视图的原始深度。

处理：

- 把深度作为单通道图送入 LaMa。
- 网络输出归一化深度。
- 用完整场景深度的 `min/max` 反归一化。

输出：

```text
LaMa/output/depth/360_<scene>_virtual/
├── 00000.npy
└── vis/00000*.png
```

见 [predict_depth.py](/home/martin/code/gsagent/submodules/Inpaint360GS/LaMa/bin/predict_depth.py:102)。

### `edit_object_removal_plyfusion.py`

入口：[edit_object_removal_plyfusion.py](/home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_removal_plyfusion.py:32)。

每个虚拟视图输入：

- 补全 RGB。
- 补全 depth `[H,W]`。
- 二值 inpaint mask。
- 相机内参 `K [3,3]`。
- camera-to-world `[4,4]`。

反投影：

```text
pixel [u,v,1]
  → K⁻¹
  → depth · ray
  → c2w
  → world point [x,y,z]
```

输出：

```text
virtual/ours_object_removal/iteration_2000/
├── fused_mask_col_dep_ply/00000.ply
└── fused_hole_col_dep_ply/00000.ply
```

这里的 “fusion” 实际是逐视角分别保存 PLY，没有把 30 个视图融合成一个统一点云。

### `edit_object_inpaint.py`

入口：[edit_object_inpaint.py](/home/martin/code/gsagent/submodules/Inpaint360GS/edit_object_inpaint.py:202)。

输入：

- 原对象感知 3DGS。
- classifier。
- 目标及 surrounding IDs。
- 30 个虚拟视图的补全 RGB/mask。
- 一份补充彩色点云。

补充点云路径被硬编码为：

```text
virtual/ours_object_removal/iteration_2000/
fused_mask_col_dep_ply/00004.ply
```

初始化过程：

1. 删除目标和 surrounding Gaussian。
2. 读取 `00004.ply`。
3. Open3D statistical outlier removal。
4. 新点 RGB 转为 SH DC。
5. `simple_knn` 三近邻距离初始化 Gaussian scale。
6. rotation 初始化为单位四元数。
7. opacity 初始化为 0.1 的 logit。
8. 根据旧背景的 5-NN 初始化新点的对象 embedding。
9. 将 `[保留背景, 新 Gaussian]` 拼接。

微调损失为：

```text
L =
(1 - λdssim) · L1(mask 外)
+ λdssim · (1 - SSIM(全图))
+ λlpips · LPIPS(mask bbox 的 2×2 patches)
```

默认：

```text
λdssim = 0.8
λlpips = 0.0005
iterations = 5000
```

训练期间只允许新增区域参与 clone/split/prune；训练后恢复原始背景，再把目标投影区域中的优化结果覆盖回来。

输出：

```text
point_cloud_object_inpaint_virtual/
└── iteration_5000/point_cloud.ply

train/ours_object_inpaint_virtual/iteration_5000/
test/ours_object_inpaint_virtual/iteration_5000/
inpaint/ours_object_inpaint_virtual/iteration_5000/
video/
```

完成深度只用于生成初始 3D 点，5000 步优化中没有显式 depth loss。

## 七、3DGS 核心模块

### `arguments/__init__.py`

输入 CLI 和 `<model_path>/cfg_args`，输出分组后的：

- `ModelParams`
- `PipelineParams`
- `OptimizationParams`
- `AssociateParams`

主要字段包括 source/model/images/object path、resolution、SH degree、训练步数和 densification 参数。见 [arguments](/home/martin/code/gsagent/submodules/Inpaint360GS/arguments/__init__.py:44)。

`get_combined_args()` 用 `eval()` 解析 `cfg_args`，因此模型目录不能来自不可信来源。

### `scene/*.py`

| 模块 | 输入 | 输出 |
|---|---|---|
| [scene/__init__.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/__init__.py:25) | ModelParams、PLY、相机数据 | `Scene`、train/test/inpaint Camera、checkpoint 读写 |
| [cameras.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/cameras.py:18) | RGB、R/T、FoV、mask | GPU Camera、view/projection matrix、camera center |
| [colmap_loader.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/colmap_loader.py:88) | COLMAP text/bin | Camera/Image/points3D 字典和 numpy 数组 |
| [dataset_readers.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/dataset_readers.py:160) | 图像、COLMAP、mask | `CameraInfo`、`SceneInfo`、点云、数据划分 |
| [gaussian_model.py](/home/martin/code/gsagent/submodules/Inpaint360GS/scene/gaussian_model.py:141) | BasicPointCloud/PLY | Gaussian 参数、optimizer、删除/补全/densification、PLY |

对于 Inpaint360 数据：

- 名称含 `test_` 的图像进入 `inpaint_cameras`。
- 其余图像每 8 张取一张普通 test，其余为 train。
- `test_*` 是实物移除后重新拍摄的评估图，不参与初始训练。

### `gaussian_renderer/*.py`

[gaussian_renderer/__init__.py](/home/martin/code/gsagent/submodules/Inpaint360GS/gaussian_renderer/__init__.py:18) 输入：

```text
Camera
GaussianModel
background [3]
pipeline flags
```

送入 CUDA：

```text
means3D     [N,3]
means2D     [N,3] dummy gradient carrier
SH          [N,16,3]
objects     [N,1,16]
opacity     [N,1]
scale       [N,3]
rotation    [N,4]
```

输出：

```python
{
  "render": [3,H,W],
  "render_object": [16,H,W],
  "depth_3dgs": [1,H,W],
  "viewspace_points": [N,3],
  "visibility_filter": [N],
  "radii": [N]
}
```

`network_gui.py` 是与 SIBR viewer 通信的 TCP 层：接收 JSON 相机，输出渲染字节。

### 自定义 CUDA rasterizer

Python 包为 `diff_gaussian_rasterization_inpaint360gs`。

相比 vanilla rasterizer，它额外 alpha-composite：

- 16 通道对象特征。
- depth。
- alpha。

核心前向公式：

```text
αᵢ = min(0.99, opacityᵢ · exp(-0.5 dᵀΣ⁻¹d))
wᵢ = αᵢTᵢ

RGB    = Σ wᵢRGBᵢ + Tfinal·background
object = Σ wᵢobjectᵢ
depth  = Σ wᵢzᵢ
alpha  = Σ wᵢ
```

RGB 固定 3 通道、对象特征固定 16 通道、CUDA tile 固定 16×16。见 [config.h](/home/martin/code/gsagent/submodules/Inpaint360GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h:13) 和 [forward.cu](/home/martin/code/gsagent/submodules/Inpaint360GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:263)。

Python wrapper 丢弃了底层返回的 alpha；depth 也没有除以 alpha，因此透明边缘会偏向零深度。

### `simple-knn`

输入 CUDA float `[N,3]`，输出 `[N]`：每点到三个最近邻的平方距离均值。

它用 Morton code、空间排序和分块 AABB 加速，主要用于：

```text
log(sqrt(distCUDA2(points)))
```

初始化 Gaussian 三轴 scale。

值得注意：标准 [install.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/install.sh:6) 安装的是 `gaussian_splatting/submodules/simple-knn`，不是顶层自定义 `submodules/simple-knn`。

## 八、工具和公共模块 I/O 索引

### `tools/`

| 模块 | 输入 → 输出 |
|---|---|
| `add_label_num_hqsam.py` | associated ID 图 + RGB → 带编号 RGB |
| `init_configs.py` | target/surrounding IDs → removal/inpaint JSON |
| `prepare_lama_data.py` | 跟踪 mask + 虚拟 RGB-D ↔ LaMa 目录 |
| `virtual_pose.py` | 相机、目标半径、完整/删除 PLY → 30 帧虚拟 RGB-D/PLY/ZIP |
| `combine_gaussian_scene.py` | inpaint PLY + surrounding PLY → 拼接后的最终 PLY |
| `metrics_fid_masked.py` | renders、GT、unseen masks → SSIM/PSNR/LPIPS/FID JSON |
| `separate_train_test_ply.py` | train_and_test COLMAP → 正式场景目录 |
| `read_write_model.py` | COLMAP text/bin ↔ Camera/Image/Point3D 字典 |
| `vis_obj_color.py` | ID PNG → 伪彩色 PNG |
| `tools/__init__.py` | 空包标记，无运行 I/O |

### `utils/`

| 模块 | 主要输入 → 输出 |
|---|---|
| [camera_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/camera_utils.py:21) | PIL/CameraInfo/mask → GPU Camera；Camera → JSON |
| [compose_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/compose_utils.py:49) | RGB 点云 → Gaussian PLY；旧 Gaussian 5-NN → 新点对象特征 |
| [general_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/general_utils.py:29) | PIL→tensor、学习率、四元数/协方差、随机状态 |
| [graphics_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/graphics_utils.py:23) | 3D 点/姿态/FoV → view、projection、投影点 |
| [image_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/image_utils.py:15) | 图像 → MSE/PSNR；实例图 → binary mask stack |
| [loss_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/loss_utils.py:15) | render/GT/mask/features → L1、SSIM、KL、cosine 正则 |
| [point_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/point_utils.py:4) | depth+K+c2w → 世界点；点+颜色 → PLY |
| [pose_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/pose_utils.py:44) | 相机轨迹 → PCA 对齐、focus、圆/椭圆/球面轨迹 |
| [sh_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/sh_utils.py:57) | SH+观察方向 → RGB；RGB ↔ SH DC |
| [stepfun.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/stepfun.py:5) | 路径采样权重 → CDF/逆 CDF → 恒速轨迹样本 |
| [system_utils.py](/home/martin/code/gsagent/submodules/Inpaint360GS/utils/system_utils.py:16) | 路径 → mkdir；checkpoint 目录 → 最大 iteration |

### `lpipsPyTorch/`

| 模块 | 输入 → 输出 |
|---|---|
| `__init__.py` | 两个 `B×3×H×W` 图像 → LPIPS 距离 |
| `modules/networks.py` | 图像 → Alex/Squeeze/VGG 多层特征 |
| `modules/lpips.py` | 多层归一化特征差 → 感知距离 |
| `modules/utils.py` | 网络类型 → 下载/读取 LPIPS 线性权重 |

当前封装每调用一次 `lpips()` 都会重新构建网络；评估大量图片时非常慢。

## 九、内嵌第三方子系统

### `gaussian_splatting/`

这是独立的 vanilla 3DGS 工程，当前主流水线只直接调用 `train.py`。其功能边界是：

- `train.py`：COLMAP/RGB → vanilla Gaussian PLY。
- `scene/**`：相机、点云、Gaussian 参数与 checkpoint。
- `gaussian_renderer/**`：RGB differentiable rasterization。
- `render.py`、`render_video.py`：离线渲染和轨迹视频。
- `metrics.py`、`full_eval.py`：vanilla 评估。
- `convert.py`、`sparse2dense.py`、`read_write_model.py`：COLMAP 工具。
- `submodules/diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`：CUDA 算子。
- `SIBR_viewers`：可视化 GUI，不参与批处理主流程。

### `seg/detectron2/`

实际运行的是：

```text
seg/detectron2/detectron2/projects/CropFormer/**
```

关键边界：

- Predictor：`BGR H×W×3` → full image + 4 crops。
- HorNet backbone：输出 stride 4/8/16/32 多尺度特征。
- MSDeformAttn pixel decoder：输出 stride-4 mask feature 和三尺度 256-D features。
- Crop-shared transformer decoder：200 queries → entity/no-object logits 和 masks。
- 最终输出 Detectron2 `Instances(pred_masks, scores, pred_classes)`。

Detectron2 其他 ROI head、trainer、dataset、测试和项目代码主要是上游框架，不进入当前推理链。

### `seg/Entity/`

这是 Entity/EntityV2 的源码快照：

- `EntitySeg`：旧版 class-agnostic FCOS/CondInst 路线。
- `Entityv2/CropFormer`：与 Detectron2 project 中运行代码的镜像。
- `High-Quality-Segmention`：粗 mask 精细化模块。
- `Open-Metrics`：Open AP/PQ/mIoU。
- COCO/entity API：RLE 和评估支撑。

主脚本没有导入该目录；真正运行的是 Detectron2 内的 CropFormer 副本。因此名称 `hqsam` 也不能解释为这里的 high-quality refinement。

### `LaMa/`

实际主链使用：

- `bin/predict_color.py`
- `bin/predict_depth.py`
- `saicinpainting.training.trainers.load_checkpoint`
- `evaluation.refinement`
- FFC generator/loss/visualizer 等模型支撑。

其他 `bin/`、mask generator、训练器、评估脚本属于上游 LaMa 工程，没有被 `run_inpaint.sh` 调用。

### `Segment-and-Track-Anything/aot/`

默认模型是 `R50-DeAOTL`：

- `configs/**`：模型、checkpoint、memory 配置。
- `dataloaders/video_transforms.py`：RGB/ID mask → ImageNet-normalized tensor。
- `networks/encoders`：ResNet-50 多尺度特征。
- `networks/models/deaot.py`：视觉和 identity 双分支传播。
- `networks/layers/transformer.py`：长时全局与短时局部 memory attention。
- `networks/decoders/fpn.py`：恢复对象 logits。
- `networks/engines/**`：reference memory、逐帧传播、多对象分组。
- correlation extension：局部注意力 CUDA 加速，失败时可回退。

## 十、源码中已经确认的重要断链与风险

1. **`hqsam` 名称与实现不符。** 主链实际使用 CropFormer EntityV2，不是 HQ-SAM，也不是标准 SAM。

2. **标准 SAM 自动分割分支是坏的。** 返回类型、RLE 和颜色空间均有接口问题；默认 `hqsam` 路径才是实际可运行路径。

3. **交互跟踪不是自动步骤。** `run_remove.sh` 会阻塞在 Gradio；需要手动加载 `images.zip`、选择目标并运行跟踪。跟踪结果和主工程后续输入之间没有通用自动桥接。

4. **当前 checkout 的 LaMa 不完整。** 两个定制预测脚本都导入 `saicinpainting.training.data.datasets`，但当前目录树中缺少 `LaMa/saicinpainting/training/data/`。仅依靠当前仓库会在 import 阶段失败，除非从外部安装或恢复该目录。

5. **跟踪 mask 二值化有数据风险。** 跟踪器输出带随机 palette 的 P 模式 PNG；`prepare_lama_data.py` 却通过 OpenCV 灰度值 `<128/≥128` 二值化，而不是按“实例 ID 是否非零”。有效目标可能因调色板亮度被误删。

6. **删除后 surrounding 对象恢复存在错误。**

   - `edit_object_removal.py` 合回对象时传入了 `select_obj_id` 而非永久 `target_id`，可能导致 surrounding 仍全部缺失。
   - `edit_object_inpaint.py` 只有 `len(surrounding_ids)>1` 才恢复；恰好一个 surrounding 不会恢复。

7. **大量路径和迭代数硬编码。**

   - 蒸馏/删除默认为 2000。
   - inpaint 为 5000。
   - 新点只读取 `iteration_2000/.../00004.ply`。
   - `prepare_lama_data.py` 从 distill JSON 读取 2000，而 virtual pose 默认加载最新 checkpoint。
   - 改一个迭代数会让后续目录断链。

8. **文档与脚本参数不一致。**

   - README 写 `inpaint360gs`，脚本使用 `inpaint360`。
   - README 称变量为参数，但 `run_*.sh` 不解析 `$@`，必须编辑脚本。
   - `resolution=1` 时 `run_seg.sh` 查找 `images_1`，而数据准备把原分辨率目录命名为 `images`。
   - `run_remove.sh` 中的 `resolution` 完全未使用。

9. **跨视角关联只是近似。** 它没有 rasterize Gaussian，也没有显式处理 alpha/opacity；非方形图像的 `[W,H]` 转换和 patch reshape 还可能造成坐标混乱。

10. **标签上限和空输入缺少保护。**

    - uint8 ID 最多 255。
    - 空实例会触发 `torch.cat([])`。
    - Delaunay 对点过少、共面或空点没有异常保护。
    - 空 mask 的 bbox 计算会失败。

11. **深度不是标准期望深度。** Renderer 输出的是未按 alpha 归一化的 `ΣαTz`，透明边缘深度偏小；Python 顶层又不暴露 alpha。

12. **评估的 masked 指标并非真正 ROI 指标。** [metrics_fid_masked.py](/home/martin/code/gsagent/submodules/Inpaint360GS/tools/metrics_fid_masked.py:137) 把 ROI 外置零后对整图算 SSIM/PSNR/LPIPS，大面积相同黑背景会抬高指标；FID 也不应用 `--crop/--resize`。

13. **GPU 和架构耦合很强。** 多处直接 `.cuda()`，`safe_state()` 固定 CUDA 0，自定义 extension 的 pip 构建主要硬编码 `sm_86`。

14. **安装脚本缺少失败保护。** `install.sh` 没有 `set -e`，会直接覆盖当前环境的 `fvcore/common/registry.py`；中间编译失败后仍可能打印安装完成。

## 十一、结合你当前打开的三个文件

- [run_data_prepare.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/scripts/run_data_prepare.sh:3) 是自有数据入口，但场景固定为 `doppelherz`。它先对 `train_and_test/input` 做 COLMAP，再构造正式场景目录。

- [download_inpaint360gs_dataset.sh](/home/martin/code/gsagent/submodules/Inpaint360GS/scripts/download_inpaint360gs_dataset.sh:1) 只是固定链接下载器，不负责 COLMAP、mask 或模型准备。

- [raw_mask_sam.py](/home/martin/code/gsagent/submodules/Inpaint360GS/seg/raw_mask_sam.py:93) 是最容易被名称误导的文件：默认 `hqsam` 分支实际执行 CropFormer Entity Segmentation；真正的 SAM 位于后面的 Segment-and-Track-Anything 交互流程中。