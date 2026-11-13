# 原始物体高斯复用与生成 mesh 的可移动组合

入口为 `run_recompose.py`，配置为 `configs/recompose.json`。当前唯一的组合方式：

- **3DGS：已补全的 inpaint 背景高斯 + 真正删除的原始物体高斯。全部直接复用，不重新训练。**
- **mesh：inpaint 背景 mesh + MV-SAM3D 生成、经过多视角位姿配准及接触/碰撞检查的物体 mesh。**

不再提供从生成 mesh 采样高斯或物体外观优化的旧路径。已有 seg/remove/inpaint 的训练代码不改动。

## 场景选择

上级目录为 `output/paintmesh/Inpaint360GS`。

| 场景 | inpaint 背景来源 |
|---|---|
| truck、doppelherz、bag、fruits | `fitlama/scenes/<scene>/inpaint` |
| car | `best_results/car/run` |
| cone_red、cone_yellow、cube、garden_toys、redbull、toys | `fit/scenes/<scene>/inpaint` |

沿已完成的清单读取真实模型、去除记录及目标 ID，不猜测训练迭代数。11 个场景包含 13 个目标，toys 包含 22/23/24 三个独立物体。

## 原始高斯的精确提取

从 inpaint 清单追溯实际 removal 清单及 workspace，读取**真正的去除前高斯模型**和**去除后、inpaint 前的保留高斯模型**。对位置、SH、透明度、尺度、旋转等共同渲染属性进行精确多重集合差，保留重复记录的数量，验证去除后模型是原始属性不变的子集。没有空间距离容差、近似匹配或替代重建。

将集合差中的所有点分配给相应目标。单目标场景直接取全部真正删除的点；多目标先按语义 ID 分组，未匹配到目标 ID 的删除点分配给最近的已标注删除点，记录来源和分配方式。所有删除点恰好分配一次。每个目标保存原始行号和完整原始 PLY 属性。

doppelherz 的同一语义 ID 曾覆盖盒子和旁边白色物体。实际删除集合只包含真正去除的部分，所以不会重新加入实际保留的同 ID 物体。生成 mesh 的输入 mask 也用原始/保留中心集合差进一步隔离，结果只写到本组合输出目录。

## mesh 的位姿估计

使用去除前的真实训练 RGB、实例 mask、标定相机及 mesh；排除 test 开头的图像。选择 8 个生成/配准视角、3 个留出检查视角。射线深度经过 mask、覆盖率和多视角一致性筛选，形成鲁棒锚点；这些深度来自 PGSR 重建，不是传感器真值。

调用固定版本官方 MV-SAM3D，多视角 mask 和已有标定 pointmaps 参与生成。严格处理 GLB / PyTorch3D / CV / COLMAP 坐标变换，非均匀形状缩放仅烘焙一次。官方初值与多方向假设经过鲁棒锚点、遮挡、轮廓及深度配准，再对实际背景三角面进行受限接触修正。

生成输入固定使用**原始分辨率 RGB → 官方 1.2 倍物体裁剪 → 518**，取消整图先缩到 768 的路径。`evidence/images` 和 `evidence/object` 的 RGB 像素与原始照片解码结果逐像素一致；mask 使用同一实例标签，按原图网格最近邻映射；`pointmaps.npz` 使用同一 PGSR proxy 在原生相机网格上分块射线求交，不把低分辨率 XYZ 放大当作原生几何。

`observation_max_size=768` 仅用于既有选图、锚点、支撑和位姿验证，保留其坐标、像素、采样与阈值，**不限制生成图像尺寸**。`observations.json/views/*/generation` 单独记录生成尺寸和内参。生成前检查 RGB 像素一致性、图像/mask/pointmap 尺寸、视角顺序以及 pointmap 重投影误差；没有低分辨率回退。

本轮保持 8 个生成视角、3 个验证视角、seed=42、50/25 推理步数、两阶段加权和 mixed 权重不变；生成器源码及权重不修改。批处理通过 `native_input_isolation.json` 确认选图、位姿观测、锚点与上轮完全相同。

不拟合平面替代背景 mesh。实际观测的支撑权重大于补全支撑；非封闭背景不用全局有符号距离。检查局部接触、穿透、物体间三角面碰撞及封闭体包含。默认留出轮廓 median IoU ≥ 0.75，距离阈值按物体范围对角线归一化，世界坐标没有假定的米单位。

如果 mesh 的接触或轮廓不通过，仍然输出精确恢复的 3DGS，mesh 标记 `needs_review`，不声称原位几何已准确。两种表示的完成状态在 `scene.json/components` 中分别记录。

## 两种表示如何同步移动

生成 mesh 的规范坐标到初始世界坐标的矩阵为 `T0=original_pose`，当前矩阵为 `T=object_to_world`。原始物体高斯本身已在正确的世界坐标中，因此：

```text
mesh 当前坐标 = T × mesh 规范坐标
高斯当前坐标 = Δ × 原始世界坐标，Δ = T × inverse(T0)
```

初始 `Δ=I`，直接复制原始高斯记录，避免来回旋转导致数值变化。移动时对高斯中心、协方差及 SH 应用同一个 Δ，mesh 应用对应 T。初始估计中的 mesh 接触微调不会移走原始物体高斯。两种表示保留相对关系，不强制生成几何与原始高斯逐顶点一致。

所有背景和物体高斯在一个渲染过程中共同深度排序。移动目的地重新检查支撑及碰撞；移动保持已标定的统一尺度。当前不做动态重打光、阴影或物理稳定性模拟。

## 环境与运行

几何/渲染使用 `paintmesh`；MV-SAM3D 使用独立 `mvsam3d` 环境。生成器仓库 `/home/member/wyc/MV-SAM3D` 固定在 `abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd`，权重配置为 `checkpoints/hf/pipeline.yaml`。真实多视角推理已验证。

```bash
cd /home/member/wyc/gsagent
# 先输出所有场景的精确恢复 3DGS，无须生成 mesh 或训练
conda run --no-capture-output -n paintmesh python scripts/paintmesh/run_recompose.py \
  run --scenes all --gpus 0,1 --until gaussians

# 完成 mesh 配准、两种表示的组合导出；复用未改变的生成/位姿缓存
conda run --no-capture-output -n paintmesh python scripts/paintmesh/run_recompose.py \
  run --scenes all --gpus 0,1

# 全场景真实渲染对照与目录报告
conda run --no-capture-output -n paintmesh python scripts/paintmesh/review_recomposition.py --gpu 0
```

命令覆盖同一输出目录。没有旧高斯训练策略的回退入口。按输入、实现与输出 hash 复用未改变的生成资产，场景锁阻止并发覆盖；每张卡独占生成/渲染阶段并检查显存余量。

原生分辨率全场景重生成入口：`regenerate_native_recomposition.py --gpus 0,1 --log-dir <日志目录>`。两条 CPU 队列准备输入，两张 GPU 分别运行一个生成任务；按 `execution.gpu_max_used_mib=8192`，单卡已用显存小于 8 GiB 即启动，否则按 `execution.gpu_poll_interval_seconds=60` 每分钟检查一次，不终止其他任务。`--resume` 复用已经完成的本轮准备与场景。完成后在原目录覆盖 mesh、组合场景和渲染对照，重新逐属性核对原始高斯，并写 `native_input_comparison.md`。

依赖导入及输入核验完成、实际加载 GPU 模型之前，再检查一次相同显存门槛，记录到 `generated/runtime.json`。其他进程在任务运行中增长显存仍可能触发 OOM；显式 CUDA OOM 自动释放子进程并等待重试，保持输入、权重、随机种子不变。非 OOM 错误直接报告。`--resume --wait-for-queue` 可在当前批次结束后仅续跑未完成场景。

两张卡共享待运行队列：显存超过门槛的卡不占住某个场景，空闲卡可接手该场景；准备完成且队列已空时，忙卡的等待线程退出，避免空队列阻塞整批结束。

## 输出文件

```text
output/paintmesh/Inpaint360GS/recomposition/<scene>/
  original_gaussians/
    manifest.json                 源模型、精确集合差及点数，training=false
    <id>/gaussians_world.ply       实际删除的原始物体高斯
    <id>/original_indices.npy      在去除前模型中的行号
  objects/<id>/
    evidence/                     原始训练 RGB、mask、相机、射线深度及检查图
    generated/                    MV-SAM3D GLB / 规范 mesh / 初始位姿
    placement/                    位姿、轮廓/支撑/碰撞报告
  composition/
    scene.json                    可移动主场景，schema_version=2
    placement_validation.json
    world/
      composed_gaussians.ply       inpaint 背景 + 原始删除高斯
      gaussians_manifest.json      3DGS 可先于 mesh 独立完成
      gaussian_ranges.json        合并 PLY 内的实例行范围
      <id>_gaussians.ply           原位或移动后的原始物体高斯
      mesh_scene.gltf              完整背景 mesh + 独立生成物体节点
      mesh_buffers/               glTF 外部缓冲，必须一起保留
      <id>_mesh.ply                生成物体的世界坐标 mesh
      <id>_textured.glb            保留原生颜色/纹理的生成物体
      layers.json
```

主场景仍引用原有背景及规范资产；跨机器移动要带齐引用文件并更新路径。完整 glTF 使用烘焙顶点颜色，原生物体 GLB 另外保留纹理。背景连通分量过滤导致整个局部 ROI 丢失时，只从同次、同模型的未过滤 TSDF 恢复局部静态 patch，并记录来源。

## 渲染与移动命令

```bash
conda run --no-capture-output -n paintmesh python scripts/paintmesh/run_recompose.py render \
  --scene output/paintmesh/Inpaint360GS/recomposition/car/composition/scene.json \
  --camera output/paintmesh/Inpaint360GS/recomposition/car/objects/7/evidence/observations/00000.npz \
  --output output/paintmesh/Inpaint360GS/recomposition/car/preview.png --gpu 0

conda run --no-capture-output -n paintmesh python scripts/paintmesh/run_recompose.py move \
  --scene output/paintmesh/Inpaint360GS/recomposition/car/composition/scene.json \
  --object 7 --pose /absolute/new_pose.json \
  --output output/paintmesh/Inpaint360GS/recomposition/car/moved.json
```

位姿文件含 `object_to_world` 的 4×4 矩阵。移动通过检查后写出新场景、验证报告和对应 mesh/高斯；没有物体位姿真值，不把程序阈值通过当成真实位姿精度。

## 本轮真实数据验证

全场景重新检测：`conda run --no-capture-output -n paintmesh python scripts/paintmesh/check_recomposition_scenes.py`。它逐属性比较组合高斯与真实源模型，逐顶点/逐三角形读取 glTF 背景与物体缓冲，核对世界坐标 PLY/GLB、原图 RGB/alpha/pointmap 和官方 518 裁剪记录；随后重新射线计算所有物体的 3 个留出视角以及支撑/碰撞结果。输出 `recomposition/SCENE_CHECKS.zh-CN.md`、`checks_all_scenes/report.json` 和每物体的检查图，不改变模型或验收阈值。

完整性通过只表示来源和文件正确，摆放检查另列。轮廓高 IoU 不能排除对称物体的朝向或纹理混淆；接触值达到 0.3 时可能是支撑未匹配的惩罚，不作为真实间距。移动代数检查不代替新位置的碰撞检测。

`audit_recomposition.py` 已对 11 个场景逐元素核对：合并后的背景及恢复物体的 XYZ、SH、opacity、scale、rotation 与各自来源记录相同，全部通过。明细保存在 `recomposition/original_gaussian_audit.json`。这验证高斯直接复用的完整性，不等同于生成 mesh 的位姿精度。

历史版本 car 曾完成 0.05 倍物体尺度的移动测试。这不能作为后续重新生成模型的移动验收；以当前 scene.json 和重新计算的接触/碰撞结果为准。2026-11-10 原始分辨率重生成版本的 car 仍是 needs_review，当前移动导出会拒绝。

完整场景图保留独立物体 ID 与变换；合并 PLY 是供查看器打开的扁平结果。需要在外部编辑器独立选择物体时，分别加载各 `<id>_gaussians.ply` 和背景高斯，或通过主场景图与 CLI 移动后重新导出。不要把扁平 PLY 当成自动保留对象层级的文件格式。
