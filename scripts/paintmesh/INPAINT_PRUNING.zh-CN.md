# EDGS-PGSR inpaint：默认只剪枝，可选自适应密度控制

此策略只作用于 `run_inpaint.sh` 的 `INPAINT_PIPELINE=edgs-pgsr` 联合训练入口。没有修改 `run_seg.sh`、`submodules/EDGS/train.py`、EDGS `Trainer` 或基础重建配置，第一阶段 seg 的训练行为保持不变。

## 可选：克隆、分裂和剪枝

独立实验配置中设置 `adaptive_density.enabled: true` 即可使用 inpaint 专属的 `AdaptiveJointGaussians`。默认关闭，既有只剪枝路径继续使用原来的 `JointGaussians`。

```yaml
adaptive_density:
  enabled: true
  from_iter: 500
  until_iter: 5000
  interval: 100
  grad_threshold: 0.0002
  percent_dense: 0.01
  max_new_points: 500000
```

累计可见高斯的屏幕空间二维位置梯度范数并除以观测次数。第 600 步开始每 100 步选择超过阈值的新增高斯：最大尺度不超过 `percent_dense × extent` 时克隆，否则按原 3DGS 的二分裂规则采样两个子点并删除父点，子点尺度除以 1.6。`extent` 固定为虚拟相机中心到其均值的最大距离乘 1.1；这是本流水线的尺度归一化，不声称与官方真实相机范围相同。5000 步停止增密，不重置 opacity。

仅初始化中的新增高斯及其后代允许克隆/分裂。局部旧高斯保持原有更新/剪枝规则，远处旧高斯不参与优化或剪枝。子点继承初始祖先的语义字段；`retained_rows.npy` 在此模式中记录每个最终点的初始祖先，允许新增祖先重复，禁止旧高斯重复，发布前逐字段检查远处背景不变。

新增高斯总数上限为 50 万，达到预算时优先增密高梯度候选。每次密度操作记录克隆数、分裂父/子点数、剪枝数和预算限制数；checkpoint 同时保存动态点数、祖先映射、梯度统计、Adam 状态及 RNG。恢复时先恢复拓扑再绑定优化器。此上限用于控制 24GB 显存占用，不是自动更改梯度阈值。

## 默认配置

在 `scripts/paintmesh/configs/edgs_inpaint.yaml` 中设置：

```yaml
freeze_background: true
pruning:
  enabled: true
  min_opacity: 0.005
  until_iter: 15000
```

与 EDGS 的 `no_densify=true` 分支采用相同的不透明度规则：每次优化器更新后，如果已完成迭代数小于 `until_iter`，删除 opacity 小于 `min_opacity` 的**新增高斯**。不做克隆、分裂、增密或不透明度重置。当前 5000 步训练的每一步都处于剪枝窗口内。

旧高斯的数量、顺序和所有 PLY 字段全部保留。旧高斯不参与优化，即使其 opacity 低于阈值也不会被剪掉。此策略不能清除旧几何中的残影。若所有新增高斯都被判定应删除，训练明确报错，不会保留一个任意高斯或发布空补片。

需要固定点数作为消融对照时，在独立 YAML 中设置 `pruning.enabled: false`，并使用 `EDGS_INPAINT_CONFIG` 指向它。

## 解冻旧高斯的独立消融

在实验专用 YAML 中设置 `freeze_background: false`，或直接调用联合训练工具时传入
`--freeze-background false`。此时旧高斯和新增高斯的坐标、尺度、旋转、SH 颜色及
opacity 均参与优化，二者都执行同一个低 opacity 剪枝规则；仍不增密、不重置
opacity。语义特征保持原始行对应的值，不参与优化。

初始化的 `editable_mask.npy` 仍然表示哪些行是新增点，不会被重写为全可编辑；
优化参数另行选择全部行。保存与发布必须显式记录 `freeze_background: false`，
允许旧行发生参数变化或被删除，但仍禁止增加、重排或伪造原始行身份。
mask 外的 RGB 保留目标始终来自训练开始前的 removed 模型，不随训练更新。
诊断分别记录剩余/剪掉的旧点和新增点。默认值为 true，因此历史冻结实验和
第一阶段 seg 的行为不变。

## 局部旧高斯更新、远处严格冻结

在专用配置中保留 `freeze_background: true` 并启用 `local_background.enabled: true`。
`run_inpaint.sh` 会在联合训练前调用 `submodules/EDGS/tools/select_paintmesh_local_background.py`
生成运行目录内的选择清单，并传给联合训练工具的 `--local-selection`；恢复运行时重新验证并复用清单。
直接调用 Python 训练工具时，需要自行生成并传入清单。默认额外扩张现有训练 mask
10 像素，要求至少两个虚拟视角中的高斯投影范围与 mask 相交；同时要求旧高斯
靠近“原目标点云＋新增初始化点云”的三维支持。距离余量为初始化 spacing 的
3 倍，椭球采用 3σ 包围球，三维候选筛选中的半径最多取 spacing 的 6 倍，
防止远处巨大高斯通过二维投影被错误解冻。全程不读取 object-free GT。

`local_selection/trainable_mask.npy` 是固定的优化/剪枝许可；初始化
`editable_mask.npy` 仍只表示新增点身份。只对许可内参数建立优化器，所有许可外
旧行始终保留且全部 PLY 字段逐值不变。局部旧点和新增点均执行既有剪枝规则。
检查点行号只能来自许可集合；最终 joint 校验重新核对选择清单哈希、配置、初始
模型及远处旧行。默认局部模式关闭，不影响现有冻结/全量解冻模式或 seg。

## 可选结构与感知监督

默认 `loss.ssim=0`、`loss.lpips=0`，因此旧配置的损失保持原样。在实验配置中
设为 0.2 和 0.005 时，分别叠加全图 `1-SSIM` 与修复区域 LPIPS；原 L1 和几何
项不重新缩放。LPIPS 使用冻结的 VGG、v0.1、spatial 输出；输入转换到 [-1,1]。
取 mask 外接框加 16 像素上下文，最大边 512、最小边 32；仅 LPIPS 路径调整尺寸。
mask 外使用相同的固定目标上下文，感知梯度只传入 mask 内像素，避免全图面积
稀释修复区误差。VGG 参数不进入高斯优化器，加载不改变训练随机数状态。
实际 VGG 权重、LPIPS 标定权重及其实现文件均记录 SHA256，训练结束再次校验。
SSIM 保持全分辨率。两项损失数值写入训练诊断，最终评估协议保持不变。

## 官方式外观监督与现有几何监督组合

在独立实验 YAML 中显式设置以下选项，选择 Inpaint360GS 公开代码的外观目标：

```yaml
appearance:
  mode: inpaint360gs
loss:
  rgb_hole: 0.0
  rgb_known_preserve: 0.2
  ssim: 0.8
  lpips: 0.0005
```

此模式的 `rgb_known_preserve` 改为 **mask 外渲染 RGB 对比 LaMa RGB**，不读取
逐像素连续性权重；洞内 L1 关闭。SSIM 仍在全图计算。LPIPS 使用冻结的 VGG v0.1
非 spatial 输出，输入为原生分辨率的 mask 紧包围框，不外扩、不缩放、不替换
mask 外上下文。按公开代码的 `unfold` 顺序切成 2×2 patch 后取平均；奇数尺寸的
末尾一行／列会被省略，任一 patch 边小于 32 时该视角 LPIPS 为零。

只切换外观监督，不自动修改深度、法线、深度—法线一致性、多视图几何损失、
geometry ramp、覆盖损失、视角权重、初始化、可训练高斯范围或剪枝策略。因而
可以在上述外观目标上保留已有几何配置。公开 `common.json` 的外观权重和论文
描述不同，此模式应称为公开代码风格对照，不是完整论文复现。

默认 `appearance.mode: masked` 保留既有计算，包括洞内加权 L1、mask 外初始
removed 模型保持，以及感知梯度限于洞内的 masked LPIPS。历史 YAML 缺少 mode
时采用此默认值。`run_seg.sh` 和基础重建训练器不使用此选项。

## 可选 PGSR 尺度扁平化

`loss.scale_flatten` 默认 0；设为 100 时增加 PGSR 原式 `mean(min(scale_xyz))`，仅统计当前可见高斯。输入是激活后的正尺度，优化最短轴以促进表面扁平化。固定背景也在可见集合的归一化分母中，但其参数是 buffer，仅贡献常数。新增点与局部可训练旧点接收梯度。

此项从第一步启用，不乘 geometry ramp，不改 opacity、旧点许可或密度控制策略。实现位于 inpaint 专属 `paintmesh_joint_losses.scale_flatten_loss`；`run_seg.sh`、基础 `Trainer` 和 `PGSRLossComposer` 保持不变。该项同时进入训练、调试输出和最终诊断；checkpoint 继续保存高斯尺度与 Adam 状态。

通过 `run_appearance_scale_ablation.py --gpu 1 --detach` 可后台比较官方式外观＋完整现有几何监督下的尺度关闭/开启两组，默认各 5000 步，保留覆盖损失。完成后自动用原协议评估七项指标并更新原对比报告。

## 后台逐项累加几何损失

`scripts/paintmesh/run_geometry_ablation.py` 从一个已完成的官方式外观＋几何实验
复制输入和配置，依次运行五组独立训练：三项外观；加 depth；加 lama_normal；
加 depth_normal_consistency；加 multiview。各项开启权重沿用参考实验。覆盖
损失 `alpha_preserve` 在五组中统一为零，保证第一组只有官方三项外观损失。
depth 初始化和其他训练设置保持不变。每组都从相同初始化重新训练，彼此
不继承 checkpoint；因此结果描述固定添加顺序下的边际变化。

```bash
/home/anaconda3/bin/python scripts/paintmesh/run_geometry_ablation.py \
  --baseline-run /absolute/path/to/original_run \
  --reference-run /absolute/path/to/completed_official_appearance_geo_run \
  --gpu 1 --detach
```

默认在 `logs/official_geometry_ablation_gpu1_<timestamp>/` 保存不可变请求、总日志、
各组日志、PID 和 `status.json`。输出位于基线同级的独立目录，各组训练后立即
发布模型并评估相同 test 相机的七项 RGB 指标，再运行下一组。阶段性表格写回
基线的 `comparison_paper_bag/comparison.md`，输出根目录另有 `comparison.md`、
`summary.json` 和已完成组的 `summary.csv`。既有实验及文档其他章节保留。

恢复时运行同一脚本的 `--resume /absolute/path/to/logs/job --detach`。已完成组验证后
跳过，未完成组复用自己的训练 checkpoint 和产物；不会继承其他组模型。脚本
校验训练实现、指标实现、输入来源和配置，防止混用；同一任务由文件锁防止
重复运行，任一组失败则停止后续队列并记录错误。`--detach` 不依赖 SSH 终端。
本脚本不重新运行 LaMa、不改 seg，也不执行 mesh 后处理。

## 检查点和输出细节

- 剪枝同步压缩新增参数及 Adam 动量，删除的点不再参与后续渲染或训练。
- 检查点保存剩余新增高斯对应的初始化行号。恢复时先重建这些参数，再恢复 Adam 状态和随机数状态。
- `edgs_joint/retained_rows.npy` 为最终 PLY 每一行到初始化 PLY 的行号映射，包含全部旧高斯和未剪掉的新增高斯。它记录在 joint manifest 中并绑定文件哈希。
- 发布校验允许删除新增行，但禁止增加、重排、删除旧行或修改旧高斯字段；保留下来的语义字段和法线字段也必须保持初始化值。
- 训练日志和 `diagnostics/step_*.json` 记录剩余新增点数和累计剪枝数。`diagnostics/final.json` 保存初始新增数量、剩余新增数量、已剪枝数量、冻结旧高斯数量及剪枝配置。

## 已有实验

修改代码不会自动重训或改变磁盘上的旧模型。旧的固定点数 joint manifest 仍可校验；新训练采用新的配置和实现身份，不能直接续用旧代码的训练请求或检查点。重跑时使用新的 `INPAINT_RUN_ROOT`，或在明确需要覆盖时先清理旧训练输出，避免混用旧模型和新实验。

做初始化消融时，基线和两个新增初始化变体必须采用相同的剪枝配置；如果基线保留旧的固定点数结果，应明确说明同时存在剪枝策略差异。
