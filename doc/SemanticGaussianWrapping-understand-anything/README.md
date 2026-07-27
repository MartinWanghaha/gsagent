# SemanticGaussianWrapping 代码知识图谱

本目录由 Understand-Anything 对 `submodules/SemanticGaussianWrapping` 进行完整分析后生成。分析基于仓库提交 `b95232b1869ca342be2d221a4f64425ebfdc057b`，输出语言为简体中文。

## 分析规模

| 项目 | 数量 |
|---|---:|
| 扫描文件 | 120 |
| code / config / docs | 102 / 11 / 7 |
| 图节点 | 790 |
| 文件 / 函数 / 类 / 文档 / 配置节点 | 102 / 569 / 101 / 7 / 11 |
| 图关系 | 1509 |
| imports / contains / exports / tested_by | 161 / 670 / 657 / 21 |
| 架构层 | 10 |
| 导览步骤 | 11 |

检测语言包括 Python、C、C++、CUDA、YAML、Markdown、TOML 和 TXT。图谱通过最终结构验证，没有缺失文件、重复层归属、dangling edge 或无效导览引用。

## 架构分层

| 层 | 文件数 | 职责 |
|---|---:|---|
| 训练与优化层 | 13 | 训练编排、checkpoint、densification、surface/mesh feedback 与 Pareto regularization |
| Mesh 重建层 | 12 | surface field 采样、边界估计、拓扑修复、质量度量与 mesh 导出 |
| 语义推理层 | 5 | Gaga 实例证据、邻域索引、几何策略及语义 surface field |
| 场景与渲染层 | 8 | 相机与 COLMAP 数据、Gaussian 状态和 differentiable renderer 接口 |
| 评估与命令工作流层 | 7 | 渲染、指标、转换、安装及 surface query benchmark |
| 原生 Rasterizer 层 | 6 | C++/CUDA semantic Gaussian rasterization extension 与 reference 路径 |
| 共享工具层 | 9 | 图形变换、损失、图像、配置 I/O、SH 与系统工具 |
| 配置与参数层 | 13 | 训练/渲染/消融配置及 Python、CUDA、pytest/Ruff 构建配置 |
| 测试验证层 | 40 | 训练恢复、语义证据、surface、mesh、renderer 和端到端回归测试 |
| 文档与许可层 | 7 | 架构、安装、依赖、第三方归属和许可证 |

## 推荐阅读导览

1. 项目全景：`README.md`
2. 架构契约：`ARCHITECTURE.md`
3. 训练入口与配置：`train.py`、`configs/default.yaml`
4. 四阶段训练引擎：`training/engine.py`、`training/checkpointing.py`
5. 数据与 Gaussian 模型：数据读取、属性注册、模型状态与 PLY/checkpoint I/O
6. 可微联合渲染：Python renderer、自定义 C++/CUDA rasterizer 与离线渲染入口
7. 语义与表面场：Gaga adapter、neighbor index、geometry policy、surface field
8. 正则化与密度控制：联合损失、surface consistency、densification、mesh feedback、Pareto guard
9. 拓扑与 Mesh 精炼：mesh pipeline、field、topology、postprocess 和导出入口
10. 双指标评估：RGB 指标与 mesh 指标
11. 端到端验证：smoke、reference rasterizer、mesh topology 和 metrics CLI 测试

完整步骤说明、节点摘要、层归属和边关系均包含在 `knowledge-graph.json` 中。

## 文件说明

- `knowledge-graph.json`：完整 Understand-Anything KnowledgeGraph，可供 dashboard、检索和后续问答使用。
- `scan-result.json`：120 个文件的确定性清单、语言、行数、类别及内部 import map。
- `fingerprints.json`：结构 fingerprints，用于后续增量分析。
- `review.json`：最终结构验证结果和统计。
- `meta.json`：分析时间、Git commit 和文件数量。

## 验证警告

最终验证为 **0 issues**。共有 20 个 orphan warnings，集中于 README/ARCHITECTURE、许可证、YAML 消融配置、`pyproject.toml` 和空 `__init__.py`。这些节点被完整保留，但没有为了消除 warning 而虚构依赖关系。

