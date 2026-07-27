#!/usr/bin/env node
const fs = require('fs');

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('usage: ua-arch-assign.js <input.json> <layers.json>');
  process.exit(1);
}
const {fileNodes = []} = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const definitions = [
  ['layer:training-optimization', '训练与优化层', '编排语义条件 3DGS 训练、checkpoint 生命周期、densification 决策以及 surface、mesh feedback 与 Pareto regularization。'],
  ['layer:mesh-reconstruction', 'Mesh 重建层', '定义 surface field 采样、边界估计、拓扑修复、质量度量与语义 mesh 提取导出流水线。'],
  ['layer:semantic-reasoning', '语义推理层', '融合 Gaga 视图一致实例证据，维护邻域索引、几何策略与可查询的语义 surface field。'],
  ['layer:scene-rendering', '场景与渲染层', '管理相机与 COLMAP 数据、Gaussian 属性和模型状态，并提供训练及离线推理使用的 differentiable renderer 接口。'],
  ['layer:evaluation-workflows', '评估与命令工作流层', '提供渲染、指标评估、数据转换、安装和 surface query benchmark 等面向用户的离线工作流。'],
  ['layer:native-rasterizer', '原生 Rasterizer 层', '实现并封装自定义 C++/CUDA semantic Gaussian rasterization extension 及其 Python reference 路径。'],
  ['layer:shared-utilities', '共享工具层', '提供图形变换、损失、图像处理、配置 I/O、SH 与系统操作等跨子系统基础能力。'],
  ['layer:configuration', '配置与参数层', '集中定义训练、渲染和消融实验参数，以及 Python package、CUDA extension 和 pytest/Ruff 构建配置。'],
  ['layer:verification', '测试验证层', '覆盖训练恢复、语义证据、surface regularization、mesh、渲染器、配置和端到端工作流的回归验证。'],
  ['layer:documentation', '文档与许可层', '记录项目架构、安装使用方法、依赖、第三方归属以及主项目和 rasterizer 的许可约束。']
];
const layers = definitions.map(([id,name,description]) => ({id,name,description,nodeIds:[]}));
const byId = Object.fromEntries(layers.map(x => [x.id, x]));

function target(n) {
  const p = n.filePath;
  // Node type is the primary signal for non-code artifacts.
  if (n.type === 'document') return 'layer:documentation';
  if (n.type === 'config') return 'layer:configuration';
  // Test file patterns override their physical package location.
  if (p.startsWith('tests/') || p.includes('/tests/') || /(^|\/)test_[^/]+\.py$/.test(p)) return 'layer:verification';
  if (p.startsWith('arguments/') || p.startsWith('configs/')) return 'layer:configuration';
  if (p.startsWith('densification/') || p.startsWith('regularization/') || p.startsWith('training/') || p === 'train.py') return 'layer:training-optimization';
  if (p.startsWith('mesh/') || p === 'extract_mesh.py') return 'layer:mesh-reconstruction';
  if (p.startsWith('semantic/')) return 'layer:semantic-reasoning';
  if (p.startsWith('scene/') || p.startsWith('gaussian_renderer/') || p === 'model_io.py') return 'layer:scene-rendering';
  if (p.startsWith('evaluation/') || p.startsWith('tools/') || ['metrics.py','render.py','convert.py','install.py'].includes(p)) return 'layer:evaluation-workflows';
  if (p.startsWith('submodules/diff-semantic-gaussian-rasterization/')) return 'layer:native-rasterizer';
  if (p.startsWith('utils/')) return 'layer:shared-utilities';
  throw new Error(`unassigned node: ${n.id}`);
}
for (const n of fileNodes) byId[target(n)].nodeIds.push(n.id);
for (const layer of layers) {
  if (!layer.nodeIds.length) throw new Error(`empty layer: ${layer.id}`);
}
const assigned = layers.flatMap(x => x.nodeIds);
if (assigned.length !== fileNodes.length || new Set(assigned).size !== fileNodes.length) {
  throw new Error(`assignment mismatch: ${assigned.length}/${new Set(assigned).size}/${fileNodes.length}`);
}
fs.writeFileSync(outputPath, JSON.stringify(layers, null, 2) + '\n');
