import fs from 'node:fs';
import path from 'node:path';

const root = '/home/martin/code/gsagent/submodules/SemanticGaussianWrapping';
const ua = path.join(root, '.ua');
const batches = JSON.parse(fs.readFileSync(path.join(ua, 'intermediate/batches.json'), 'utf8')).batches;

const fileInfo = {
  'scene/__init__.py': ['组装数据集、相机与 GaussianModel 的场景生命周期，负责分辨率选择、训练/测试相机构建以及检查点和点云保存。', ['场景-管理','入口点','相机-装配']],
  'scene/cameras.py': ['定义完整相机与轻量 MiniCam 表示，并实现投影矩阵、图像/语义观测缩放裁剪和语义边界计算。', ['相机-模型','几何-变换','观测-预处理']],
  'scene/colmap_loader.py': ['解析 COLMAP 文本及二进制稀疏重建模型，将相机内外参、图像姿态与三维点转换为 Python 数据结构。', ['COLMAP','数据-解析','相机-标定']],
  'scene/dataset_readers.py': ['统一读取 COLMAP 与 Blender/NeRF Synthetic 数据集，构造相机载荷、点云和训练测试划分，并支持 PLY 输入输出。', ['数据集-加载','点云','相机-观测']],
  'scene/gaussian_attributes.py': ['提供可扩展 Gaussian 属性注册表，统一管理 Parameter、optimizer 状态、拓扑变更以及 checkpoint/PLY 序列化。', ['属性-注册表','optimizer','序列化']],
  'scene/gaussian_model.py': ['实现 Semantic Gaussian 核心模型，管理渲染、语义和几何属性，支持训练配置、证据更新、checkpoint 以及 densification/pruning。', ['Gaussian-模型','语义-表示','拓扑-变更']],
  'semantic/geometry_policy.py': ['将 Gaussian 的语义与几何证据转换为软策略分布，并借助邻域结构传播置信度和采样监督目标。', ['几何-策略','证据-传播','置信度']],
  'semantic/neighbor_index.py': ['实现 Gaussian 邻域索引，提供 SciPy 与精确 PyTorch 后端，并按各向异性支撑范围进行分块候选筛选和重排。', ['邻域-索引','空间-查询','内存-分块']],
  'semantic/surface_field.py': ['基于 Gaussian 邻域插值构建可微语义表面场，分块查询 SDF、法线、语义与几何置信度。', ['表面-场','可微-查询','语义-几何']],
  'tests/test_attribute_registry.py': ['验证动态 Gaussian 属性注册、optimizer 状态迁移、拓扑戳、PLY/checkpoint 往返及 COLMAP 语义观测对齐。', ['测试','属性-注册表','拓扑-一致性']],
  'tests/test_camera_intrinsics.py': ['验证非中心针孔内参得到保留，并拒绝当前管线不支持的畸变 COLMAP 相机。', ['测试','相机-内参','COLMAP']],
  'tests/test_neighbor_index.py': ['覆盖精确与 SciPy 邻域查询、各向异性支撑排序、拓扑失效机制以及候选和 workspace 内存上界。', ['测试','邻域-索引','内存-约束']],
  'tests/test_semantic_evidence.py': ['验证语义置信度传播、边界抑制、checkpoint schema 迁移以及不可变推理快照。', ['测试','语义-证据','checkpoint']],
  'tests/test_surface_field.py': ['验证表面场快照、梯度、紧凑查询上下文、分块复用、远场 SDF 与空间索引失效行为。', ['测试','表面-场','梯度-一致性']],
  'regularization/__init__.py': ['汇总并公开 regularization 子系统的损失、mesh feedback、Pareto 梯度守卫和训练阶段调度接口。', ['正则化','入口点','公共-接口']],
  'regularization/mesh_correspondence.py': ['将查询点稳健投影到 triangle mesh，结合局部尺度、法线和语义门控计算 point-to-plane correspondence。', ['mesh-对应','鲁棒-损失','几何-投影']],
  'regularization/mesh_feedback.py': ['实现异步 mesh feedback 正则器，管理 mesh 抽取、清理、质量门控、缓存恢复、平滑切换与可微损失。', ['mesh-feedback','异步-刷新','质量-门控']],
  'regularization/pareto.py': ['通过梯度投影组合 photometric 主目标与辅助目标，避免冲突梯度破坏主任务优化方向。', ['Pareto-优化','梯度-投影','多目标']],
  'regularization/scheduler.py': ['定义训练阶段及线性 ramp，按迭代生成 photometric、semantic、geometry 与 mesh 正则权重。', ['阶段-调度','损失-权重','训练-课程']],
  'tests/test_mesh_correspondence.py': ['验证 triangle projection、鲁棒 point-to-plane 损失、尺度不变性、语义门控与 CUDA 窄相计算。', ['测试','mesh-对应','CUDA']],
  'tests/test_mesh_feedback.py': ['覆盖 mesh feedback 的可微性、异步刷新、缓存迁移、异常隔离、资源清理和训练查询复用。', ['测试','mesh-feedback','异步-缓存']],
  'tests/test_mesh_feedback_v4.py': ['验证 v4 mesh feedback 的 post-commit 刷新、freshness/quality gate、平滑 promotion、尺度归一化与 checkpoint 行为。', ['测试','mesh-feedback','质量-门控']],
  'tests/test_pareto.py': ['验证 Pareto 梯度守卫能投影冲突辅助梯度，同时保留方向一致的辅助梯度。', ['测试','Pareto-优化','梯度-投影']],
};

const classInfo = {
  Scene:'协调数据集、相机集合与 GaussianModel 的加载、保存和访问。', Camera:'封装图像、相机内外参及语义观测，并支持保持标定一致的 resize/crop。', MiniCam:'保存渲染所需的轻量相机矩阵与视场参数。', Image:'表示 COLMAP 图像记录并提供四元数到旋转矩阵转换。',
  BasicPointCloud:'保存点坐标、颜色与法线的基础点云记录。', CameraPayload:'延迟保存相机 RGB、alpha 与语义观测载荷。', CameraInfo:'汇总单视角标定、姿态、图像和语义元数据。', SceneInfo:'汇总点云、训练/测试相机和场景归一化信息。',
  AttributeSpec:'描述 Gaussian 属性的形状、dtype、训练角色和序列化策略。', GaussianAttributeRegistry:'集中注册与变换 Gaussian 属性，并同步 optimizer 状态及持久化 schema。', TopologyStamp:'记录 Gaussian 拓扑版本和变更计数以驱动缓存失效。', SemanticDecoder:'将 Gaussian semantic embedding 解码为类别 logits。', GaussianModel:'管理 Gaussian 全量属性、训练状态、证据、持久化和 densification/pruning 生命周期。',
  _TensorMapping:'为 dataclass 输出提供 Mapping、to 与 detach 语义。', GeometryPolicyOutput:'承载几何策略 logits、posterior 与相关置信度输出。', SoftGeometryPolicyBank:'融合多个几何 expert，生成可正则化的软策略 posterior。', GeometryEvidenceProjector:'借助邻域属性投影几何证据并传播语义置信度。',
  _SupportBucket:'记录 ragged 支撑候选的长度分桶。', GaussianSupportAttributes:'封装支撑排序使用的 scaling、rotation 与 opacity 属性。', GaussianNeighborIndex:'维护可失效的 Gaussian 空间索引，并执行精确或 SciPy 分块邻域与支撑查询。',
  _ResultMapping:'为表面查询结果提供 Mapping、设备迁移与 detach 操作。', SurfaceQueryResult:'承载 SDF、法线、语义和置信度等表面查询结果。', SurfaceQueryContext:'缓存一次查询中可复用的紧凑 Gaussian 属性上下文。', SemanticSurfaceField:'通过邻域 Gaussian 的各向异性支撑构造可微语义表面查询。',
  TriangleProjection:'承载点到 triangle mesh 投影的位置、法线、距离与有效性。', TriangleMeshProjector:'构建 triangle 候选索引并执行带属性门控的最近点投影。', MeshSnapshotStamp:'标识 mesh 快照对应的训练步和 Gaussian 拓扑。', MeshCache:'保存清理后的 mesh、投影器及快照元数据。', MeshQualityReport:'记录候选 mesh 的覆盖率、距离和接受判据。', MeshFeedbackBatch:'承载 mesh regularization 的损失分量与覆盖信号。', _RefreshJob:'追踪后台 mesh 刷新任务和冻结快照。', MeshFeedbackRegularizer:'协调异步 mesh 抽取、候选验收、缓存切换与训练损失计算。', PhotometricParetoGuard:'投影冲突的辅助梯度并与 photometric 主梯度安全合并。', Phase:'描述一个训练阶段的边界及各损失目标权重。', PhaseScheduler:'按迭代阶段和 ramp 计算动态正则化权重。',
  _Points:'测试用最小点集模型。', _SupportPoints:'测试用 Gaussian 支撑属性模型。', _CountingSupportPoints:'统计支撑属性访问次数的测试替身。', _CountingGaussianModel:'统计 Gaussian 属性读取次数的测试替身。', QueryResult:'测试用表面查询结果。', Field:'测试用可控 surface field 替身。', Gaussians:'测试用最小 Gaussian 模型替身。', LossQuery:'测试用 mesh loss 查询载体。'
};

const functionInfo = {
  camera_from_info:'根据数据集相机信息与目标分辨率构造运行时 Camera，并保留 RGB、alpha、语义和内参。', cameraList_from_camInfos:'批量将相机信息转换为运行时 Camera 列表。', semantic_boundary_from_ids:'比较相邻 semantic id，生成像素级语义边界掩码。', resize_observations:'在统一目标尺寸下缩放 RGB、alpha、语义 id、置信度和边界观测。',
  read_model:'根据文件类型读取完整 COLMAP 稀疏模型并返回相机、图像和点云记录。', readColmapSceneInfo:'读取 COLMAP 场景、划分相机并构建或加载初始化点云。', readCamerasFromTransforms:'解析 Blender transforms JSON 并生成带 alpha/语义载荷的 CameraInfo。', readNerfSyntheticInfo:'加载 NeRF Synthetic 训练测试视角和初始化点云。', fetchPly:'从 PLY 文件读取坐标、颜色与法线。', storePly:'将坐标与 RGB 写入标准 vertex PLY。',
  quaternion_to_matrix:'将归一化 quaternion 批量转换为旋转矩阵。', _selection_mask:'把索引、布尔 mask 或切片规范化为 Gaussian 选择掩码。', _exponential_lr:'构造带 delay 的指数学习率调度函数。',
  geman_mcclure:'计算有界 Geman–McClure 鲁棒惩罚以抑制远距离 outlier。', detached_local_scale:'从 Gaussian scaling 推导停止梯度的稳定局部尺度。', robust_point_to_plane_loss:'计算按局部尺度归一化并用鲁棒核截断的 point-to-plane 损失。', _closest_points_on_triangles:'批量计算点在候选 triangle 上的最近点及 barycentric 权重。', _weighted_mean:'对有效样本计算数值稳定的加权均值。'
};

function human(name) { return name.replace(/^test_/, '').replace(/^_/, '').replaceAll('_', ' '); }
function summarizeFunction(name, file) {
  if (functionInfo[name]) return functionInfo[name];
  if (name.startsWith('test_')) return `验证 ${human(name)} 场景下的行为、数值约束与回归不变量。`;
  if (name.startsWith('read_') || name.startsWith('read')) return `读取并解析 ${human(name)} 对应的数据结构，转换为项目内部表示。`;
  if (name.startsWith('save') || name.startsWith('store')) return `将 ${human(name)} 对应状态以可恢复格式持久化。`;
  if (name.startsWith('_')) return `为 ${path.basename(file)} 提供 ${human(name)} 的内部辅助逻辑。`;
  return `实现 ${human(name)} 操作，供 ${path.basename(file)} 的主要流程复用。`;
}
function summarizeClass(name, file) { return classInfo[name] ?? `封装 ${human(name)} 的状态与操作，服务于 ${path.basename(file)} 的核心流程。`; }
function complexity(lines) { return lines > 200 ? 'complex' : lines >= 50 ? 'moderate' : 'simple'; }
function nodeTags(file, kind, name) {
  if (file.startsWith('tests/')) return ['测试','回归-验证', kind === 'class' ? '测试-替身' : '行为-契约'];
  if (kind === 'class') return ['核心-类型','状态-管理','Python'];
  return ['核心-函数','数据-处理','Python'];
}

for (const batchIndex of [4, 5]) {
  const batch = batches.find(b => b.batchIndex === batchIndex);
  const extraction = JSON.parse(fs.readFileSync(path.join(ua, `tmp/ua-file-extract-results-${batchIndex}.json`), 'utf8'));
  const nodes = [], edges = [];
  const exported = new Map();
  for (const result of extraction.results) exported.set(result.path, new Set((result.exports ?? []).map(x => x.name)));
  for (const result of extraction.results) {
    const [summary, tags] = fileInfo[result.path];
    const fileId = `file:${result.path}`;
    nodes.push({id:fileId,type:'file',name:path.basename(result.path),filePath:result.path,summary,tags,complexity:complexity(result.nonEmptyLines)});
    const exp = exported.get(result.path);
    for (const fn of result.functions ?? []) {
      const significant = (fn.endLine - fn.startLine + 1 >= 10) || exp.has(fn.name);
      if (!significant) continue;
      const id = `function:${result.path}:${fn.name}`;
      nodes.push({id,type:'function',name:fn.name,filePath:result.path,lineRange:[fn.startLine,fn.endLine],summary:summarizeFunction(fn.name,result.path),tags:nodeTags(result.path,'function',fn.name),complexity:complexity(fn.endLine-fn.startLine+1)});
      edges.push({source:fileId,target:id,type:'contains',direction:'forward',weight:1.0});
      if (exp.has(fn.name)) edges.push({source:fileId,target:id,type:'exports',direction:'forward',weight:0.8});
    }
    for (const cls of result.classes ?? []) {
      const significant = (cls.endLine-cls.startLine+1 >= 20) || ((cls.methods ?? []).length >= 2) || exp.has(cls.name);
      if (!significant) continue;
      const id = `class:${result.path}:${cls.name}`;
      nodes.push({id,type:'class',name:cls.name,filePath:result.path,lineRange:[cls.startLine,cls.endLine],summary:summarizeClass(cls.name,result.path),tags:nodeTags(result.path,'class',cls.name),complexity:complexity(cls.endLine-cls.startLine+1)});
      edges.push({source:fileId,target:id,type:'contains',direction:'forward',weight:1.0});
      if (exp.has(cls.name)) edges.push({source:fileId,target:id,type:'exports',direction:'forward',weight:0.8});
    }
  }
  for (const f of batch.files) for (const target of batch.batchImportData[f.path] ?? []) {
    edges.push({source:`file:${f.path}`,target:`file:${target}`,type:'imports',direction:'forward',weight:0.7});
    if (f.path.startsWith('tests/') && batch.files.some(x => x.path === target) && !target.startsWith('tests/')) {
      edges.push({source:`file:${target}`,target:`file:${f.path}`,type:'tested_by',direction:'forward',weight:0.5});
    }
  }
  const nodeCount=nodes.length, edgeCount=edges.length;
  const partCount=Math.ceil(Math.max(nodeCount/60,edgeCount/120));
  const files=[...batch.files].sort((a,b)=>a.path.localeCompare(b.path)).map(x=>x.path);
  const groupSize=Math.ceil(files.length/partCount);
  for (let p=0;p<partCount;p++) {
    const group=new Set(files.slice(p*groupSize,(p+1)*groupSize));
    const partNodes=nodes.filter(n=>group.has(n.filePath));
    const ids=new Set(partNodes.map(n=>n.id));
    const partEdges=edges.filter(e=>ids.has(e.source));
    fs.writeFileSync(path.join(ua,'intermediate',`batch-${batchIndex}-part-${p+1}.json`),JSON.stringify({nodes:partNodes,edges:partEdges},null,2)+'\n');
  }
  console.log(JSON.stringify({batchIndex,nodeCount,edgeCount,partCount,importsExpected:Object.values(batch.batchImportData).reduce((n,a)=>n+a.length,0),importsActual:edges.filter(e=>e.type==='imports').length,skipped:extraction.filesSkipped}));
}
