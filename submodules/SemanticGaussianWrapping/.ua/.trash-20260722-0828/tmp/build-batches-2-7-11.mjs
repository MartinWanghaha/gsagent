import fs from 'fs';
import path from 'path';

const ROOT='/home/martin/code/gsagent/submodules/SemanticGaussianWrapping';
const UA=path.join(ROOT,'.ua');
const wanted=[2,7,8,9,10,11];
const batches=JSON.parse(fs.readFileSync(path.join(UA,'intermediate/batches.json'),'utf8')).batches;

const fileSummaries={
  'extract_mesh.py':'网格提取命令行入口：加载训练场景或 surface field，推导采样边界与分辨率，并驱动语义网格提取和导出。',
  'mesh/__init__.py':'网格子系统的公共 API 聚合层，集中暴露采样、提取、拓扑过滤、后处理、I/O 与质量评估能力。',
  'mesh/bounds.py':'根据 Gaussian 的位置、尺度和旋转估计空间支持边界，并支持选择与离群点裁剪。',
  'mesh/extractors.py':'实现 blocked marching cubes、marching tetrahedra 与 Delaunay tetrahedralization 等等值面提取算法，并负责合并和定向三角网格。',
  'mesh/field.py':'把不同模型查询接口适配为统一的语义 surface field contract，规范几何值、语义、置信度和设备转换。',
  'mesh/io.py':'提供带法线、语义标签和自定义顶点属性的 PLY/OBJ 网格读写，以及点云加载。',
  'mesh/metrics.py':'提供确定性表面采样、最近邻距离、Chamfer distance、precision/recall/F-score 等网格质量指标。',
  'mesh/pipeline.py':'编排 surface field 采样、等值面提取、语义拓扑过滤、后处理和导出的完整网格流水线。',
  'mesh/postprocess.py':'实现法线重算、退化面清理、小连通分量移除、seam-aware 聚类与面数简化。',
  'mesh/sampling.py':'定义空间 Bounds、octree 与 blocked grid 自适应采样器，依据场值、语义边界和不确定性细化采样。',
  'mesh/topology.py':'构建语义 contact graph 并根据标签、embedding 与置信度过滤网格拓扑，同时提供 seam 和连通分量分析。',
  'mesh/types.py':'定义 surface samples 与 triangle mesh 的核心数据结构，集中执行形状、dtype 和索引一致性校验。',
  'submodules/diff-semantic-gaussian-rasterization/diff_semantic_gaussian_rasterization/__init__.py':'提供可微语义 Gaussian rasterizer 的 Python API，在 native CUDA extension 与 PyTorch reference 实现之间调度并封装 autograd。',
  'submodules/diff-semantic-gaussian-rasterization/diff_semantic_gaussian_rasterization/reference.py':'以纯 PyTorch 实现 Gaussian 投影、协方差传播、tile-free alpha compositing 及 RGB、语义、深度、法线输出，作为 CUDA 对照实现。',
  'ARCHITECTURE.md':'说明 Semantic Gaussian Wrapping 的运行时数据、Gaussian state、renderer、邻域、surface field、训练、拓扑和 checkpoint contract。',
  'LICENSE.md':'项目研究用途许可证，规定源码使用、再分发、责任限制及商业用途约束。',
  'README.md':'项目总览与操作指南，覆盖安装、数据布局、训练、渲染评估、网格提取、验证和许可信息。',
  'THIRD_PARTY_NOTICES.md':'列出项目包含或改编的第三方组件及其许可归属。',
  'requirements.txt':'声明训练、图像处理、几何处理、评估和 CUDA extension 构建所需的 Python 依赖及最低版本。',
  'pyproject.toml':'定义主项目的 PEP 517 构建、包元数据、命令行入口、package data、pytest 与 Ruff 配置。',
  'configs/default.yaml':'训练与渲染的完整默认配置，覆盖模型、数据、renderer、semantic、optimization、density、surface、mesh 和 logging。',
  'configs/full.yaml':'启用完整 Semantic Gaussian Wrapping 实验方案的精简配置覆盖层。',
  'configs/full_no_confidence_propagation.yaml':'完整方案消融配置，关闭 confidence propagation 以隔离其训练影响。',
  'configs/full_no_expert_certainty.yaml':'完整方案消融配置，关闭 expert certainty 相关监督或门控。',
  'configs/full_no_mesh_feedback.yaml':'完整方案消融配置，关闭 mesh feedback regularization。',
  'configs/full_no_prune_replace.yaml':'完整方案消融配置，关闭 Gaussian prune-and-replace 机制。',
  'configs/full_no_surface_topology.yaml':'完整方案消融配置，关闭 surface topology 约束。',
  'configs/rgb_only.yaml':'RGB-only 对照实验配置，禁用语义与 surface 相关扩展并保留基础 3D Gaussian Splatting 路径。',
  'configs/semantic_render_only.yaml':'只启用 semantic rendering 的对照配置，用于隔离语义渲染本身的效果。',
  'submodules/diff-semantic-gaussian-rasterization/LICENSE.md':'语义 Gaussian rasterization 子模块继承的研究用途许可证与权利声明。',
  'submodules/diff-semantic-gaussian-rasterization/README.md':'介绍可微语义 Gaussian rasterizer 的构建方式、Python API、输出 contract 与测试方法。',
  'submodules/diff-semantic-gaussian-rasterization/pyproject.toml':'为 CUDA rasterization extension 声明最小 PEP 517 build-system 依赖。',
  'arguments/__init__.py':'构建训练、渲染和语义流程的 argparse 参数组，并合并命令行参数与已有模型配置。',
  'configs/__init__.py':'将 configs 目录标记为 Python package，供配置相关模块稳定导入。',
  'convert.py':'封装 COLMAP 特征提取、匹配、三角化、去畸变和图像缩放，将原始数据转换为训练布局。',
  'install.py':'检测 CUDA 环境并安装语义 Gaussian rasterization extension，可转发显式 GPU architecture 设置。',
  'submodules/diff-semantic-gaussian-rasterization/cuda_rasterizer/rasterize.cu':'实现 RGB、语义、深度、alpha 与法线联合输出的 CUDA Gaussian rasterization forward/backward kernels，并以 dual number 回放投影导数。',
  'submodules/diff-semantic-gaussian-rasterization/ext.cpp':'通过 pybind11 注册 native rasterizer 的 forward/backward C++ 接口。',
  'submodules/diff-semantic-gaussian-rasterization/rasterize.h':'声明语义 Gaussian rasterizer 的 C++ forward/backward extension API。',
  'submodules/diff-semantic-gaussian-rasterization/setup.py':'配置并构建 PyTorch CUDAExtension，处理 headless/PEP 517 环境与 CUDA architecture 默认值。',
  'utils/__init__.py':'将通用工具目录标记为 Python package。',
  'utils/sh_utils.py':'实现 RGB 与 spherical harmonics 表示转换及分阶 SH 求值。',
  'utils/system_utils.py':'提供递归建目录和扫描最大 checkpoint iteration 的文件系统辅助函数。'
};

function isTest(p){return /(^|\/)tests?\//.test(p)||/(^|\/)test_/.test(p)}
function fileSummary(p){
  if(fileSummaries[p]) return fileSummaries[p];
  if(isTest(p)) return `验证 ${path.basename(p,'.py').replace(/^test_/,'').replaceAll('_',' ')} 相关行为、边界条件与回归约束的自动化测试。`;
  return `实现 ${path.basename(p).replaceAll('_',' ')} 相关功能，并作为项目内部代码组件参与运行。`;
}
function complexity(nonEmpty,total){const n=nonEmpty??total??0; return n>200?'complex':n>=50?'moderate':'simple'}
function fileTags(p,cat){
  if(isTest(p)) return ['test','回归测试','质量保障'];
  if(cat==='docs') return p.includes('LICENSE')?['documentation','许可','合规']:['documentation','项目指南','参考资料'];
  if(cat==='config') return ['configuration',p.endsWith('.yaml')?'实验配置':'build-system','可复现性'];
  if(p.endsWith('__init__.py')) return ['entry-point','barrel','公共-api'];
  if(p.endsWith('.cu')) return ['cuda','rasterization','autograd','gpu-kernel'];
  if(p.endsWith('.h')||p.endsWith('.cpp')) return ['cuda-extension','native-binding','rasterization'];
  if(p.includes('mesh/')) return ['mesh','geometry','surface-reconstruction'];
  return ['python','核心逻辑','工具函数'];
}
function symbolSummary(name,p,type){
  const readable=name.replace(/^_+/,'').replace(/([a-z])([A-Z])/g,'$1 $2').replaceAll('_',' ');
  if(name.startsWith('test_')) return `验证 ${readable.slice(5)} 的预期行为及关键回归条件。`;
  if(type==='class') return `封装 ${readable} 的状态与操作，承担 ${path.basename(p)} 中的核心领域职责。`;
  const verbs={load:'加载并规范化',write:'写出',read:'读取并解析',compute:'计算',sample:'采样',extract:'提取',build:'构建',infer:'推导',filter:'过滤',project:'投影',rasterize:'光栅化',main:'执行命令行流程',configure:'配置',convert:'转换',eval:'求值',remove:'移除',merge:'合并',simplify:'简化',postprocess:'后处理'};
  const k=Object.keys(verbs).find(x=>name.toLowerCase().includes(x));
  return `${k?verbs[k]:'实现'} ${readable} 相关逻辑，供 ${path.basename(p)} 的处理流程复用。`;
}
function symTags(name,type,p){
  if(name.startsWith('test_')) return ['test','回归测试','边界条件'];
  if(type==='class') return ['data-model','领域对象',p.includes('raster')?'rasterization':'核心组件'];
  return ['utility',p.endsWith('.cu')?'cuda-kernel':'核心逻辑',name.startsWith('_')?'内部-api':'公共-api'];
}
function nodeForFile(f,r){
  const typ=f.fileCategory==='config'?'config':f.fileCategory==='docs'?'document':'file';
  const prefix=typ;
  return {id:`${prefix}:${f.path}`,type:typ,name:path.basename(f.path),filePath:f.path,summary:fileSummary(f.path),tags:fileTags(f.path,f.fileCategory),complexity:complexity(r?.nonEmptyLines,f.sizeLines)};
}
function exportedSet(r){return new Set((r.exports||[]).map(x=>x.name))}
function createBatch(batch){
  const ex=JSON.parse(fs.readFileSync(path.join(UA,'tmp',`ua-file-extract-results-${batch.batchIndex}.json`),'utf8'));
  const rm=new Map(ex.results.map(x=>[x.path,x]));
  const nodes=[],edges=[];
  for(const f of batch.files){
    const r=rm.get(f.path); const fn=fileNodeId(f);
    nodes.push(nodeForFile(f,r));
    if(!r) continue;
    const exported=exportedSet(r);
    for(const item of [...(r.functions||[]).map(x=>({...x,_type:'function'})),...(r.classes||[]).map(x=>({...x,_type:'class'}))]){
      const span=item.endLine-item.startLine+1;
      const significant=exported.has(item.name)||(item._type==='function'?span>=10:((item.methods||[]).length>=2||span>=20));
      if(!significant) continue;
      const id=`${item._type}:${f.path}:${item.name}`;
      nodes.push({id,type:item._type,name:item.name,filePath:f.path,lineRange:[item.startLine,item.endLine],summary:symbolSummary(item.name,f.path,item._type),tags:symTags(item.name,item._type,f.path),complexity:complexity(span,span)});
      edges.push({source:fn,target:id,type:'contains',direction:'forward',weight:1.0});
      if(exported.has(item.name)) edges.push({source:fn,target:id,type:'exports',direction:'forward',weight:0.8});
    }
  }
  // Supplement significant structure in skipped CUDA source.
  if(batch.batchIndex===11){
    const p='submodules/diff-semantic-gaussian-rasterization/cuda_rasterizer/rasterize.cu', fid=`file:${p}`;
    const dualId=`class:${p}:Dual`;
    nodes.push({id:dualId,type:'class',name:'Dual',filePath:p,lineRange:[66,73],summary:'封装单切向 dual number 的数值与导数分量，为 CUDA 投影反向传播提供轻量 forward-mode differentiation。',tags:['data-model','autograd','cuda'],complexity:'simple'});
    edges.push({source:fid,target:dualId,type:'contains',direction:'forward',weight:1.0});
    const funcs=[['quaternion_matrix_dual',130,152],['project_one_dual',154,319],['quaternion_matrix',321,340],['project_one',342,483],['tile_bounds',485,499],['project_and_count_kernel',501,567],['duplicate_with_keys_kernel',569,602],['identify_tile_ranges_kernel',604,617],['render_kernel',619,721],['evaluate_alpha',723,749],['render_backward_kernel',751,939],['projection_backward_kernel',941,1008],['semantic_gaussian_rasterize_forward',1010,1179],['semantic_gaussian_rasterize_backward',1181,1363]];
    for(const [name,s,e] of funcs){const id=`function:${p}:${name}`;nodes.push({id,type:'function',name,filePath:p,lineRange:[s,e],summary:symbolSummary(name,p,'function'),tags:symTags(name,'function',p),complexity:complexity(e-s+1,e-s+1)});edges.push({source:fid,target:id,type:'contains',direction:'forward',weight:1.0});if(name==='semantic_gaussian_rasterize_forward'||name==='semantic_gaussian_rasterize_backward')edges.push({source:fid,target:id,type:'exports',direction:'forward',weight:0.8});}
  }
  for(const f of batch.files){
    if(f.fileCategory!=='code') continue;
    for(const target of batch.batchImportData[f.path]||[]) edges.push({source:fileNodeId(f),target:`file:${target}`,type:'imports',direction:'forward',weight:0.7});
  }
  return {nodes,edges,skipped:ex.filesSkipped||[]};
}
function fileNodeId(f){return `${f.fileCategory==='config'?'config':f.fileCategory==='docs'?'document':'file'}:${f.path}`}
function writeParts(i,g,batch){
  const count=Math.ceil(Math.max(g.nodes.length/60,g.edges.length/120,1));
  const files=[...batch.files].sort((a,b)=>a.path.localeCompare(b.path));
  const size=Math.ceil(files.length/count); const written=[];
  for(let k=0;k<count;k++){
    const paths=new Set(files.slice(k*size,(k+1)*size).map(x=>x.path));
    const ns=g.nodes.filter(n=>paths.has(n.filePath)); const ids=new Set(ns.map(n=>n.id));
    const es=g.edges.filter(e=>ids.has(e.source));
    const out={nodes:ns,edges:es}; const name=count===1?`batch-${i}.json`:`batch-${i}-part-${k+1}.json`;
    fs.writeFileSync(path.join(UA,'intermediate',name),JSON.stringify(out,null,2)+'\n'); written.push(name);
  }
  return written;
}

const report=[];
for(const i of wanted){
  const batch=batches.find(x=>x.batchIndex===i); const g=createBatch(batch); const files=writeParts(i,g,batch);
  const expected=Object.values(batch.batchImportData).reduce((a,x)=>a+x.length,0);
  const actual=g.edges.filter(x=>x.type==='imports').length;
  if(expected!==actual) throw new Error(`batch ${i}: import edges ${actual} != ${expected}`);
  const ids=new Set(g.nodes.map(n=>n.id)); if(ids.size!==g.nodes.length) throw new Error(`batch ${i}: duplicate nodes`);
  report.push({batchIndex:i,parts:files,nodes:g.nodes.length,edges:g.edges.length,imports:actual,skipped:g.skipped});
}
console.log(JSON.stringify(report,null,2));
