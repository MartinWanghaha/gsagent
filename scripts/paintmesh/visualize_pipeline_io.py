#!/usr/bin/env python3
"""Read-only, CPU visual audit of the exact runs selected in best_results.

PNG/JPEG panels are visualizations of saved data, not new model predictions.
Every figure has a source manifest; missing intermediate data are explicit.
"""
import argparse
import gc
import html
import io
import json
import math
import os
from pathlib import Path
import traceback
import zipfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
BEST = ROOT / 'output/paintmesh/Inpaint360GS/best_results'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FP = FontProperties(fname=FONT)
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
SCENES = ['bag', 'car', 'cone_red', 'cone_yellow', 'cube', 'doppelherz',
          'fruits', 'garden_toys', 'redbull', 'toys', 'truck']
TITLES = {
 '01_reconstruction': 'Seg 1｜输入照片 → EDGS / PGSR 重建',
 '02_segmentation': 'Seg 2–5｜逐帧分割 → ID 关联 → 语义蒸馏',
 '03_semantic_mesh': 'Seg 6｜高斯语义 → Mesh 语义提升',
 '04_removal': 'Remove 1–3｜选中物体 → 高斯删除 → 几何变化',
 '05_virtual_cameras': 'Remove 4｜虚拟相机轨迹与图像连续性',
 '06_tracking': 'Remove 5 / Inpaint 2｜跟踪 mask → 膨胀 mask',
 '07_lama_rgb': 'Inpaint 2–3｜LaMa RGB 输入 → 补全输出',
 '08_lama_depth': 'Inpaint 2–3｜LaMa 深度输入 → 补全输出',
 '09_lama_normal': 'Inpaint 2–3｜LaMa 法线输入 → 补全输出',
 '10_correspondence': 'Inpaint 4｜特征匹配 / 三角化 / 锚点证据',
 '11_initialization': 'Inpaint 4｜混合初始化 → 初始高斯中心',
 '12_local_background': 'Inpaint 5 前处理｜局部可更新 / 冻结背景',
 '13_training_rgb': 'Inpaint 5｜固定训练视角的外观变化',
 '14_training_geometry': 'Inpaint 5｜固定训练视角的几何变化',
 '15_training_losses': 'Inpaint 5｜监督、剪枝与点数诊断',
 '16_final_geometry': 'Inpaint 6–8｜发布模型 / Mesh / 最终几何',
 '17_test_evaluation': 'Evaluation｜测试 GT / 渲染 / mask / 误差',
}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

def pathof(v):
    return Path(v['path'] if isinstance(v, dict) else v)

def font(n):
    return ImageFont.truetype(FONT, n)

def sample(a, n=6):
    a = list(a)
    return [a[i] for i in np.unique(np.linspace(0, len(a)-1, min(n, len(a))).astype(int))] if a else []

def image(a):
    if isinstance(a, Image.Image):
        return a.convert('RGB')
    a = np.asarray(a)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif np.issubdtype(a.dtype, np.floating):
        a = np.uint8(np.clip(np.nan_to_num(a), 0, 1)*255+.5)
    elif a.dtype != np.uint8:
        a = np.uint8(np.clip(a, 0, 255))
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    return Image.fromarray(a).convert('RGB')

def missing(text='未保存 / 不适用', size=(520, 320)):
    im = Image.new('RGB', size, '#f0f1f4')
    d = ImageDraw.Draw(im)
    d.text((18, 60), text, font=font(24), fill='#9e2e32')
    return im

def overlay(rgb, mask, color=(255, 70, 70), alpha=.4):
    im = image(rgb)
    m = np.asarray(mask).astype(bool)
    if m.shape != (im.height, im.width):
        m = np.asarray(Image.fromarray(m).resize(im.size, Image.Resampling.NEAREST))
    a = np.asarray(im).astype(float)
    a[m] = a[m]*(1-alpha)+np.array(color)*alpha
    boundary = m & ~np.asarray(Image.fromarray(m.astype(np.uint8)*255).filter(ImageFilter.MinFilter(3))).astype(bool)
    a[boundary] = color
    return image(a.astype(np.uint8))

def bbox(m, pad=.25):
    yy, xx = np.where(m)
    if not len(xx):
        return (0, 0, m.shape[1], m.shape[0])
    w, h = xx.max()-xx.min()+1, yy.max()-yy.min()+1
    p = max(12, int(max(w, h)*pad))
    return (max(0, int(xx.min())-p), max(0, int(yy.min())-p),
            min(m.shape[1], int(xx.max())+p+1), min(m.shape[0], int(yy.max())+p+1))

def crop(im, m):
    return image(im).crop(bbox(m))

def heat(a, lo=0, hi=1, cmap='magma', valid=None):
    a = np.asarray(a)
    rgb = plt.get_cmap(cmap)(np.clip((np.nan_to_num(a)-lo)/max(hi-lo, 1e-12), 0, 1))[..., :3]
    if valid is not None:
        rgb[~valid] = 0
    return image(rgb)

def normal_vis(a, valid=None):
    v = np.isfinite(a).all(-1) & (np.linalg.norm(a, axis=-1) > .1)
    if valid is not None:
        v &= valid
    rgb = np.clip(a*.5+.5, 0, 1)
    rgb[~v] = 0
    return image(rgb)

def id_color(a):
    a = np.asarray(a).astype(np.int64)
    if a.ndim == 3:
        return image(a)
    col = np.stack(((a*67+31) % 255, (a*131+63) % 255, (a*197+97) % 255), -1).astype(np.uint8)
    col[a == 0] = 0
    return image(col)

def fig_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def text_panel(lines, width=1050, height=700, fs=25):
    im = Image.new('RGB', (width, height), '#fafbfc')
    d = ImageDraw.Draw(im)
    y = 18
    for line in lines:
        line = str(line)
        chunk = max(22, int(width/(fs*.64)))
        for start in range(0, max(1, len(line)), chunk):
            d.text((20, y), line[start:start+chunk], fill='#183247', font=font(fs))
            y += int(fs*1.6)
    return im

def ply_vertices(p):
    """Map only the scalar vertex block. Never read giant mesh face arrays."""
    types = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1',
             'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2', 'char': 'i1',
             'float32': 'f4', 'float64': 'f8', 'int32': 'i4', 'uint16': 'u2'}
    with Path(p).open('rb') as f:
        header, props, active, count, fmt = [], [], False, None, None
        for _ in range(1000):
            line = f.readline().decode('ascii').strip()
            header.append(line)
            z = line.split()
            if z[:1] == ['format']: fmt = z[1]
            if z[:2] == ['element', 'vertex']: count, active = int(z[2]), True
            elif z[:1] == ['element']: active = False
            if z[:1] == ['property'] and active:
                if z[1] == 'list': raise ValueError('unsupported list vertex property')
                props.append((z[2], types[z[1]]))
            if line == 'end_header': break
        offset = f.tell()
    if fmt not in ('binary_little_endian', 'binary_big_endian'):
        from plyfile import PlyData
        return PlyData.read(str(p))['vertex'].data
    endian = '<' if fmt == 'binary_little_endian' else '>'
    return np.memmap(p, mode='r', offset=offset, dtype=np.dtype([(k, endian+t) for k,t in props]), shape=(count,))

def xyz(v, indices=None):
    if indices is not None: v = v[indices]
    return np.stack([v[k] for k in ['x', 'y', 'z']], -1)

class Audit:
    def __init__(self, scene, out):
        self.scene, self.b = scene, BEST/scene
        self.out = out/scene
        self.out.mkdir(parents=True, exist_ok=True)
        self.sel = read(self.b/'selection.json')
        self.cfg = read(self.b/'training_config.json')
        self.step = self.sel['configuration']['iteration']
        self.run = (self.b/'run').resolve()
        self.j = read(self.b/'joint_manifest.json')
        self.init = read(self.b/'initialization_manifest.json')
        self.lama = read(self.b/'lama_manifest.json')
        self.li = read(self.lama['input_manifest']['path'])
        self.cam = read(self.b/'camera.json')
        self.rem = pathof(self.j['inputs']['camera']).parent.parent
        self.base = self.rem.parent.parent
        self.result = read(self.b/'results.json')
        self.data = Path(self.result['inputs']['source_path'])
        self.frames = sorted(self.li['frames'])
        self.fsel = sample(self.frames)
        self.sources, self.figs, self.problems = {}, [], []
        self.key = 'initial'
        self.diag = read(pathof(self.init['outputs']['diagnostics']))
        self.support = dict(np.load(pathof(self.init['outputs']['support'])))
        self.edit = np.load(pathof(self.init['outputs']['editable_mask'])).astype(bool)
        self.vertices = ply_vertices(pathof(self.init['outputs']['ply']))
        assert len(self.edit) == len(self.vertices)
        self.new_xyz = xyz(self.vertices, np.flatnonzero(self.edit))
        self.center = np.median(self.new_xyz, axis=0)
        self.radius = max(float(np.linalg.norm(np.percentile(self.new_xyz, 98, axis=0)-np.percentile(self.new_xyz, 2, axis=0))), .001)
        ids = np.arange(0, len(self.vertices), max(1, len(self.vertices)//160000))
        self.old_sample = xyz(self.vertices, ids[~self.edit[ids]])
        local = self.old_sample[np.linalg.norm(self.old_sample-self.center, axis=1) < self.radius*1.3]
        if len(local) < 30: local = self.new_xyz
        _, _, self.axes = np.linalg.svd((local-local.mean(0))[::max(1, len(local)//50000)], full_matrices=False)
        self.evidence = dict(np.load(pathof(self.init['outputs']['surface_evidence']))) if 'surface_evidence' in self.init['outputs'] else {}
        self.local = read(pathof(self.j['inputs']['local_selection'])) if 'local_selection' in self.j['inputs'] else None
        self.removed = read(pathof(self.init['inputs']['removed_model']))
        self.ids = self.removed.get('parameters', {}).get('target_ids', [])
        self.stats = {'scene': scene, 'selected_step': self.step, 'training_budget': self.cfg['iterations'],
                      'virtual_frames': len(self.frames), 'test_frames': self.result['frame_count'], 'target_ids': self.ids}
        self.sources['manifest_chain'] = {str((self.b/k).resolve()) for k in
            ['selection.json','training_config.json','joint_manifest.json','camera.json',
             'lama_manifest.json','initialization_manifest.json','results.json']}
        self.sources['manifest_chain'].add(str(pathof(self.lama['input_manifest']).resolve()))

    def start(self, key):
        self.key = key
        self.sources.setdefault(key, set())

    def record(self, p):
        if p is not None: self.sources.setdefault(self.key, set()).add(str(Path(p).resolve()))

    def load(self, p):
        self.record(p)
        if not Path(p).is_file():
            self.problems.append({'module': self.key, 'missing': str(p)})
            return None
        if str(p).endswith('.npy'): return np.load(p)
        with Image.open(p) as im: return np.array(im)

    def im(self, p):
        a = self.load(p)
        return image(a) if a is not None else missing()

    def virtual(self, f):
        entry, out = self.li['frames'][f], self.lama['frames'][f]['outputs']
        inp = entry['inputs']; pre = Path(inp['reference_depth']['path']).parent.parent
        rgb0 = self.load(pre/'renders'/f'{f}.png')
        rgb1 = self.load(pathof(inp['removed_rgb']))
        mask = self.load(pathof(inp['mask']))
        dil = self.load(pathof(entry['outputs']['color_mask']))
        rgb2 = self.load(pathof(out['color']))
        return rgb0, rgb1, mask > 0, dil > 0, rgb2

    def tracking_rgb(self, f):
        # The tracker sees the removal-render archive, not the intact object.
        tracker = pathof(self.li['frames'][f]['inputs']['mask']).parents[3]
        archive = tracker/'images.zip'
        self.record(archive)
        if not archive.is_file():
            self.problems.append({'module':self.key,'missing':str(archive)})
            return None
        with zipfile.ZipFile(archive) as z:
            with Image.open(io.BytesIO(z.read(f+'.png'))) as im:
                return np.array(im.convert('RGB'))

    def save_grid(self, key, rows, columns, notes='', labels=None, tilew=510, tileh=340):
        self.start(key)
        ncol = len(columns)
        top, gap, left, rowh = 200, 12, 150 if labels else 20, tileh+52
        width = left+ncol*(tilew+gap)+20
        canvas = Image.new('RGB', (width, top+len(rows)*rowh+75), '#f7f9fc')
        d = ImageDraw.Draw(canvas)
        title = TITLES.get(key, key)
        d.text((22, 15), f'{self.scene} · {title}', fill='#15354d', font=font(35))
        d.text((22, 69), f'实际选中模型：{self.step:,} 步 / 训练预算 {self.cfg["iterations"]:,} 步   |   {len(self.frames)} 个虚拟视角   |   target ID={self.ids}', fill='#30485b', font=font(23))
        # Wrap the explanatory note without losing units or display conventions.
        chars = max(40, int(width/15))
        for i, line in enumerate([notes[j:j+chars] for j in range(0, len(notes), chars)][:2]):
            d.text((22, 108+i*29), line, fill='#555f6b', font=font(21))
        for c, label in enumerate(columns):
            d.text((left+c*(tilew+gap)+7, top-35), label, fill='#174669', font=font(24))
        for r, row in enumerate(rows):
            if labels:
                for i, line in enumerate(str(labels[r]).split('\n')):
                    d.text((12, top+r*rowh+20+i*29), line, fill='#183247', font=font(20))
            for c, cell in enumerate(row):
                im, caption = cell if isinstance(cell, tuple) else (cell, '')
                im = missing() if im is None else image(im)
                fit = ImageOps.contain(im, (tilew, tileh), Image.Resampling.LANCZOS)
                x, y = left+c*(tilew+gap), top+r*rowh
                d.rectangle((x, y, x+tilew, y+tileh), fill='#e5e9ee')
                canvas.paste(fit, (x+(tilew-fit.width)//2, y+(tileh-fit.height)//2))
                d.text((x+4, y+tileh+5), caption[:48], fill='#374c5e', font=font(18))
        d.text((22, canvas.height-45), 'CPU 读取已有产物；颜色/误差/投影均为诊断可视化。原文件路径见 sources.json；原始分辨率保留在文件中。', fill='#5b6674', font=font(21))
        p = self.out/f'{key}.jpg'
        canvas.save(p, quality=95, subsampling=0)
        item={'key': key, 'file': p.name, 'title': title, 'width': canvas.width, 'height': canvas.height}
        existing=next((i for i,v in enumerate(self.figs) if v['key']==key),None)
        if existing is None:self.figs.append(item)
        else:self.figs[existing]=item
        return p

    def save_plot(self, key, im, note=''):
        return self.save_grid(key, [[im]], [''], notes=note, tilew=2800, tileh=int(image(im).height/image(im).width*2800))

    def cloud(self, groups, title, limit=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        lim = limit or self.radius*.68
        for ax, ij, name in zip(axes, [(0,1), (0,2), (1,2)], ['PCA 1/2', 'PCA 1/3', 'PCA 2/3']):
            for points, color, label in groups:
                points = points[np.isfinite(points).all(1)]
                points = points[::max(1, len(points)//22000)]
                v = (points-self.center)@self.axes.T
                keep = (np.abs(v[:,ij[0]])<lim) & (np.abs(v[:,ij[1]])<lim)
                ax.scatter(v[keep,ij[0]], v[keep,ij[1]], s=.8, c=color, alpha=.6, label=label, rasterized=True)
            ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect('equal');ax.grid(alpha=.2)
            ax.set_title(name);ax.set_xlabel('world units');ax.legend(markerscale=5, fontsize=8, loc='upper right')
        fig.suptitle(title, fontproperties=FP, fontsize=16)
        return fig_image(fig)

    def reconstruction(self):
        self.start('01_reconstruction')
        ed = self.base/'edgs/train/ours_30000'
        files = sample(sorted((ed/'renders').glob('*.png')))
        self.real = [p.name for p in files]
        rows = []
        for f in self.real:
            gt, pred = self.load(ed/'gt'/f), self.load(ed/'renders'/f)
            err = np.abs(gt.astype(float)-pred.astype(float)).mean(-1)/255 if gt.shape==pred.shape else None
            rows.append([(image(gt),f'{f} / {gt.shape[1]}×{gt.shape[0]}'), image(pred),
                         heat(err,0,.25) if err is not None else missing('尺寸不一致'),
                         self.im(ed/'renders_depth'/f),self.im(ed/'renders_normal'/f)])
        self.save_grid(self.key, rows, ['输入 RGB（训练相机）','输出重建 RGB','RGB |差| 0→0.25','导出的深度预览','导出的法线预览'],
                       '按训练视角序列均匀抽样；误差黑→黄代表增大。此阶段深度 PNG 是原导出预览，不与后续数值深度共用色标。', [Path(f).stem for f in self.real])

    def segmentation(self):
        self.start('02_segmentation');rows=[]
        sem=self.base/'semantic_3dgs/train/ours_2000'
        for f in self.real:
            rgb=self.load(sem/'gt'/f);raw=self.load(self.data/'raw_hqsam'/f);assoc=self.load(self.data/'associated_hqsam'/f);pred=self.load(sem/'objects_pred'/f)
            rows.append([image(rgb),id_color(raw) if raw is not None else missing(),id_color(assoc) if assoc is not None else missing(),id_color(pred) if pred is not None else missing(),overlay(rgb,np.isin(pred,self.ids)) if pred is not None else missing()])
        self.save_grid(self.key,rows,['输入图像','逐帧 raw ID','关联后的全局 ID','蒸馏后的预测 ID','选中 ID 在图像上的投影'],
                       '同一数字使用同一伪彩；raw ID 仅在单帧有意义，不能跨帧按颜色认同一个物体。红色是选中目标，不是测试 GT mask。',[Path(f).stem for f in self.real])

    def semantic_mesh(self):
        self.start('03_semantic_mesh');root=self.base/'semantic_mesh';p=root/'semantic_manifest.json';self.record(p)
        m=read(p);vp=pathof(m['inputs']['gaussian_ply']);self.record(vp);v=ply_vertices(vp)
        idx=np.arange(0,len(v),max(1,len(v)//150000));points=xyz(v,idx)
        lab=self.load(root/'gaussian_instance_id.npy')[idx]
        g1=self.cloud([(points[~np.isin(lab,self.ids)],'#869eae','other Gaussians'),(points[np.isin(lab,self.ids)],'#e14d4d','target ID')],'语义高斯中心：目标与其他高斯')
        mp=root/'semantic_mesh.ply';self.record(mp);mv=ply_vertices(mp);mi=np.arange(0,len(mv),max(1,len(mv)//180000));pts=xyz(mv,mi)
        labs=np.load(root/'vertex_instance_id.npy',mmap_mode='r')[mi];self.record(root/'vertex_instance_id.npy')
        g2=self.cloud([(pts[~np.isin(labs,self.ids)],'#869eae','other mesh vertices'),(pts[np.isin(labs,self.ids)],'#e14d4d','target vertices')],'语义 Mesh 顶点：相同坐标、相同视窗')
        self.save_grid(self.key,[[g1],[g2]],['三正交投影；顶点/中心抽样，不是表面光栅化图'],
                       '上：高斯中心标签；下：Mesh 顶点标签。红色检查目标覆盖与外泄。'+str(m.get('counts',{})),tilew=2800,tileh=800)

    def removal(self):
        self.start('04_removal');rows=[]
        for f in self.fsel:
            before,after,mask,dil,out=self.virtual(f)
            a=self.load(pathof(self.li['frames'][f]['inputs']['removed_alpha']))
            err=np.abs(before.astype(float)-after.astype(float)).mean(-1)/255
            rows.append([image(before),overlay(before,mask),image(after),heat(err,0,.3),heat(a.squeeze(),0,1,'gray'),crop(after,dil)])
        self.save_grid(self.key,rows,['删除前虚拟 RGB','跟踪 mask 覆盖位置','删除后虚拟 RGB','删除前后 RGB 差 0→0.3','删除后 alpha 0→1','删除区域放大'],
                       '删除前后使用同一份相机；mask 是已保存的最终跟踪输出。颜色变化显示影响范围，不能单独证明 3D 点已删除。',self.fsel)

    def cameras(self):
        self.start('05_virtual_cameras');self.record(self.b/'camera.json')
        cams=self.cam['cameras'];centers=[];rots=[]
        for c in cams:
            w=np.eye(4);w[:3,:3]=np.asarray(c['R']).T;w[:3,3]=c['T'];inv=np.linalg.inv(w)
            centers.append((inv[:3,3]+np.asarray(c.get('trans',[0,0,0])))*c.get('scale',1))
            u,_,vt=np.linalg.svd(w[:3,:3]);rots.append(u@vt)
        centers=np.array(centers);rots=np.array(rots);delta=np.einsum('nij,nkj->nik',rots[1:],rots[:-1]);deg=np.degrees(np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1)))
        c=(centers-self.center)@self.axes.T
        fig=plt.figure(figsize=(16,9));ax=fig.add_subplot(221,projection='3d');ax.plot(*c.T,lw=1,color='gray');ax.scatter(*c.T,c=np.arange(len(c)),cmap='viridis',s=12);ax.scatter(0,0,0,c='r',s=60,marker='x');ax.set_title('Camera centers; red = new-point median')
        ax=fig.add_subplot(222);ax.plot(c[:,0],c[:,1],lw=1);sc=ax.scatter(c[:,0],c[:,1],c=np.arange(len(c)),cmap='viridis',s=16);fig.colorbar(sc,ax=ax,label='frame');ax.set_aspect('equal');ax.set_title('Top projection (same PCA basis)')
        ax=fig.add_subplot(223);ax.plot(np.arange(1,len(c)),deg);ax.set_title(f'Adjacent rotation max={deg.max():.2f} deg');ax.set_xlabel('destination frame');ax.grid(alpha=.3)
        ax=fig.add_subplot(224);ax.plot(np.arange(1,len(c)),np.linalg.norm(np.diff(centers,axis=0),axis=1));ax.set_title('Adjacent translation (world units)');ax.grid(alpha=.3)
        self.stats['max_adjacent_rotation_deg']=float(deg.max());self.save_plot(self.key,fig_image(fig),'相机矩阵先完整求逆，保留场景尺度；旋转角用极分解去除尺度。没有把最后一帧到第一帧假定为跟踪边。')

    def tracking(self):
        self.start('06_tracking');rows=[]
        areas=[];dareas=[]
        for f in self.frames:
            entry=self.li['frames'][f];size=np.prod(entry['shape'][:2]);areas.append(entry['mask_foreground_before']/size);dareas.append(entry['mask_foreground_after']/size)
        self.stats['mask_fraction_range']=[float(min(areas)),float(max(areas))]
        for f in self.fsel:
            before,after,m,dil,out=self.virtual(f);extra=dil & ~m
            track=self.tracking_rgb(f)
            rows.append([image(track) if track is not None else missing('跟踪归档缺失'),image(m),overlay(track,m) if track is not None else missing(),image(dil),overlay(after,extra,(255,180,0)),crop(overlay(before,m),dil)])
        self.save_grid(self.key,rows,['实际跟踪输入（归档）','最终跟踪 mask','实际输入叠加 mask','实际 LaMa mask','膨胀增加区域（金色）','删除前目标参考（放大）'],
                       f'MASK_DILATION={self.li["parameters"].get("dilation")}；直接读取 images.zip 的实际输入。逐帧 mask 原值非零为前景。未保存交互修改前的完整 mask 历史。',self.fsel)
        self.problems.append({'module':self.key,'note':'只有最终跟踪 mask；没有完整的修改前 mask 历史'})
        fig,ax=plt.subplots(figsize=(15,4));ax.plot(areas,label='tracked mask');ax.plot(dareas,label='LaMa mask');ax.set_xlabel('frame');ax.set_ylabel('image fraction');ax.legend();ax.grid(alpha=.3)
        self.start('06_tracking_area');self.record(pathof(self.lama['input_manifest']))
        self.save_plot('06_tracking_area',fig_image(fig),'面积跳变是检查线索，不直接等价于跟踪错误；物体遮挡和相机俯仰也会改变面积。')
        for page,start in enumerate(range(0,len(self.frames),30),1):
            key=f'tracking_all_{page:02d}';self.start(key);tiles=[]
            for f in self.frames[start:start+30]:
                _,_,m,dil,_=self.virtual(f);rgb=self.tracking_rgb(f)
                tiles.append((overlay(overlay(rgb,dil,(255,180,0),.18),m,(255,60,60),.28) if rgb is not None else missing('跟踪归档缺失'),f'{f} | mask={areas[self.frames.index(f)]:.2%}'))
            rows=[tiles[i:i+5] for i in range(0,len(tiles),5)]
            while len(rows[-1])<5:rows[-1].append(missing(''))
            self.save_grid(key,rows,['跟踪全序列']*5,'底图为 images.zip 实际输入；红：最终跟踪 mask；金：LaMa 膨胀边界；完整序列按序显示，没有择优。',tilew=580,tileh=380)

    def lama_rgb(self):
        self.start('07_lama_rgb');rows=[]
        for f in self.fsel:
            orig,removed,m,dil,out=self.virtual(f)
            removed=self.load(pathof(self.li['frames'][f]['outputs']['color']))
            masked=removed.copy();masked[dil]=0
            err=np.abs(out.astype(float)-removed.astype(float)).mean(-1)/255
            outside=float(err[~dil].max()) if (~dil).any() else 0
            rows.append([image(removed),image(masked),overlay(removed,dil),image(out),crop(out,dil),(heat(err,0,.3),f'max 外部变化={outside:.6f}')])
        self.save_grid(self.key,rows,['实际输入 RGB','RGB×(1-mask)示意','实际输入 mask','LaMa RGB 输出','输出洞区放大','输出−输入 |差| 0→0.3'],
                       '第二列仅展示遮挡范围，不代表 LaMa 输入文件本身已涂黑。RGB 输出属于伪目标，不能当作 GT。RECURSIVE_GUIDE='+str(self.lama['parameters'].get('recursive_guide')),self.fsel)

    def lama_geometry(self):
        for modality,key in [('depth','08_lama_depth'),('normal','09_lama_normal')]:
            self.start(key);rows=[]
            for f in self.fsel:
                entry=self.li['frames'][f];outs=self.lama['frames'][f]['outputs'];m=self.load(pathof(entry['outputs']['color_mask']))>0
                a=self.load(pathof(entry['outputs'][modality]));b=self.load(pathof(outs[modality]))
                if modality=='depth':
                    ref=self.load(pathof(entry['inputs']['reference_depth']));valid=[np.isfinite(x)&(x>0) for x in [ref,a,b]]
                    allz=np.concatenate([x[v][::7] for x,v in zip([ref,a,b],valid) if v.any()]);lo,hi=np.percentile(allz,[2,98]);span=max(float(hi-lo)*.15,1e-5)
                    rows.append([(heat(ref,lo,hi,'turbo',valid[0]),f'共用色域 [{lo:.3g}, {hi:.3g}]'),heat(a,lo,hi,'turbo',valid[1]),image(m),heat(b,lo,hi,'turbo',valid[2]),crop(heat(b,lo,hi,'turbo',valid[2]),m),(heat(b-a,-span,span,'coolwarm',valid[1]&valid[2]),f'差值色域 ±{span:.3g}')])
                else:
                    va=self.load(pathof(entry['outputs']['normal_valid']))>0;vb=self.load(pathof(outs['normal_valid']))>0
                    infer=self.load(pathof(entry['outputs']['normal_inference_mask']))>0 if 'normal_inference_mask' in entry['outputs'] else m
                    dot=(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
                    ang=np.degrees(np.arccos(np.clip(dot,-1,1)))
                    rows.append([normal_vis(a,va),image(va),image(infer),normal_vis(b,vb),crop(normal_vis(b,vb),m),heat(ang,0,90,'magma',va&vb)])
            cols=['删除前参考深度','删除后输入深度','实际补全 mask','补全深度输出','补全深度洞区','输出−输入深度'] if modality=='depth' else ['输入相机空间法线','输入有效位置','实际法线推理 mask','补全相机空间法线','法线洞区放大','夹角差 0→90°']
            note='深度按每行前后共同的 2–98% 色域显示，黑色为无效值；深度单位遵循渲染场景坐标，不能直接视为米。' if modality=='depth' else 'RGB = (Nx,Ny,Nz)/2+0.5，相机空间朝向相机；黑色无效。法线推理 mask 包括洞与无效像素，因此可能大于 RGB mask。'
            self.save_grid(key,rows,cols,note,self.fsel)

    def matching(self):
        self.start('10_correspondence');self.record(pathof(self.init['outputs']['support']));self.record(pathof(self.init['outputs']['diagnostics']))
        s=self.support;rows=[]
        # Final depth-anchor points do not retain raw matcher correspondences.
        # For that branch, display saved three-view anchor observations instead.
        if 'anchor_frames' in self.evidence:
            self.record(pathof(self.init['outputs']['surface_evidence']))
            fr=self.evidence['anchor_frames'];uv=self.evidence['anchor_uv'];pairs=np.stack([fr[:,0],fr[:,1]],-1);quality=self.evidence['anchor_reprojection']
            note='展示真实保存的三视角锚点观测中的前两视角；颜色线仅表示对应关系。这里不是全部原始 RoMa 候选；原始候选未完整落盘。'
        else:
            valid=(s['source_kind']==0)&(s['other_frame']>=0);pairs=np.stack([s['frame'][valid],s['other_frame'][valid]],-1);uv=np.stack([s['pixel_uv'][valid],s['other_uv'][valid]],1);quality=-s['confidence'][valid]
            note='展示最终保留的 RGB 三角化点对应；不代表被拒绝的全部匹配。置信度数值不等价于真实表面准确率。'
        unique,counts=np.unique(pairs,axis=0,return_counts=True)
        for pair in unique[np.argsort(counts)[-3:][::-1]]:
            mask=(pairs==pair).all(1);idx=np.flatnonzero(mask);idx=idx[np.argsort(quality[idx])];idx=np.array(sample(idx,70))
            f1,f2=[self.frames[int(i)] for i in pair]
            a=self.im(pathof(self.lama['frames'][f1]['outputs']['color']));b=self.im(pathof(self.lama['frames'][f2]['outputs']['color']))
            joined=Image.new('RGB',(a.width+b.width,max(a.height,b.height)));joined.paste(a);joined.paste(b,(a.width,0));d=ImageDraw.Draw(joined)
            for k,ii in enumerate(idx):
                p,q=uv[ii,:2];color=tuple(int(x*255) for x in plt.get_cmap('turbo')(k/max(len(idx)-1,1))[:3]);d.line((float(p[0]),float(p[1]),float(q[0])+a.width,float(q[1])),fill=color,width=1)
                for x,y in [(p[0],p[1]),(q[0]+a.width,q[1])]: d.ellipse((float(x)-2,float(y)-2,float(x)+2,float(y)+2),fill=color)
            rows.append([(joined,f'{f1} ↔ {f2} | 保存匹配={int(mask.sum())}，图示≤70')])
        if not rows:rows=[[missing('没有保留可显示的匹配对')]]
        self.save_grid(self.key,rows,['输入补全 RGB 与保存的匹配证据'],note,tilew=2600,tileh=850)
        pairsdiag=self.diag.get('pairs',[])
        sums={k:sum(float(p.get(k,0)) for p in pairsdiag) for k in ['matched','triangulated','accepted']}
        self.stats['initialization_counts']={k:self.diag.get(k) for k in ['rgb_candidates','rgb_points','depth_points','initialized_points']};self.stats['matching_pair_totals']=sums

    def initialization(self):
        self.start('11_initialization');self.record(pathof(self.init['outputs']['ply']));self.record(pathof(self.init['outputs']['editable_mask']));self.record(pathof(self.init['outputs']['support']))
        old=self.old_sample;new=self.new_xyz;s=self.support;kind=s.get('source_kind');groups=[(old,'#93a5b6','old background')]
        names={0:('RGB triangulated','#e97629'),1:('depth backprojected','#1889c4'),2:('depth-anchor output','#d33f72')}
        if len(kind)==len(new):
            groups += [(new[kind==k],names.get(int(k),('unknown','#555'))[1],names.get(int(k),('unknown','#555'))[0]) for k in np.unique(kind)]
        else:groups.append((new,'#d33f72','new points'))
        a=self.cloud(groups,'初始化：高斯中心（不是高斯椭球）')
        if 'anchor_xyz' in self.evidence:
            self.record(pathof(self.init['outputs']['surface_evidence']))
            b=self.cloud([(old,'#b5bec6','old background'),(self.evidence['anchor_xyz'],'#24a076','saved 3-view anchors'),(new,'#d33f72','corrected output')],'深度锚点校正：锚点与输出（不是拟合平面）')
        else:
            b=self.cloud(groups,'初始化宽视窗：用于检查远离修复区的点',limit=self.radius*1.5)
        fig,axs=plt.subplots(1,3,figsize=(17,4))
        nidx=np.flatnonzero(self.edit);vv=self.vertices[nidx];opacity=1/(1+np.exp(-np.clip(vv['opacity'],-60,60)));scales=np.exp(np.stack([vv[f'scale_{i}'] for i in range(3)],-1))
        axs[0].hist(opacity,bins=45);axs[0].set_title('Initial new opacity')
        axs[1].hist(np.log10(np.maximum(scales.max(1),1e-12)),bins=45);axs[1].set_title('log10(max new scale)')
        if 'corrected_depth' in s:
            axs[2].hist(s['corrected_depth']-s['raw_depth'],bins=60);axs[2].set_title('Depth correction (world camera Z)')
        else:axs[2].hist(s['confidence'],bins=50);axs[2].set_title('Saved matching confidence')
        self.save_grid(self.key,[[a],[b],[fig_image(fig)]],['同一局部坐标和比例，PCA 仅用于观察，不参与初始化'],
                       f'旧点 {int((~self.edit).sum()):,}；新点 {int(self.edit.sum()):,}；surface_init={self.sel["configuration"].get("surface_init")}。中心图抽样；可见点云不是最终渲染。',tilew=2800,tileh=800)

    def local_background(self):
        self.start('12_local_background')
        if self.local:
            self.record(pathof(self.j['inputs']['local_selection']));train=self.load(pathof(self.local['outputs']['trainable_mask'])).astype(bool);votes=self.load(pathof(self.local['outputs']['overlap_votes']))
            assert len(train)==len(self.edit)
            inds=np.arange(0,len(self.edit),max(1,len(self.edit)//180000));x=xyz(self.vertices,inds)
            a=self.cloud([(x[~train[inds]],'#9da9b2','frozen old'),(x[train[inds]&~self.edit[inds]],'#ed9a25','trainable old'),(self.new_xyz,'#cf3b75','trainable new')],'实际 trainable_mask：冻结旧点 / 局部旧点 / 新点')
            fig,axs=plt.subplots(1,2,figsize=(14,4));axs[0].hist(votes[~self.edit],bins=60);axs[0].set_title('Old point overlap votes')
            cnt=[int((~train&~self.edit).sum()),int((train&~self.edit).sum()),int(self.edit.sum())];axs[1].bar(['frozen old','trainable old','new'],cnt,color=['#9da9b2','#ed9a25','#cf3b75']);axs[1].set_title('Exact initial point counts')
            self.stats['trainable_counts']=dict(zip(['frozen_old','trainable_old','new'],cnt));b=fig_image(fig)
        else:
            a=missing('历史实验未保存 local_selection\n不能按当前代码推断冻结策略',(1400,500));b=text_panel([f'freeze_background={self.cfg.get("freeze_background", "未声明")}',f'local_background={self.cfg.get("local_background", "未声明")}',f'该实验总点数={len(self.vertices):,}'])
            self.problems.append({'module':self.key,'note':'历史实验缺少局部选择 manifest'})
        self.save_grid(self.key,[[a],[b]],['输入初始化 + mask → 实际可训练集合'],
                       '颜色表示训练资格，不代表参数一定发生更新。使用保存的 trainable_mask，不按空间距离重新猜测。',tilew=2800,tileh=800)

    def training(self):
        debug=self.run/'edgs_joint/debug';steps=sorted(int(p.stem.split('_')[-1]) for p in debug.glob('step_*.npz') if int(p.stem.split('_')[-1])<=self.step)
        use=sample(steps,6)
        if not use:
            for key in ['13_training_rgb','14_training_geometry']:
                self.save_grid(key,[[missing('未保存训练调试数组')]],['无可用历史预览'])
            return
        idx=self.cfg.get('debug',{}).get('view_index',0);f=self.frames[idx];entry=self.li['frames'][f];out=self.lama['frames'][f]['outputs']
        self.start('13_training_rgb');target=self.load(pathof(out['color']));mask=self.load(pathof(entry['outputs']['color_mask']))>0;rows=[]
        arrs=[]
        for step in use:
            p=debug/f'step_{step:06d}.npz';self.record(p);a=dict(np.load(p));arrs.append(a);render=image(a['rgb']);err=np.abs(np.asarray(render).astype(float)-target.astype(float)).mean(-1)/255
            rows.append([image(target),render,heat(err,0,.25),crop(target,mask),crop(render,mask),(crop(heat(err,0,.25),mask),f'洞内 MAE={err[mask].mean():.5f}')])
        self.save_grid(self.key,rows,['LaMa 伪目标','训练时实际渲染','RGB |差| 0→0.25','LaMa 洞区放大','渲染洞区放大','洞区误差放大'],
                       f'固定 debug view_index={idx} / frame={f}；只使用 ≤{self.step} 步的保存记录。这是对伪目标的拟合，不是测试 GT 精度。',[f'step\n{x:,}' for x in use])
        self.start('14_training_geometry');depth=self.load(pathof(out['depth']));normal=self.load(pathof(out['normal']));valid=self.load(pathof(out['normal_valid']))>0
        vals=np.concatenate([a['depth'][np.isfinite(a['depth'])&(a['depth']>0)][::20] for a in arrs]+[depth[depth>0][::20]]);lo,hi=np.percentile(vals,[2,98]);rows=[]
        for step,a in zip(use,arrs):
            va=np.isfinite(a['depth'])&(a['depth']>0);rows.append([heat(depth,lo,hi,'turbo',depth>0),heat(a['depth'],lo,hi,'turbo',va),normal_vis(normal,valid),normal_vis(a['normal'],a['alpha']>.01),heat(a['alpha'],0,1,'gray'),crop(heat(a['depth'],lo,hi,'turbo',va),mask)])
        self.save_grid(self.key,rows,['LaMa 深度目标','渲染深度','LaMa 法线目标','渲染法线','渲染 alpha 0→1','渲染深度洞区'],
                       f'同一固定相机；所有深度共用色域 [{lo:.5g}, {hi:.5g}]，黑色无效。图可检查鼓包、孔洞和法线突变，但 LaMa 目标不是几何真值。',[f'step\n{x:,}' for x in use])

    def losses(self):
        self.start('15_training_losses');ds=[];xs=[]
        for p in sorted((self.run/'edgs_joint/diagnostics').glob('step_*.json')):
            x=int(p.stem.split('_')[-1])
            if x<=self.step:ds.append(read(p));xs.append(x);self.record(p)
        fig,axs=plt.subplots(2,3,figsize=(18,9))
        groups=[(['total','rgb_hole','rgb_known_preserve'],'RGB / total (raw logged terms)'),(['ssim','lpips','alpha_preserve'],'Appearance / alpha'),(['depth','lama_normal','depth_normal_consistency','multiview'],'Geometry (raw logged terms)'),(['old_gaussians','new_gaussians'],'Retained point counts'),(['pruned_old_total','pruned_new_total'],'Cumulative pruned'),(['weight_depth','weight_lama_normal','weight_multiview','weight_depth_normal_consistency'],'Effective geometry weights')]
        for ax,(keys,title) in zip(axs.flat,groups):
            for k in keys:
                yy=[d.get(k,np.nan) for d in ds]
                if np.isfinite(yy).any():ax.plot(xs,yy,label=k)
            ax.set_title(title);ax.set_xlabel('training step');ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'Not saved in this run',ha='center',transform=ax.transAxes)
        self.save_plot(self.key,fig_image(fig),'记录点来自训练诊断；训练抽样视角会变化，因此不是固定验证集曲线。图中空白表示历史代码没记录该项，不能解释为损失=0。')

    def final_geometry(self):
        self.start('16_final_geometry');p=(self.b/'point_cloud.ply').resolve();self.record(p);v=ply_vertices(p)
        rp=self.run/f'edgs_joint/snapshots/iteration_{self.step}/retained_rows.npy'
        if not rp.exists() and self.step==self.cfg['iterations'] and 'retained_rows' in self.j['outputs']:rp=pathof(self.j['outputs']['retained_rows'])
        if rp.exists():
            rows=self.load(rp);valid=rows>=0;new=np.ones(len(v),bool);new[valid]=self.edit[rows[valid]];assert len(rows)==len(v)
            newxyz=xyz(v,np.flatnonzero(new));oldidx=np.flatnonzero(~new)[::max(1,int((~new).sum())//130000)];oldxyz=xyz(v,oldidx)
            a=self.cloud([(self.old_sample,'#c2c8ce','initial old'),(self.new_xyz,'#24a3be','initial new'),(newxyz,'#d54371','final new')],'初始新点 → 最终新点：检查漂浮与迁移')
            b=self.cloud([(oldxyz,'#9aa6b0','final old'),(newxyz,'#d54371','final new')],'最终发布模型：旧点与新点')
            self.stats['final_new_points']=int(new.sum())
        else:
            idx=np.arange(0,len(v),max(1,len(v)//180000));a=self.cloud([(xyz(v,idx),'#688ba6','final centers')],'最终模型中心（缺少精确行映射，不推断新旧）');b=text_panel(['历史结果没有 retained_rows。','无法可靠区分最终点来自旧背景还是新增点。'])
            self.problems.append({'module':self.key,'note':'最终模型没有可用 retained_rows 行映射'})
        meshes=sorted((self.b/'model').glob(f'mesh/ours_{self.step}/tsdf_fusion_post.ply'))
        if meshes:
            mp=meshes[0];self.record(mp);mv=ply_vertices(mp);mi=np.arange(0,len(mv),max(1,len(mv)//180000));c=self.cloud([(xyz(mv,mi),'#389f8b','mesh vertices')],'所选快照实际输出 Mesh 顶点抽样（不是面渲染）');self.stats['selected_mesh_vertices']=len(mv)
        else:
            c=missing(f'所选 {self.step} 步快照没有 Mesh 产物\n不借用其它实验或其它步数的 Mesh',(1800,600));self.problems.append({'module':self.key,'note':f'所选 {self.step} 步模型没有 mesh/ours_{self.step}/tsdf_fusion_post.ply'})
        self.stats['final_points']=len(v)
        panels=[[a],[b],[c]]
        sem=self.run/'inpainted_mesh/semantic_manifest.json'
        if sem.is_file():
            sm=read(sem)
            if pathof(sm['inputs']['gaussian_ply']).resolve()==p:
                self.record(sem);mp=sem.parent/'semantic_mesh.ply';self.record(mp);mv=ply_vertices(mp);mi=np.arange(0,len(mv),max(1,len(mv)//180000));pts=xyz(mv,mi)
                labels=np.load(sem.parent/'vertex_instance_id.npy',mmap_mode='r')[mi];self.record(sem.parent/'vertex_instance_id.npy')
                panels.append([self.cloud([(pts[~np.isin(labels,self.ids)],'#869eae','other vertex labels'),(pts[np.isin(labels,self.ids)],'#df4d50','removed target labels')],'Inpaint 8：最终 Mesh 语义标签（红色检查目标标签残留）')])
        self.save_grid(self.key,panels,['最终发布 PLY / 几何产物'],
                       f'精确选中快照={self.step}；最终高斯数={len(v):,}。PCA 三视图使用初始化的同一坐标与比例。只画中心/顶点，不能据此判断椭球尺度。',tilew=2800,tileh=800)

    def test_rows(self,names):
        rows=[];labels=[]
        for f in names:
            pred=self.load(self.b/'renders'/f'{f}.png');gt=self.load(self.b/'gt'/f'{f}.png');m=self.load(self.b/'masks'/f'{f}.png')
            if m.ndim==3:m=m[...,0]
            m=m>127;err=np.abs(pred.astype(float)-gt.astype(float)).mean(-1)/255;v=self.result['per_view'][f]
            rows.append([image(gt),image(pred),overlay(gt,m),crop(gt,m),crop(pred,m),(heat(err,0,.25),f'PSNR={v["PSNR"]:.2f} / mPSNR={v["masked_PSNR"]:.2f}')]);labels.append(f.replace('test_','test_\n'))
        return rows,labels

    def evaluation(self):
        self.start('17_test_evaluation');pv=self.result['per_view'];names=sorted(pv);ordered=sorted(names,key=lambda f:pv[f]['masked_PSNR']);chosen=sample(ordered,6);rows,labels=self.test_rows(chosen)
        cols=['测试 GT','所选模型渲染','评估 mask 叠加','GT 洞区放大','模型洞区放大','RGB |差| 0→0.25']
        self.save_grid(self.key,rows,cols,'按 masked_PSNR 从差到好取分位点，包含最差视角；不是只挑好看的图。masked_PSNR 使用洞外归零后的全图平均，不是纯洞内 PSNR。',labels)
        for page,start in enumerate(range(0,len(names),6),1):
            key=f'test_all_{page:02d}';self.start(key);rows,labels=self.test_rows(names[start:start+6]);self.save_grid(key,rows,cols,'所有测试视角按文件名排序；GT、输出、评估 mask 使用同名文件一一对齐。',labels)
        self.start('17_test_metrics');self.record(self.b/'results.json');fig,axs=plt.subplots(2,2,figsize=(16,8))
        for ax,keys in zip(axs.flat,[['PSNR','masked_PSNR'],['SSIM','masked_SSIM'],['LPIPS','masked_LPIPS'],['mask_fraction']]):
            for k in keys:ax.plot([pv[f].get(k,np.nan) for f in names],label=k)
            ax.set_xlabel('test frame (filename order)');ax.legend();ax.grid(alpha=.2)
        self.save_plot('17_test_metrics',fig_image(fig),'完整测试集逐帧指标。FID 是集合指标，不画不存在的单帧 FID。')

    def finish(self):
        self.problems=list({json.dumps(p,sort_keys=True):p for p in self.problems}.values())
        main=[f for f in self.figs if f['key'] in TITLES]
        overview=Image.new('RGB',(2400,140+math.ceil(len(main)/3)*520),'#f4f7fb');d=ImageDraw.Draw(overview);d.text((24,20),f'{self.scene}｜PaintMesh 全模块输入输出索引',font=font(43),fill='#12334b');d.text((24,83),'点开对应单模块大图查看；这是历史最佳运行的实际链路，不是统一配置实验。',font=font(26),fill='#556270')
        for n,f in enumerate(main):
            im=Image.open(self.out/f['file']);tile=ImageOps.contain(im,(780,440));x=(n%3)*800+10;y=140+(n//3)*520;overview.paste(tile,(x,y));d.text((x,y+444),f['key']+' '+f['title'].split('｜')[-1][:19],font=font(21),fill='#15354d')
        overview.save(self.out/'00_overview.jpg',quality=94,subsampling=0)
        write(self.out/'sources.json',{k:sorted(v) for k,v in self.sources.items()})
        write(self.out/'audit.json',{'stats':self.stats,'figures':self.figs,'missing_or_limits':self.problems,'selected_run':str(self.run),'config':self.cfg})
        cards=[]
        for f in self.figs:
            cards.append(f'<article><h3>{html.escape(f["title"])}</h3><a href="{f["file"]}" target="_blank"><img loading="lazy" src="{f["file"]}"></a><p>{f["width"]} × {f["height"]} · 点击查看原图</p></article>')
        title=f'{self.scene} · PaintMesh 模块调试图'
        document=page(title,f'<p><a href="../index.html">全部场景</a> · <a href="00_overview.jpg">模块总览</a> · <a href="sources.json">实际来源</a> · <a href="audit.json">配置与检查记录</a></p><p>所选快照 {self.step} 步；运行 {html.escape(str(self.run))}</p><p>所有跟踪帧 {len(self.frames)}，所有测试帧 {self.result["frame_count"]}，都在下方展开。GT 只用于测试，LaMa 是训练伪目标。</p>'+''.join(cards))
        (self.out/'index.html').write_text(document)
        (self.out/'README.md').write_text(f'# {title}\n\n[图片索引](index.html) · [模块总览](00_overview.jpg) · [输入输出来源](sources.json) · [配置和检查记录](audit.json)\n\n'+ '\n'.join(f'- [{f["title"]}]({f["file"]})' for f in self.figs)+'\n\n局限/缺失：\n'+ '\n'.join(f'- {p["module"]}: {p.get("note",p.get("missing"))}' for p in self.problems))
        return self.stats

    def run_all(self):
        for method in ['reconstruction','segmentation','semantic_mesh','removal','cameras','tracking','lama_rgb','lama_geometry','matching','initialization','local_background','training','losses','final_geometry','evaluation']:
            print(f'{self.scene}: {method}',flush=True)
            getattr(self,method)();gc.collect()
        return self.finish()

def page(title,body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui,sans-serif;margin:28px;background:#eef2f6;color:#183247}a{color:#0c6999}article{background:white;padding:16px;margin:22px 0;border-radius:12px}img{max-width:100%;height:auto}p{overflow-wrap:anywhere}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ccd5de}</style><h1>'+html.escape(title)+'</h1>'+body+'</html>'

def make_index(out):
    rows=[];stats=[];figs=0;limits=0
    for s in SCENES:
        p=out/s/'audit.json'
        if not p.exists():continue
        d=read(p);stats.append(d['stats']);figs+=len(d['figures'])+1;limits+=len(d['missing_or_limits'])
        rows.append(f'<tr><td><a href="{s}/index.html">{s}</a></td><td>{d["stats"]["selected_step"]}</td><td>{d["stats"]["virtual_frames"]}</td><td>{d["stats"]["test_frames"]}</td><td><a href="{s}/00_overview.jpg">总览图</a></td><td>{len(d["figures"])+1}</td></tr>')
    body='<p>本报告只读取已保存的产物，未重训、未修改模型、未生成补全图片。各场景来自 best_results 的实际选中实验，包含不同初始化和训练步数。</p><p>每场景按模块列出输入→输出、全跟踪序列、全部测试视角。黄色/红色 mask 是诊断叠加；误差和点云是科学可视化，不是新模型预测。</p><p>点开场景后再点击大图，可按原尺寸查看。复制本 module_debug 文件夹即可离线看图；sources.json 指向服务器原始文件。</p><table><tr><th>场景</th><th>选中步数</th><th>完整虚拟帧</th><th>完整测试帧</th><th>模块总览</th><th>图片数</th></tr>'+''.join(rows)+'</table><p>初始化和 Mesh 图为顶点/高斯中心抽样的正交投影，不是椭球或表面渲染；PCA 只用于显示。没有完整原始匹配候选、没有跟踪修订历史、没有对应快照 Mesh 的情况在模块图和 audit.json 中明确标注。</p>'
    body+='<h2>按模块横向查看所有场景</h2><ul>'+''.join(f'<li><a href="modules/{k}.html">{html.escape(v)}</a></li>' for k,v in TITLES.items())+'</ul>'
    modules=out/'modules';modules.mkdir(exist_ok=True)
    for key,title in TITLES.items():
        cards=[]
        for s in SCENES:
            p=out/s/f'{key}.jpg'
            if p.exists():cards.append(f'<article><h2>{s}</h2><a href="../{s}/{key}.jpg" target="_blank"><img loading="lazy" src="../{s}/{key}.jpg"></a><p><a href="../{s}/index.html">该场景全部模块</a> · <a href="../{s}/sources.json">输入输出来源</a></p></article>')
        (modules/f'{key}.html').write_text(page(title,'<p><a href="../index.html">返回总索引</a> · 全场景同模块对照；点击图片查看原尺寸。</p>'+''.join(cards)))
    (out/'index.html').write_text(page('PaintMesh · 11 场景全模块输入输出调试图',body))
    write(out/'summary.json',{'scene_count':len(stats),'figure_count':figs,'scenes':stats,'limits_count':limits})
    (out/'README.md').write_text('# PaintMesh 全模块输入输出调试图\n\n打开 [index.html](index.html) 查看全部场景。图片可点击放大；也可直接看每个场景目录下的 00_overview.jpg。\n\n本报告使用 best_results 实际选中的历史实验，读取其 camera / LaMa / 初始化 / 局部选择 manifest；不混用其他实验的结果。完整覆盖保存的跟踪帧和测试帧。每张图的原始输入文件见各场景 sources.json。\n\n'+ '\n'.join(f'- [{s}](./{s}/index.html)' for s in SCENES if (out/s/'audit.json').exists())+f'\n\n当前完成 {len(stats)} 场景，共 {figs} 张图。\n\n生成脚本：scripts/paintmesh/visualize_pipeline_io.py。CPU 只读诊断，不重训、不调用图像生成模型。缺失中间产物明确标注，原始匹配候选和修改前的 mask 历史无法由最终结果反推。\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenes',nargs='+',default=SCENES);ap.add_argument('--output',type=Path,default=BEST/'module_debug');ap.add_argument('--modules',nargs='+',choices=['reconstruction','segmentation','semantic_mesh','removal','cameras','tracking','lama_rgb','lama_geometry','matching','initialization','local_background','training','losses','final_geometry','evaluation']);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    failures=[]
    for scene in args.scenes:
        try:
            a=Audit(scene,args.output)
            if args.modules:
                previous=read(a.out/'audit.json');a.figs=previous['figures'];a.stats=previous['stats'];a.problems=previous['missing_or_limits'];chain=a.sources['manifest_chain'];a.sources={k:set(v) for k,v in read(a.out/'sources.json').items()};a.sources['manifest_chain']=chain
                if 'segmentation' in args.modules:a.real=[p.name for p in sample(sorted((a.base/'edgs/train/ours_30000/renders').glob('*.png')))]
                for method in args.modules:print(f'{scene}: {method}',flush=True);getattr(a,method)()
                a.finish()
            else:a.run_all()
            del a;gc.collect();make_index(args.output);print(f'DONE {scene}',flush=True)
        except Exception:
            failures.append(scene);traceback.print_exc()
    make_index(args.output)
    if failures:raise SystemExit('Failed scenes: '+','.join(failures))

if __name__=='__main__':main()
