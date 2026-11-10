"""Compact shared VM-SH color field, inspired by MILo's fixed-mesh evaluation.

This is a documented adaptation, not a reproduction of MILo's 3000-grid/full-image
training. Geometry is never optimized and test RGB never enters this module.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def sh2(d):
    x,y,z=d.unbind(-1)
    return torch.stack([torch.full_like(x,.28209479177387814),-.4886025119*y,.4886025119*z,-.4886025119*x,
                        1.0925484306*x*y,-1.0925484306*y*z,.3153915653*(3*z*z-1),-1.0925484306*x*z,.5462742153*(x*x-y*y)],-1)

class ColorField(nn.Module):
    def __init__(self,center,extent,resolution=256,rank=16):
        super().__init__()
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32))
        self.register_buffer('extent',torch.tensor(float(extent)))
        self.planes=nn.Parameter(.2*torch.randn(3,rank,resolution,resolution))
        self.lines=nn.Parameter(.2*torch.randn(3,rank,resolution,1))
        self.basis=nn.Linear(3*rank,27,bias=False)

    def forward(self,p,d):
        p=(p-self.center)/self.extent
        r=p.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        p=torch.where(r<=1,p,(2-1/r)*p/r)*.5
        uv=torch.stack([p[:,[0,1]],p[:,[0,2]],p[:,[1,2]]]).unsqueeze(2)
        line=torch.stack([p[:,2],p[:,1],p[:,0]])
        line=torch.stack([torch.zeros_like(line),line],-1).unsqueeze(2)
        a=F.grid_sample(self.planes,uv,align_corners=True).flatten(0,1).squeeze(-1)
        b=F.grid_sample(self.lines,line,align_corners=True).flatten(0,1).squeeze(-1)
        rgb=(self.basis((a*b).T).reshape(-1,3,9)*sh2(d)[:,None]).sum(-1)+.5
        return torch.relu(rgb)

    def tv(self):
        return ((self.planes[:,:,1:]-self.planes[:,:,:-1]).square().mean()+
                (self.planes[:,:,:,1:]-self.planes[:,:,:,:-1]).square().mean())*.02

def fit(pool,output,config,device):
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=ColorField(pool['center'],pool['extent'],config['grid_resolution'],config['rank']).to(device)
    optim=torch.optim.Adam([{'params':[model.planes,model.lines],'lr':.02},
                            {'params':model.basis.parameters(),'lr':.001}],betas=(.9,.99))
    n=len(pool['positions']);steps=config['appearance_steps'];decay=.1**(1/steps)
    positions=pool['positions'];directions=pool['directions'];rgb=pool['rgb'];valid=pool['valid']
    history=[]
    for step in range(steps):
        ids=rng.integers(n,size=config['batch_rays']);ok=valid[ids]
        optim.zero_grad(set_to_none=True)
        if ok.any():
            good=ids[ok]
            pp=torch.from_numpy(positions[good]).to(device);dd=torch.from_numpy(directions[good]).to(device)
            target=torch.from_numpy(rgb[good].astype('float32')/255).to(device)
            loss=((model(pp,dd)-target)**2).sum()/(len(ids)*3)
        else:loss=model.planes.sum()*0
        tv=model.tv()*(decay**step);total=loss+tv;total.backward();optim.step()
        for group in optim.param_groups:group['lr']*=decay
        if step%250==0 or step==steps-1:
            row=dict(step=step+1,mse=float(loss.detach()),tv=float(tv.detach()))
            history.append(row);print('COLOR_FIELD',row,flush=True)
    torch.save({'state_dict':model.state_dict(),'config':config,'history':history},output)
    return model.eval(),history

def render(model,camera,depth,device):
    from .core import rays
    rr=rays(camera);z=depth.reshape(-1);good=z>0
    output=np.zeros((len(z),3),dtype='float32');ids=np.flatnonzero(good)
    with torch.no_grad():
        for start in range(0,len(ids),16384):
            ii=ids[start:start+16384];p=rr[ii,:3]+rr[ii,3:]*z[ii,None]
            direction=rr[ii,3:];direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            output[ii]=model(torch.from_numpy(p).to(device),torch.from_numpy(direction).to(device)).clamp(0,1).cpu().numpy()
    return np.rint(output.reshape(camera['height'],camera['width'],3)*255).astype('uint8')
