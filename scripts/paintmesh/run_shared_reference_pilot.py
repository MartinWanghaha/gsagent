#!/usr/bin/env python3
"""Truck/cube fixed-reference pilot; separate GPU workers, matched existing controls."""
from __future__ import annotations
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from run_geometry_ablation import REPO, METRICS, read, write, atomic, now, sha
from run_muti_plan import PYTHON, Runner, validate_evaluation
from edgs_inpaint_io import implementation, record, verify_record

OLD = REPO / "output/paintmesh/Inpaint360GS/muti_plan"
LAMA = "/home/member/.conda/envs/lama/bin/python"
GUIDE = Path(__file__).with_name("shared_reference_completion.py")
GPU_MAP = {"cube": 0, "truck": 1}


def codes():
    result = implementation()
    for p in (Path(__file__), GUIDE, Path(__file__).with_name("run_muti_plan.py"),
              Path(__file__).with_name("run_inpaint360_full.py"), Path(__file__).with_name("eval_inpaint360.py"),
              REPO / "submodules/Inpaint360GS/LaMa/bin/predict_color.py",
              REPO / "submodules/Inpaint360GS/LaMa/saicinpainting/evaluation/refinement.py",
              REPO / "submodules/Inpaint360GS/LaMa/configs/prediction/default.yaml",
              REPO / "submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py"):
        result[str(p.relative_to(REPO))] = sha(p)
    return result


def prepare(root):
    if (root / "request.json").exists():
        if read(root / "request.json")["code"] != codes():
            raise ValueError("Pilot implementation changed; do not mix results")
        return
    req = read(OLD / "request.json")
    gpu_lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuids = {int(x.split(",")[0]):x.split(",")[1].strip() for x in gpu_lines.strip().splitlines()}
    rows=[]
    for scene, gpu in GPU_MAP.items():
        row = copy.deepcopy(next(s for s in req["scenes"] if s["scene"] == scene))
        baseline = OLD / "scenes" / scene / "00_control"
        config = read(baseline / "config.json")
        if config != row["variants"][0]["config"]:
            raise ValueError("Control configuration changed")
        joint = read(baseline / "manifests/edgs_joint_manifest.json")
        if joint["request"]["implementation"] != implementation():
            raise ValueError("Training implementation no longer matches completed control")
        for step in (5000,10000):
            r = read(baseline / f"inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json")
            validate_evaluation(r, row["historical_evaluation"])
        dest = root / "scenes" / scene
        dest.mkdir(parents=True,exist_ok=True)
        write(dest / "source.json",row);shutil.copyfile(baseline / "config.json",dest / "config.json")
        if not (dest / "baseline").exists(): (dest / "baseline").symlink_to(baseline, target_is_directory=True)
        rows.append(dict(scene=scene,gpu=gpu,gpu_uuid=uuids[gpu],baseline=str(baseline),source=record(dest/"source.json"),
                         config=record(dest/"config.json"), frames=row["frame_count"]))
    request=dict(created_at=now(),root=str(root),scenes=rows,code=codes(),
        steps=[5000,10000],primary_step=5000,seed=0,resolution=4,
        gate=dict(each_scene_masked_psnr_gain_min=0.2,each_scene_psnr_drop_max=0.05,
                  each_scene_masked_lpips_relative_increase_max=0.05,
                  target_disagreement_reduction_min=0.05,mean_guide_coverage_min=0.1,
                  step10000_masked_psnr_drop_max=0.1),
        policy="Only training RGB targets change; fixed reference selected without test GT; same initial PLY and old-point eligibility; no automatic all-scene expansion before pilot review")
    write(root / "request.json",request)
    atomic(root / "TRAINING_PLAN.zh-CN.md", """# truck / cube 共享参考补全验证

目的：验证跨虚拟视角的 RGB 伪目标不一致是否限制重建效果。只改变 RGB 目标，不改损失、几何目标、初始化、旧点更新名单、剪枝、相机、mask 或随机种子。

- **cube → GPU 0，truck → GPU 1**，每场景训练 10000 步，保留真实 5000 步快照，两节点都评估全部真实 test 视角。
- 对照使用 `muti_plan/scenes/<scene>/00_control` 已完成的同代码、同配置、同 seed 训练，并核对代码哈希，不重复训练。
- truck 沿用 adaptive 180、dilation 10、depth 初始化和局部旧点更新；cube 沿用其匹配对照的 hemisphere 90、dilation 20、RGB 匹配初始化及旧点冻结。cube 的这些历史差异本轮不同时改变。复用初始化 PLY，不重新采样/截断点数。
- 从训练相机中均匀取最多 18 个参考候选，用洞区域几何覆盖率选一个；其原有 LaMa 补全图固定为所有视角共享的参考。
- 目标完成深度反投影到参考相机，要求深度相对误差 ≤2%、往返像素误差 ≤1.5、视线夹角 ≤60°，避开局部深度跳变 >5% 的区域，接受区域再腐蚀 1px。只将通过检查的参考 RGB 作为已知条件；LaMa 补全剩余洞。零可靠重叠视角沿用独立补全。`RECURSIVE_GUIDE=false`。
- 深度一致性仅是伪几何兼容性检查，不能证明深度真实；固定参考可能本身补错，也可能覆盖不足。本实验明确记录这些边界。
- 每张图 mask 外逐字节保持原输入；depth、normal、初始高斯全部字段、局部可训练名单保持原样。派生清单记录原始初始化/补全的来源，没有把复用初始化冒充重新初始化。
- `guide/validation.json` 报告训练视角间 RGB 不一致度（与选参考使用不同采样），`guide/target_comparison.png` 展示均匀视角及高覆盖视角；不按 test 指标选图。

## 预先固定的判断标准

5000 步为主要验证节点：两个场景各自 masked_PSNR 至少提升 0.2 dB，全图 PSNR 下降不超过 0.05 dB，masked_LPIPS 恶化不超过 5%。同时训练目标跨视角 RGB 不一致度至少下降 5%，平均参考覆盖至少 10%。10000 步用于稳定性复核，masked_PSNR 不应比对照下降超过 0.1 dB。全部 7 项指标和两节点均完整报告，不只选改善指标或最佳节点。

这是单 seed、两个已诊断场景的机制验证，达到门槛也不代表统计显著或保证所有场景改善；先看指标和图，再决定推广。失败/覆盖不足时不自动全场景重训。

## 文件

- `request.json`：配置、GPU、代码哈希、判据。
- `scenes/<scene>/source.json`、`config.json`：匹配对照输入和完整训练设置。
- `scenes/<scene>/guide/`：参考图、条件输入、剩余洞、可信传播区、覆盖与一致性验证、对比图。
- `scenes/<scene>/lama/output/color/`：新的训练 RGB 目标。
- `scenes/<scene>/manifests/`：派生目标、复用初始化、训练产物清单。
- `scenes/<scene>/edgs_joint/`：训练日志、检查点、点云和 5000 步快照。
- `scenes/<scene>/inpainted_3dgs_{5000,10000}/evaluation/inpaint360_r4/`：test 渲染和指标。
- `scenes/<scene>/status.json`、`logs/`：实时状态和逐阶段日志。
- `comparison.md/json`：全部指标及对应对照差值；`decision.json`：门槛是否满足。
""")
    atomic(root / "README.md", "# 共享参考补全：truck / cube\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [结果对比](comparison.md) · [请求配置](request.json)\n\n场景 GPU 分配：cube 0，truck 1。旧结果保留，派生结果均写入此目录。\n")
    report(root)


def report(root):
    with (root / "report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        req=read(root/"request.json");rows=[];checks={};pending=[]
        lines=["# 共享参考补全与独立 LaMa 对照", "", "固定对照、所有 test 视角、同节点比较。5000 为主要节点，10000 为复核。未完成项目不填数。", "",
               "| 场景 | 步数 | 方法 | " + " | ".join(METRICS) + " |", "|---|---:|---|"+"---:|"*len(METRICS)]
        for s in req["scenes"]:
            scene=s["scene"];dest=root/"scenes"/scene
            scene_rows={}
            for step in req["steps"]:
                base=read(Path(s["baseline"])/f"inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json")
                lines.append(f"| {scene} | {step} | 独立 LaMa | "+" | ".join(f"{base['metrics'][k]:.6f}" for k in METRICS)+" |")
                p=dest/f"inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json"
                if not p.exists():
                    pending.append(f"{scene}/{step}");continue
                result=read(p);validate_evaluation(result,base)
                delta={k:result["metrics"][k]-base["metrics"][k] for k in METRICS}
                row=dict(scene=scene,step=step,baseline=base["metrics"],shared=result["metrics"],delta=delta,result=record(p))
                rows.append(row);scene_rows[step]=row
                lines.append(f"| {scene} | {step} | 共享参考 | "+" | ".join(f"{result['metrics'][k]:.6f}" for k in METRICS)+" |")
                lines.append(f"| {scene} | {step} | 差值 | "+" | ".join(f"{delta[k]:+.6f}" for k in METRICS)+" |")
            if len(scene_rows)==2:
                q=read(dest/"guide/validation.json");primary=scene_rows[5000];d=primary["delta"];gate=req["gate"]
                before=q["consistency"]["independent_mae_255"];after=q["consistency"]["shared_mae_255"]
                checks[scene]=dict(masked_psnr=d["masked_PSNR"]>=gate["each_scene_masked_psnr_gain_min"],
                    full_psnr=d["PSNR"]>=-gate["each_scene_psnr_drop_max"],
                    masked_lpips=d["masked_LPIPS"]<=gate["each_scene_masked_lpips_relative_increase_max"]*primary["baseline"]["masked_LPIPS"],
                    target_consistency=before is not None and after<=before*(1-gate["target_disagreement_reduction_min"]),
                    reference_coverage=q["mean_hole_coverage"]>=gate["mean_guide_coverage_min"],
                    step10000_stability=scene_rows[10000]["delta"]["masked_PSNR"]>=-gate["step10000_masked_psnr_drop_max"])
        passed=not pending and len(checks)==2 and all(all(x.values()) for x in checks.values())
        decision=dict(updated_at=now(),complete=not pending,pending=pending,checks=checks,
            eligible_for_all_scene_review=passed,all_scene_started=False,
            conclusion="awaiting_pilot" if pending else ("pilot_passed_review_visuals_before_expansion" if passed else "do_not_expand_with_current_policy"))
        write(root/"comparison.json",dict(updated_at=now(),rows=rows,decision=decision));write(root/"decision.json",decision)
        lines += ["", "当前结论：`"+decision["conclusion"]+"`。", "", "[判据逐项结果](decision.json) · [训练方案](TRAINING_PLAN.zh-CN.md)", ""]
        atomic(root/"comparison.md","\n".join(lines))


class Worker:
    def __init__(self,root,scene):
        self.root=root;self.scene=scene;self.dest=root/"scenes"/scene;self.req=read(root/"request.json")
        self.spec=next(s for s in self.req["scenes"] if s["scene"]==scene);self.child=None
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.spec["gpu_uuid"],CUDA_DEVICE_ORDER="PCI_BUS_ID",
            GPU=str(self.spec["gpu"]),PYTHONUNBUFFERED="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
        self.env.pop("PYTHONPATH",None)
        self.state=dict(pid=os.getpid(),scene=scene,gpu=self.spec["gpu"],gpu_uuid=self.spec["gpu_uuid"])

    def save(self,**kw):
        self.state.update(kw,updated_at=now());write(self.dest/"status.json",self.state)

    def stop(self,*_):
        if self.child and self.child.poll() is None:os.killpg(self.child.pid,signal.SIGTERM)
        self.save(state="stopped");raise InterruptedError("Pilot stopped; saved checkpoints preserved")

    def execute(self,stage,cmd,cwd,expected,gpu=False):
        if self.req["code"]!=codes():raise ValueError("Pilot source changed")
        receipt=self.dest/"receipts"/(stage+".json");cmd=list(map(str,cmd))
        if receipt.exists():
            r=read(receipt)
            if r["command"]!=cmd:raise ValueError("Stage command changed")
            for v in r["artifacts"]:verify_record(v)
            return
        if shutil.disk_usage(self.root).free<30*1024**3:raise RuntimeError("Insufficient output disk space")
        if gpu:
            minimum=6000 if self.scene=="cube" else 8000
            while True:
                free=int(subprocess.check_output(["nvidia-smi","-i",self.spec["gpu_uuid"],"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if free>=minimum:break
                self.save(state="waiting_for_gpu_memory",stage=stage,free_mib=free,required_mib=minimum)
                print(now(),"Waiting for free GPU memory",free,flush=True);time.sleep(30)
        logfile=self.dest/"logs"/(stage+".log");logfile.parent.mkdir(parents=True,exist_ok=True)
        self.save(state="running",stage=stage,command=cmd,log=str(logfile),progress=None)
        print(now(),self.scene,stage,flush=True)
        child_env = dict(self.env)
        if cmd[0] == LAMA:
            child_env["PYTHONPATH"] = str(REPO / "submodules/Inpaint360GS/LaMa")
        with logfile.open("a") as log:
            self.child=subprocess.Popen(cmd,cwd=cwd,env=child_env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            self.save(child_pid=self.child.pid)
            for line in self.child.stdout:
                log.write(line);log.flush()
                if line.startswith("PGSR joint ") or "Guide ready" in line or "error:" in line:
                    print(line.rstrip(),flush=True);self.save(progress=line.rstrip())
            code=self.child.wait()
        self.save(child_pid=None)
        if code:raise RuntimeError(f"{stage} failed ({code}); see {logfile}")
        write(receipt,dict(command=cmd,complete=True,finished_at=now(),artifacts=[record(p) for p in expected]))

    def run(self):
        with (self.dest/"run.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for k in ("source","config"):verify_record(self.spec[k])
            self.execute("prepare_guide",[PYTHON,GUIDE,"prepare","--root",self.dest],REPO,[self.dest/"guide/evidence.json"])
            if read(self.dest/"guide/evidence.json")["guided_views"]:
                source=read(self.dest/"source.json");lama=read(source["inputs"]["lama"]["path"])
                model=Path(lama["model"]["config"]["path"]).parent
                expected=[self.dest/"lama/output/color"/(s+".png") for s,v in read(self.dest/"guide/evidence.json")["frames"].items() if v["mode"]=="warp_and_lama_residual"]
                self.execute("lama_shared_reference",[LAMA,"bin/predict_color.py","--input-dir",self.dest/"guide/input",
                    "--output-dir",self.dest/"lama/output/color","--model-path",model],REPO/"submodules/Inpaint360GS/LaMa",expected,gpu=True)
            self.execute("validate_and_bind",[PYTHON,GUIDE,"finalize","--root",self.dest],REPO,
                [self.dest/"training_source.json",self.dest/"guide/validation.json"])
            row=read(self.dest/"training_source.json")
            self.execute("train_10000",Runner.training_command(self,row,self.dest),REPO/"submodules/EDGS",
                [self.dest/"manifests/edgs_joint_manifest.json",self.dest/"edgs_joint/snapshots/iteration_5000/manifest.json"],gpu=True)
            from run_inpaint360_full import export_snapshot
            export_snapshot(self.dest,5000)
            self.execute("publish_10000",Runner.publish_command(self,row,self.dest),REPO/"submodules/Inpaint360GS",
                [self.dest/"inpainted_3dgs_10000/model_manifest.json"])
            for step in (5000,10000):
                model=self.dest/f"inpainted_3dgs_{step}";result=model/"evaluation/inpaint360_r4/results.json"
                self.execute(f"evaluate_{step}",[PYTHON,REPO/"scripts/paintmesh/eval_inpaint360.py","run",
                    "--model-path",model,"--source-path",row["source"],"--resolution",4,"--iteration",step,"--batch-size",4],REPO,[result],gpu=True)
                validate_evaluation(read(result),row["historical_evaluation"]);report(self.root)
            self.save(state="complete",stage="complete",child_pid=None,finished_at=now());report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path)
    p.add_argument("--worker",choices=tuple(GPU_MAP));p.add_argument("--prepare-only",action="store_true");p.add_argument("--report",action="store_true")
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.report:report(root);return
    if a.worker:
        w=Worker(root,a.worker)
        try:w.run()
        except BaseException as e:
            if not isinstance(e,InterruptedError):w.save(state="failed",error=str(e),child_pid=None)
            traceback.print_exc();sys.exit(1)
        return
    prepare(root)
    if a.prepare_only:return
    processes={}
    for scene in GPU_MAP:
        dest=root/"scenes"/scene
        state=read(dest/"status.json") if (dest/"status.json").exists() else {}
        if state.get("state")=="complete":continue
        if state.get("pid") and Path(f"/proc/{state['pid']}/cmdline").exists():
            cmd=Path(f"/proc/{state['pid']}/cmdline").read_bytes().replace(b"\0",b" ").decode()
            if str(root) in cmd and "--worker" in cmd:raise RuntimeError("Worker already running: "+scene)
        with (dest/"worker.log").open("a") as log:
            process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),"--root",str(root),"--worker",scene],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[scene]=dict(pid=process.pid,gpu=GPU_MAP[scene]);write(root/"workers.json",processes)
    print(root,processes,flush=True)


if __name__=="__main__":main()
