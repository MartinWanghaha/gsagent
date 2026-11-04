#!/usr/bin/env python3
"""Two independent GPU workers for an existing immutable muti_plan experiment.

Only scheduling changes: reuse original training commands, receipts and request
hashes so existing checkpoints retain their exact optimizer/RNG trajectory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from run_muti_plan import (DEFAULT_ROOT, PYTHON, REPO, Runner, atomic, now, read,
                           report, sha, validate_evaluation, verify_code, write)


@contextmanager
def queue_lock(root):
    with (root / "parallel/queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def choose_task(tasks, gpu):
    """Reserved resumable jobs stay on their original GPU; all others are shared."""
    for task in tasks:
        if task["state"] == "pending" and task.get("preferred_gpu") in (None, gpu):
            return task
    return None


def claim_task(root, gpu):
    with queue_lock(root):
        queue = read(root / "parallel/queue.json")
        task = choose_task(queue["tasks"], gpu)
        if task is None:
            return None
        task.update(state="running", gpu=gpu, worker_pid=os.getpid(), started_at=now())
        write(root / "parallel/queue.json", queue)
        return dict(task)


def finish_task(root, key, gpu, state, error=None):
    with queue_lock(root):
        queue = read(root / "parallel/queue.json")
        task = next(t for t in queue["tasks"] if t["key"] == key)
        if task.get("worker_pid") != os.getpid() or task.get("gpu") != gpu or task["state"] != "running":
            raise ValueError("Task ownership changed: " + key)
        task.update(state=state, finished_at=now())
        if error:
            task["error"] = error
        write(root / "parallel/queue.json", queue)


def prepare(root, gpus):
    if len(gpus) != len(set(gpus)) or not gpus or min(gpus) < 0:
        raise ValueError("GPU indices must be nonnegative and unique")
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        req = read(root / "request.json")
        verify_code(req)
        if not read(root / "preflight.json")["complete"]:
            raise ValueError("Original input preflight incomplete")
        directory = root / "parallel"
        directory.mkdir(exist_ok=True)
        if (directory / "request.json").exists():
            raise ValueError("Parallel plan already exists; use --resume --detach")
        old = read(root / "status.json")
        inventory, completed = [], []
        import torch
        from local_geometry_io import identity
        for s in req["scenes"]:
            for v in s["variants"]:
                key = s["scene"] + "/" + v["name"]
                output = root / "scenes" / s["scene"] / v["name"]
                files = [output / f"inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json" for step in (5000, 10000)]
                if all(p.exists() for p in files):
                    for p in files:
                        validate_evaluation(read(p), s["historical_evaluation"])
                    completed.append(dict(key=key, scene=s["scene"], variant=v["name"], state="complete",
                        results=[dict(path=str(p), sha256=sha(p)) for p in files]))
                    continue
                item = dict(key=key, scene=s["scene"], variant=v["name"], state="pending", resume_step=0)
                checkpoint = output / "edgs_joint/checkpoints/latest.pth"
                if checkpoint.exists() and not (output / "manifests/edgs_joint_manifest.json").exists():
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if saved["request_id"] != identity(read(output / "edgs_joint/request.json")):
                        raise ValueError("Checkpoint/request mismatch: " + key)
                    required = {"local", "optimizer", "view_order", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
                    if not required <= saved.keys():
                        raise ValueError("Incomplete resumable checkpoint: " + key)
                    item.update(resume_step=saved["next_step"], checkpoint=str(checkpoint),
                                checkpoint_sha256=sha(checkpoint))
                    if req["gpu"] in gpus:
                        item["preferred_gpu"] = req["gpu"]
                    del saved
                inventory.append(item)
        devices = {str(g): subprocess.check_output(["nvidia-smi", "-i", str(g), "--query-gpu=uuid",
                    "--format=csv,noheader"], text=True).strip() for g in gpus}
        parallel = dict(created_at=now(), gpus=gpus, devices=devices,
            original_request_sha256=sha(root / "request.json"), dispatcher_sha256=sha(__file__),
            completed_before_switch=completed, remaining_at_switch=inventory,
            policy="one job per GPU; atomic dynamic claiming; unchanged training inputs/objective/optimizer; resume checkpoints")
        write(directory / "request.json", parallel)
        write(directory / "queue.json", dict(tasks=inventory))
        write(directory / "previous_status.json", old)
        shutil.copyfile(__file__, directory / "dispatcher.py.snapshot")
        text = (root / "TRAINING_PLAN.zh-CN.md").read_text()
        text = text.replace(f"运行设备：GPU {req['gpu']}（{req['gpu_uuid']}）。", "运行设备：GPU 0 和 GPU 1；两条独立队列并行。")
        text = text.replace("串行执行，优先 bag/truck/cube/garden_toys 等诊断场景。每个场景按 control→no_normal→uniform→both 执行，之后进入下一个场景。",
            "双卡并行：每张 GPU 同时运行一组独立实验，空闲卡从剩余队列领取下一组。优先续跑已有检查点，然后沿用原场景/变体优先顺序。")
        text = text.replace(f"{PYTHON} {REPO / 'scripts/paintmesh/run_muti_plan.py'} --root {root} --resume --detach",
            f"{PYTHON} {Path(__file__).resolve()} --root {root} --resume --detach")
        text += (f"\n## 双卡调度调整（{now()}）\n\n已停止原单卡队列；{len(completed)} 组完整结果保留，剩余 {len(inventory)} 组由 GPU 0/1 动态领取。"
                 "原 request.json 和训练代码哈希不改写；只新增调度器，不修改训练实现，因此旧 checkpoint 可以恢复优化器及 RNG。\n\n"
                 "运行总览：status.json；卡内状态：parallel/workers/gpu0.json、gpu1.json；队列：parallel/queue.json。"
                 "日志：logs/parallel_gpu0.log、logs/parallel_gpu1.log、logs/parallel_controller.log；各实验原阶段日志继续追加。"
                 "comparison.md 由控制进程定期统一写入，避免两张卡相互覆盖；完成项目不会重复分配。\n")
        atomic(root / "TRAINING_PLAN.zh-CN.md", text)
        atomic(root / "README.md", "# PaintMesh muti_plan\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [实时比较](comparison.md) · [双卡状态](status.json)\n\n"
            "GPU 0 / GPU 1 并行，训练配置保持原样。\n\n[GPU 0 日志](logs/parallel_gpu0.log) · [GPU 1 日志](logs/parallel_gpu1.log) · [控制器日志](logs/parallel_controller.log)\n")
        print(dict(completed=len(completed), remaining=len(inventory), gpus=gpus,
                   resumable=[(t["key"], t["resume_step"]) for t in inventory if t["resume_step"]]), flush=True)


def verify_parallel(root):
    parallel = read(root / "parallel/request.json")
    if sha(root / "request.json") != parallel["original_request_sha256"]:
        raise ValueError("Original training request changed")
    if sha(__file__) != parallel["dispatcher_sha256"]:
        raise ValueError("Dispatcher changed after scheduling")
    verify_code(read(root / "request.json"))
    return parallel


class Worker(Runner):
    def __init__(self, root, gpu):
        super().__init__(root)
        self.gpu = gpu
        parallel = verify_parallel(root)
        self.env.update(CUDA_VISIBLE_DEVICES=parallel["devices"][str(gpu)], GPU=str(gpu))
        self.state = dict(state="starting", gpu=gpu, gpu_uuid=parallel["devices"][str(gpu)], pid=os.getpid())
        self.state_path = root / f"parallel/workers/gpu{gpu}.json"
        self.task = None

    def save(self, **values):
        self.state.update(values, updated_at=now())
        write(self.state_path, self.state)

    def stop(self, *_):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        self.save(state="stopped", child_pid=None)
        raise InterruptedError("Parallel worker stopped; resume from last atomic checkpoint")

    def case(self, s, v):
        # The original command builders and stage receipts are deliberately reused.
        from run_inpaint360_full import export_snapshot
        root = self.root / "scenes" / s["scene"] / v["name"]
        self.execute(s["scene"], v["name"], "train_10000", self.training_command(s, root), REPO / "submodules/EDGS",
                     [root / "manifests/edgs_joint_manifest.json", root / "edgs_joint/snapshots/iteration_5000/manifest.json"])
        export_snapshot(root, 5000)
        self.execute(s["scene"], v["name"], "publish_10000", self.publish_command(s, root), REPO / "submodules/Inpaint360GS",
                     [root / "inpainted_3dgs_10000/model_manifest.json"])
        for step in (5000, 10000):
            model = root / f"inpainted_3dgs_{step}"
            result = model / "evaluation/inpaint360_r4/results.json"
            self.execute(s["scene"], v["name"], f"evaluate_{step}", [PYTHON, REPO / "scripts/paintmesh/eval_inpaint360.py", "run",
                "--model-path", model, "--source-path", s["source"], "--resolution", "4", "--iteration", str(step), "--batch-size", "4"], REPO, [result])
            validate_evaluation(read(result), s["historical_evaluation"])

    def run(self):
        with (self.root / f"parallel/gpu{self.gpu}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
            self.save(state="idle")
            scenes = {s["scene"]: s for s in self.req["scenes"]}
            while (task := claim_task(self.root, self.gpu)) is not None:
                self.task = task
                s = scenes[task["scene"]]
                v = next(v for v in s["variants"] if v["name"] == task["variant"])
                self.save(state="running", scene=s["scene"], variant=v["name"], resume_step=task["resume_step"])
                print(f"GPU {self.gpu} claimed {task['key']} (resume step {task['resume_step']})", flush=True)
                try:
                    self.case(s, v)
                except InterruptedError as exc:
                    finish_task(self.root, task["key"], self.gpu, "interrupted", str(exc))
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    finish_task(self.root, task["key"], self.gpu, "failed", str(exc))
                    self.save(state="idle", error=str(exc), child_pid=None)
                else:
                    finish_task(self.root, task["key"], self.gpu, "complete")
                    self.save(state="idle", error=None, child_pid=None)
                self.task = None
            self.save(state="complete", child_pid=None, finished_at=now())


def aggregate(root, parallel, state, children):
    queue = read(root / "parallel/queue.json")
    cases = {t["key"]: dict(t) for t in parallel["completed_before_switch"] + queue["tasks"]}
    workers = {}
    for gpu in parallel["gpus"]:
        path = root / f"parallel/workers/gpu{gpu}.json"
        workers[str(gpu)] = read(path) if path.exists() else dict(state="starting")
    status = dict(state=state, mode="parallel", pid=os.getpid(), gpus=parallel["gpus"], updated_at=now(),
        workers=workers, worker_pids={str(g): p.pid for g, p in children.items()}, cases=cases,
        completed=sum(t["state"] == "complete" for t in cases.values()), total=len(cases),
        failed=[k for k, t in cases.items() if t["state"] == "failed"])
    # Only the controller writes aggregate status and reports. Worker writes are
    # isolated by GPU; task mutations are serialized by queue.lock.
    write(root / "status.json", status)
    report(root)


def controller(root):
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parallel = verify_parallel(root)
        children = {}
        handles = []
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True
            for process in children.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        # GPU locks protect against live orphaned workers after controller loss.
        for gpu in parallel["gpus"]:
            with (root / f"parallel/gpu{gpu}.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with queue_lock(root):
            queue = read(root / "parallel/queue.json")
            for task in queue["tasks"]:
                if task["state"] in ("running", "interrupted", "failed"):
                    task["state"] = "pending"
                    task.pop("error", None)
            write(root / "parallel/queue.json", queue)
        for gpu in parallel["gpus"]:
            logfile = root / f"logs/parallel_gpu{gpu}.log"
            handle = logfile.open("a")
            handles.append(handle)
            children[gpu] = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--root", str(root), "--worker", str(gpu)],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1"))
        write(root / "parallel/processes.json", dict(controller_pid=os.getpid(), workers={str(g): p.pid for g, p in children.items()}, started_at=now()))
        print("Parallel workers:", {g: p.pid for g, p in children.items()}, flush=True)
        try:
            while any(p.poll() is None for p in children.values()):
                aggregate(root, parallel, "stopping" if stopping else "running", children)
                time.sleep(5)
            queue = read(root / "parallel/queue.json")
            complete = all(t["state"] == "complete" for t in queue["tasks"])
            state = "stopped" if stopping else ("complete" if complete else "completed_with_failures")
            aggregate(root, parallel, state, children)
            print("Parallel queue:", state, "worker exit codes:", {g: p.returncode for g, p in children.items()}, flush=True)
        except BaseException:
            stop()
            for p in children.values():
                p.wait(timeout=30)
            aggregate(root, parallel, "stopped", children)
            raise
        finally:
            for handle in handles:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--worker", type=int)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root, args.gpus)
    elif args.worker is not None:
        Worker(root, args.worker).run()
    elif args.detach:
        verify_parallel(root)
        with (root / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = root / "logs/parallel_controller.log"
        with log.open("a") as handle:
            process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--root", str(root), "--resume"],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1"))
        atomic(root / "run.pid", str(process.pid) + "\n")
        print(dict(controller_pid=process.pid, log=str(log)), flush=True)
    else:
        controller(root)


if __name__ == "__main__":
    main()
