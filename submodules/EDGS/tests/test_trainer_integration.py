from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import source.trainer as trainer_module  # noqa: E402
from source.trainer import EDGSTrainer  # noqa: E402


class _FakeLPIPS(torch.nn.Module):
    def forward(self, image, target):
        return (image - target).square().mean()


class _FakeScene:
    def __init__(self, cameras=()):
        self._cameras = list(cameras)

    def getTrainCameras(self):
        return self._cameras


class _FakeGaussians:
    def __init__(self):
        self.value = torch.nn.Parameter(torch.tensor(0.5))
        self.optimizer = torch.optim.SGD((self.value,), lr=0.1)
        self.max_radii2D = torch.zeros(3)
        self.learning_rate_step = None
        self.densification_mask = None

    def update_learning_rate(self, step):
        self.learning_rate_step = step

    def oneupSHdegree(self):
        raise AssertionError("SH degree must not change in this test")

    def add_densification_stats(self, viewspace_points, visibility):
        assert viewspace_points.grad is not None
        self.densification_mask = visibility.clone()


class _FakeGS:
    def __init__(self, backend="native", cameras=()):
        self.gaussians = _FakeGaussians()
        self.renderer = SimpleNamespace(backend=backend)
        self.scene = _FakeScene(cameras)
        self.viewpoint_stack = list(cameras)
        self.calls = []

    def __call__(self, viewpoint_cam, **options):
        self.calls.append((viewpoint_cam, options))
        screen = torch.zeros(3, 3, requires_grad=True)
        image = (self.gaussians.value + screen.sum() * 0.0).expand(3, 2, 2)
        return {
            "render": image,
            "viewspace_points": screen,
            "visibility_filter": torch.tensor(((0,), (2,))),
            "radii": torch.tensor((1.0, 0.0, 2.0)),
        }


class _FakeTimer:
    def pause(self):
        pass

    def start(self):
        pass


class _FakeComposer:
    def __init__(self, neighbor):
        self.neighbor = neighbor

    def required_outputs(self, step):
        assert step == 1
        return True, True

    def multi_view_active(self, step):
        assert step == 1
        return True

    def photo_terms(self, image, target):
        return {"photo": (image - target).square().mean()}

    def pgsr_terms(
        self,
        step,
        camera,
        pkg,
        gaussians,
        neighbors,
        render_neighbor,
        diagnostics=None,
    ):
        neighbor_pkg = render_neighbor(self.neighbor)
        if diagnostics is not None:
            diagnostics["reprojection_weight"] = torch.ones(2, 2)
        return {"geometry": 0.25 * neighbor_pkg["render"].mean()}


class _FakeDebugVisualizer:
    def __init__(self, capture=False):
        self.capture = capture
        self.saved = []

    def should_capture(self, step):
        return self.capture

    def save(self, step, camera, package, target, diagnostics):
        self.saved.append((step, camera, package, target, diagnostics))
        return Path("debug/frame.jpg")


def _training_config(**values):
    defaults = {
        "lambda_dssim": 0.2,
        "save_iterations": [],
        "batch_size": 1,
        "densify_until_iter": 100,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_trainer_keeps_legacy_config_on_native_backend(monkeypatch):
    monkeypatch.setattr(trainer_module.lpips, "LPIPS", lambda **_: _FakeLPIPS())
    gs = _FakeGS(backend="native")

    trainer = EDGSTrainer(
        gs,
        _training_config(),
        device="cpu",
        log_wandb=False,
    )

    assert trainer.loss_composer.enabled is False
    assert trainer.neighbor_map == {}
    assert trainer.lpips is None
    assert isinstance(trainer._get_lpips(), _FakeLPIPS)


def test_trainer_rejects_pgsr_loss_with_native_renderer():
    gs = _FakeGS(backend="native")

    with pytest.raises(ValueError, match="renderer.backend=pgsr"):
        EDGSTrainer(
            gs,
            _training_config(pgsr_loss={"enabled": True}),
            device="cpu",
            log_wandb=False,
        )


def test_checkpoint_restore_rebuilds_radii_and_uses_checkpoint_step(tmp_path):
    class RestoredGaussians:
        def __init__(self):
            self.optimizer = object()
            self.restored = None
            self._xyz = torch.empty(0, 3)

        @property
        def get_xyz(self):
            return self._xyz

        def restore(self, state, training_config):
            self.restored = (state, training_config)
            self._xyz = torch.zeros(4, 3)
            self.optimizer = object()

    checkpoint_path = tmp_path / "chkpnt7.pth"
    torch.save(({"model": "state"}, 7), checkpoint_path)
    gaussians = RestoredGaussians()
    trainer = EDGSTrainer.__new__(EDGSTrainer)
    trainer.GS = SimpleNamespace(gaussians=gaussians)
    trainer.training_config = _training_config()
    trainer.device = torch.device("cpu")
    trainer.CONSOLE = SimpleNamespace(print=lambda *args, **kwargs: None)
    trainer.training_step = 1
    trainer.gs_step = 0

    trainer.load_checkpoints(SimpleNamespace(gs=str(tmp_path), gs_step=7))

    assert gaussians.restored == ({"model": "state"}, trainer.training_config)
    assert gaussians.tmp_radii.shape == (4,)
    assert trainer.GS_optimizer is gaussians.optimizer
    assert trainer.training_step == 8
    assert trainer.gs_step == 7


def test_correspondence_initializer_routes_mvroma_without_loading_roma(monkeypatch):
    import source.correspondence as correspondence

    calls = []

    def initialize(gaussians, scene, cfg, device, **options):
        calls.append((gaussians, scene, cfg, device, options))
        return ["camera"], torch.tensor(((1,),)), {"backend": "mvroma"}

    monkeypatch.setattr(correspondence, "init_gaussians_with_mvroma", initialize)
    gaussians = SimpleNamespace(
        _xyz=torch.zeros(2, 3),
        _scaling=torch.ones(2, 3),
        scaling_activation=lambda value: value,
        scaling_inverse_activation=lambda value: value,
    )
    scene = object()
    trainer = EDGSTrainer.__new__(EDGSTrainer)
    trainer.GS = SimpleNamespace(gaussians=gaussians)
    trainer.gaussians = gaussians
    trainer.scene = scene
    trainer.device = torch.device("cpu")
    config = SimpleNamespace(use=True, backend="mvroma", add_SfM_init=True)
    model = object()
    prematcher = object()

    result = trainer.init_with_corr(
        config,
        mvroma_model=model,
        mvroma_prematcher=prematcher,
    )

    assert result == {"backend": "mvroma"}
    assert calls == [
        (
            gaussians,
            scene,
            config,
            torch.device("cpu"),
            {"verbose": False, "model": model, "prematcher": prematcher},
        )
    ]
    assert torch.allclose(gaussians._scaling, torch.full((2, 3), 0.5))


def test_checkpoint_resume_restores_only_explicit_mvroma_hybrid_graph(monkeypatch):
    import source.correspondence.manifest as manifest

    cameras = [SimpleNamespace(image_name=name) for name in ("a", "b")]
    restored = {"a": [cameras[1]], "b": [cameras[0]]}
    calls = []
    monkeypatch.setattr(
        manifest,
        "load_pgsr_neighbors",
        lambda model_path, current: calls.append((model_path, current)) or restored,
    )
    trainer = EDGSTrainer.__new__(EDGSTrainer)
    trainer.scene = SimpleNamespace(
        model_path="/model",
        getTrainCameras=lambda: cameras,
    )
    trainer.loss_composer = SimpleNamespace(enabled=True)
    trainer.neighbor_map = {}

    trainer.configure_correspondence_resume(
        SimpleNamespace(
            use=True,
            backend="mvroma",
            mvroma={"training": {"pgsr_neighbor_strategy": "hybrid"}},
        )
    )

    assert trainer.neighbor_map == restored
    assert calls == [("/model", cameras)]

    trainer.configure_correspondence_resume(
        SimpleNamespace(use=True, backend="roma")
    )
    assert len(calls) == 1


def test_train_step_requests_geometry_and_normalizes_visibility():
    reference = SimpleNamespace(original_image=torch.zeros(3, 2, 2))
    neighbor = SimpleNamespace(original_image=torch.zeros(3, 2, 2))
    gs = _FakeGS(backend="pgsr", cameras=(reference,))
    trainer = EDGSTrainer.__new__(EDGSTrainer)
    trainer.GS = gs
    trainer.gaussians = gs.gaussians
    trainer.GS_optimizer = gs.gaussians.optimizer
    trainer.scene = gs.scene
    trainer.viewpoint_stack = [reference]
    trainer.training_config = _training_config()
    trainer.loss_composer = _FakeComposer(neighbor)
    trainer.pgsr_debug = _FakeDebugVisualizer()
    trainer.neighbor_map = {"reference": [neighbor]}
    trainer.training_step = 1
    trainer.gs_step = 0
    trainer.device = torch.device("cpu")
    trainer.timer = _FakeTimer()
    trainer.logs_losses = {}
    trainer.log_wandb = False
    trainer.ema_loss_for_log = 0.0

    radii = trainer.train_step_gs(max_lr=False, no_densify=False)

    assert gs.gaussians.learning_rate_step == 1
    assert gs.calls[0][1] == {
        "return_plane": True,
        "return_depth_normal": True,
    }
    assert gs.calls[1][1] == {
        "return_plane": True,
        "return_depth_normal": False,
    }
    assert gs.gaussians.densification_mask.tolist() == [True, False, True]
    assert gs.gaussians.max_radii2D.tolist() == [1.0, 0.0, 2.0]
    assert radii.tolist() == [1.0, 0.0, 2.0]
    assert set(trainer.logs_losses[1]) == {"loss", "photo", "geometry"}


def test_debug_capture_forces_full_plane_outputs_and_reuses_training_package():
    class PhotoOnlyComposer(_FakeComposer):
        def required_outputs(self, step):
            return False, False

    reference = SimpleNamespace(original_image=torch.zeros(3, 2, 2))
    neighbor = SimpleNamespace(original_image=torch.zeros(3, 2, 2))
    gs = _FakeGS(backend="pgsr", cameras=(reference,))
    trainer = EDGSTrainer.__new__(EDGSTrainer)
    trainer.GS = gs
    trainer.gaussians = gs.gaussians
    trainer.GS_optimizer = gs.gaussians.optimizer
    trainer.scene = gs.scene
    trainer.viewpoint_stack = [reference]
    trainer.training_config = _training_config()
    trainer.loss_composer = PhotoOnlyComposer(neighbor)
    trainer.pgsr_debug = _FakeDebugVisualizer(capture=True)
    trainer.neighbor_map = {"reference": [neighbor]}
    trainer.training_step = 1
    trainer.gs_step = 0
    trainer.device = torch.device("cpu")
    trainer.timer = _FakeTimer()
    trainer.logs_losses = {}
    trainer.log_wandb = False
    trainer.ema_loss_for_log = 0.0
    trainer.CONSOLE = SimpleNamespace(print=lambda *args, **kwargs: None)

    trainer.train_step_gs(max_lr=False, no_densify=True)

    assert gs.calls[0][1] == {
        "return_plane": True,
        "return_depth_normal": True,
    }
    assert len(trainer.pgsr_debug.saved) == 1
    saved = trainer.pgsr_debug.saved[0]
    assert saved[0] == 1
    # One reference render plus the loss composer's one neighbor render.  The
    # visualizer must not trigger a third render.
    assert len(gs.calls) == 2
    assert saved[4]["reprojection_weight"].shape == (2, 2)
