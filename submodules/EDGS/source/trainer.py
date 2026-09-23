from pathlib import Path
from random import randint

import lpips
import torch
import wandb
from rich.console import Console
from rich.theme import Theme
from tqdm.rich import trange

from source.losses import l1_loss, psnr, ssim
from source.networks import Warper3DGS
from source.pgsr_debug import PGSRDebugVisualizer
from source.pgsr_geometry import build_neighbor_map
from source.pgsr_losses import PGSRLossComposer
from source.renderers import visible_mask
from source.timer import Timer
from source.utils_aux import log_samples

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
    }
)


class EDGSTrainer:
    def __init__(
        self,
        GS: Warper3DGS,
        training_config,
        dataset_white_background=False,
        device=torch.device("cuda"),
        log_wandb=True,
    ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background
        self.device = torch.device(device)

        pgsr_config = getattr(training_config, "pgsr_loss", None)
        if pgsr_config is None:
            pgsr_config = {"enabled": False}
        self.loss_composer = PGSRLossComposer(
            pgsr_config,
            lambda_dssim=training_config.lambda_dssim,
        )
        if self.loss_composer.enabled and GS.renderer.backend != "pgsr":
            raise ValueError(
                "PGSR losses require renderer.backend=pgsr; use the gs=pgsr profile"
            )
        self.pgsr_debug = PGSRDebugVisualizer(
            getattr(training_config, "pgsr_debug", None),
            model_path=getattr(self.scene, "model_path", "."),
            default_from_iter=self.loss_composer.multi_view_from_iter,
        )
        if self.pgsr_debug.enabled and GS.renderer.backend != "pgsr":
            raise ValueError(
                "PGSR debug visualization requires renderer.backend=pgsr; "
                "use the gs=pgsr profile"
            )
        if self.pgsr_debug.enabled and not (
            self.loss_composer.enabled
            and (
                self.loss_composer.geo_weight != 0.0
                or self.loss_composer.ncc_weight != 0.0
            )
        ):
            raise ValueError(
                "PGSR debug visualization requires an enabled multi-view "
                "geometry or NCC loss"
            )
        self.pgsr_debug.prepare()
        self.neighbor_map = (
            build_neighbor_map(self.scene.getTrainCameras(), pgsr_config)
            if self.loss_composer.enabled
            else {}
        )
        self._mvroma_context = None

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        # LPIPS is only used during evaluation.  Delaying the VGG allocation is
        # especially important for MV-RoMa initialization, whose multi-view
        # transformer and UFM prematcher briefly occupy substantial GPU memory.
        self.lpips = None
        self.timer = Timer()
        self.log_wandb = log_wandb

    def _get_lpips(self):
        """Build the evaluation-only LPIPS network on first use."""

        if self.lpips is None:
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            self.lpips.eval()
        return self.lpips

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            if load_cfg.gs_step is None:
                raise ValueError("load.gs_step is required when load.gs is set")
            checkpoint_path = Path(load_cfg.gs) / f"chkpnt{load_cfg.gs_step}.pth"
            model_state, checkpoint_step = torch.load(
                checkpoint_path,
                map_location=self.device,
            )
            self.GS.gaussians.restore(model_state, self.training_config)
            self.GS.gaussians.tmp_radii = torch.zeros(
                self.GS.gaussians.get_xyz.shape[0],
                dtype=self.GS.gaussians.get_xyz.dtype,
                device=self.GS.gaussians.get_xyz.device,
            )
            self.GS_optimizer = self.GS.gaussians.optimizer
            resume_step = int(checkpoint_step)
            if resume_step != int(load_cfg.gs_step):
                self.CONSOLE.print(
                    f"Checkpoint metadata reports iteration {resume_step}; "
                    f"the requested filename used {load_cfg.gs_step}.",
                    style="warning",
                )
            self.CONSOLE.print(
                f"3DGS loaded from checkpoint for iteration {resume_step}",
                style="info",
            )
            self.training_step += resume_step
            self.gs_step += resume_step

    def train(self, train_cfg):
        # 3DGS training
        self.CONSOLE.print(
            "Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info"
        )
        with trange(
            self.training_step,
            self.training_step + train_cfg.gs_epochs,
            desc="[green]Train gaussians",
        ) as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(
                    max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify
                )
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if (
                            self.gs_step < self.training_config.densify_until_iter
                            and self.gs_step % 10 == 0
                        ):
                            opacities_new = torch.log(
                                torch.exp(self.GS.gaussians._opacity.data) * 0.99
                            )
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix(
                            {"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"},
                            refresh=True,
                        )
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (
                            self.training_step <= 3000 and self.training_step % 500 == 0
                        ) or (
                            self.training_step > 3000
                            and self.training_step % 1000 == 228
                        ):
                            self.evaluate()

                    self.timer.start()

    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        train_cameras = self.scene.getTrainCameras()
        validation_configs = (
            {
                "name": "test",
                "cameras": self.scene.getTestCameras(),
                "cam_idx": self.training_config.TEST_CAM_IDX_TO_LOG,
            },
            {
                "name": "train",
                "cameras": (
                    [
                        train_cameras[idx % len(train_cameras)]
                        for idx in range(0, 150, 5)
                    ]
                    if train_cameras
                    else []
                ),
                "cam_idx": 10,
            },
        )
        if self.log_wandb:
            wandb.log(
                {f"Number of Gaussians": len(self.GS.gaussians._xyz)},
                step=self.training_step,
            )
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        self.GS(
                            viewpoint,
                            return_plane=False,
                            return_depth_normal=False,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to(self.device), 0.0, 1.0
                    )
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(
                        image.unsqueeze(0), gt_image.unsqueeze(0)
                    ).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += (
                        self._get_lpips()(image, gt_image).detach().double()
                    )
                    if idx in [config["cam_idx"]]:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_splat_test /= len(config["cameras"])
                if self.log_wandb:
                    wandb.log(
                        {
                            f"{config['name']}/L1": l1_test.item(),
                            f"{config['name']}/PSNR": psnr_test.item(),
                            f"{config['name']}/SSIM": ssim_test.item(),
                            f"{config['name']}/LPIPS_splat": lpips_splat_test.item(),
                        },
                        step=self.training_step,
                    )
                self.CONSOLE.print(
                    "\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                        self.training_step,
                        len(self.GS.gaussians._xyz),
                        config["name"],
                        l1_test.item(),
                        psnr_test.item(),
                        ssim_test.item(),
                        lpips_splat_test.item(),
                    ),
                    style="info",
                )
        if self.log_wandb and log_real_images and log_gen_images:
            with torch.no_grad():
                log_samples(
                    torch.stack((log_real_images[0], log_gen_images[0])),
                    [],
                    self.training_step,
                    caption="Real and Generated Samples",
                )
        if self.log_wandb:
            wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr=False, no_densify=False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(
            randint(0, len(self.viewpoint_stack) - 1)
        )

        need_plane, need_depth_normal = self.loss_composer.required_outputs(
            self.gs_step
        )
        capture_pgsr_debug = self.pgsr_debug.should_capture(
            self.gs_step
        ) and self.loss_composer.multi_view_active(self.gs_step)
        if capture_pgsr_debug:
            # Debug images need the complete plane-aware package even if the
            # corresponding geometry loss is disabled at this iteration.
            need_plane = True
            need_depth_normal = True
        self.GS_optimizer.zero_grad(set_to_none=True)
        render_pkg = self.GS(
            viewpoint_cam=viewpoint_cam,
            return_plane=need_plane,
            return_depth_normal=need_depth_normal,
        )
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.to(self.device)

        loss_terms = self.loss_composer.photo_terms(image, gt_image)
        diagnostics = {} if capture_pgsr_debug else None
        pgsr_arguments = {}
        if diagnostics is not None:
            pgsr_arguments["diagnostics"] = diagnostics
        loss_terms.update(
            self.loss_composer.pgsr_terms(
                self.gs_step,
                viewpoint_cam,
                render_pkg,
                self.gaussians,
                self.neighbor_map,
                render_neighbor=lambda neighbor: self.GS(
                    viewpoint_cam=neighbor,
                    return_plane=True,
                    return_depth_normal=False,
                ),
                **pgsr_arguments,
            )
        )
        loss = sum(loss_terms.values(), image.new_zeros(()))

        self.timer.pause()
        # Match PGSR's original behavior: a montage is meaningful only when
        # this exact training step selected a real neighbor and computed the
        # geometric reprojection weights shown in its lower-left panel.
        if diagnostics is not None and "reprojection_weight" in diagnostics:
            debug_path = self.pgsr_debug.save(
                self.gs_step,
                viewpoint_cam,
                render_pkg,
                gt_image,
                diagnostics,
            )
            self.CONSOLE.print(f"PGSR debug visualization: {debug_path}", style="info")
        self.logs_losses[self.training_step] = {
            "loss": loss.detach().item(),
            **{name: value.detach().item() for name, value in loss_terms.items()},
        }

        if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = (
            0.4 * self.logs_losses[self.training_step]["loss"]
            + 0.6 * self.ema_loss_for_log
        )
        self.timer.start()
        loss.backward()
        with torch.no_grad():
            if (
                self.gs_step < self.training_config.densify_until_iter
                and not no_densify
            ):
                visibility = visible_mask(render_pkg)
                self.GS.gaussians.max_radii2D[visibility] = torch.max(
                    self.GS.gaussians.max_radii2D[visibility],
                    render_pkg["radii"][visibility],
                )
                self.GS.gaussians.add_densification_stats(
                    render_pkg["viewspace_points"],
                    visibility,
                )

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii=None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and (
                self.gs_step % self.training_config.densification_interval == 0
            ):
                size_threshold = (
                    20
                    if self.gs_step > self.training_config.opacity_reset_interval
                    else None
                )
                self.GS.gaussians.densify_and_prune(
                    self.training_config.densify_grad_threshold,
                    0.005,
                    self.GS.scene.cameras_extent,
                    size_threshold,
                    radii,
                )
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                self.dataset_white_background
                and self.gs_step == self.training_config.densify_from_iter
            ):
                self.GS.gaussians.reset_opacity()

    def save_model(self):
        print("\n[ITER {}] Saving Gaussians".format(self.gs_step))
        self.scene.save(self.gs_step)
        print("\n[ITER {}] Saving Checkpoint".format(self.gs_step))
        torch.save(
            (self.GS.gaussians.capture(), self.gs_step),
            self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth",
        )

    def init_with_corr(
        self,
        cfg,
        verbose=False,
        roma_model=None,
        mvroma_model=None,
        mvroma_prematcher=None,
    ):
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optional preinitialized RoMa model.
            mvroma_model: optional preinitialized MV-RoMa model.
            mvroma_prematcher: optional preinitialized UFM prematcher.
        """
        if not cfg.use:
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        backend = str(getattr(cfg, "backend", "roma")).lower()
        if backend == "roma":
            # Keep the original dependency lazy so selecting MV-RoMa does not
            # import or allocate the pairwise RoMa stack.
            from source.corr_init import (
                init_gaussians_with_corr,
                init_gaussians_with_corr_fast,
            )

            init_fn = (
                init_gaussians_with_corr_fast
                if cfg.nns_per_ref == 1
                else init_gaussians_with_corr
            )
            camera_set, selected_indices, visualization_dict = init_fn(
                self.GS.gaussians,
                self.scene,
                cfg,
                self.device,
                verbose=verbose,
                roma_model=roma_model,
            )
        elif backend == "mvroma":
            from source.correspondence import init_gaussians_with_mvroma
            from source.correspondence.config import MVRoMaTrainingSettings
            from source.correspondence.contracts import MVRoMaInitializationResult
            from source.correspondence.manifest import write_pgsr_neighbors
            from source.correspondence.pgsr_graph import build_hybrid_neighbor_map

            initialization = init_gaussians_with_mvroma(
                self.GS.gaussians,
                self.scene,
                cfg,
                self.device,
                verbose=verbose,
                model=mvroma_model,
                prematcher=mvroma_prematcher,
            )
            camera_set, selected_indices, visualization_dict = initialization
            if isinstance(initialization, MVRoMaInitializationResult):
                self._mvroma_context = initialization
                training_settings = MVRoMaTrainingSettings.from_config(cfg.mvroma)
                if (
                    training_settings.pgsr_neighbor_strategy == "hybrid"
                    and self.loss_composer.enabled
                ):
                    self.neighbor_map = build_hybrid_neighbor_map(
                        camera_set,
                        self.neighbor_map,
                        initialization.overlap_matrix,
                        initialization.pair_quality,
                        training_settings,
                    )
                    model_path = getattr(self.scene, "model_path", None)
                    if model_path is not None:
                        write_pgsr_neighbors(model_path, self.neighbor_map)
        else:
            raise ValueError(
                f"Unknown correspondence backend {backend!r}; "
                "expected 'roma' or 'mvroma'"
            )

        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(
                    self.gaussians._xyz.shape[0],
                    dtype=self.gaussians._xyz.dtype,
                    device=self.gaussians._xyz.device,
                )
                mask = torch.concat(
                    [
                        torch.ones(
                            N_splats_at_init,
                            dtype=torch.bool,
                            device=self.gaussians._xyz.device,
                        ),
                        torch.zeros(
                            N_splats_after_init - N_splats_at_init,
                            dtype=torch.bool,
                            device=self.gaussians._xyz.device,
                        ),
                    ],
                    axis=0,
                )
                self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians = self.gaussians
            gaussians._scaling = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(gaussians._scaling) * 0.5
            )
        return visualization_dict

    def configure_correspondence_resume(self, cfg):
        """Restore MV-only training context after skipping initialization."""

        if not cfg.use or str(getattr(cfg, "backend", "roma")).lower() != "mvroma":
            return
        from source.correspondence.config import MVRoMaTrainingSettings

        settings = MVRoMaTrainingSettings.from_config(cfg.mvroma)
        if settings.pgsr_neighbor_strategy != "hybrid":
            return
        if not self.loss_composer.enabled:
            return
        from source.correspondence.manifest import load_pgsr_neighbors

        self.neighbor_map = load_pgsr_neighbors(
            self.scene.model_path,
            self.scene.getTrainCameras(),
        )

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None
