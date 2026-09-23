from random import randint

import torch

from source.vendor import bootstrap_gaussian_splatting

bootstrap_gaussian_splatting()

from scene import GaussianModel, Scene

from source.data_utils import scene_cameras_train_test_split
from source.renderers import build_renderer


class Warper3DGS(torch.nn.Module):
    def __init__(
        self,
        sh_degree,
        opt,
        pipe,
        dataset,
        viewpoint_stack,
        verbose,
        renderer=None,
        do_train_test_split=True,
    ):
        super(Warper3DGS, self).__init__()
        """
        Init Warper using all the objects necessary for rendering gaussian splats.
        Here we merely link class objects to the objects instantiated outsided the class.
        """
        self.gaussians = GaussianModel(sh_degree)
        self.renderer = build_renderer(renderer)
        self.render = self.renderer.render
        self.gs_config_opt = opt
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = pipe
        self.scene = Scene(dataset, self.gaussians, shuffle=False)
        self.gaussians.tmp_radii = torch.zeros(
            self.gaussians.get_xyz.shape[0],
            dtype=self.gaussians.get_xyz.dtype,
            device=self.gaussians.get_xyz.device,
        )
        if do_train_test_split:
            scene_cameras_train_test_split(self.scene, verbose=verbose)

        self.gaussians.training_setup(opt)
        self.viewpoint_stack = viewpoint_stack
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()

    def forward(self, viewpoint_cam=None, **render_options):
        """
        For a provided camera viewpoint_cam we render gaussians from this viewpoint.
        If no camera provided then we use the self.viewpoint_stack (list of cameras).
        If the latter is empty we reinitialize it using the self.scene object.
        """
        if viewpoint_cam is None:
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = self.viewpoint_stack[
                randint(0, len(self.viewpoint_stack) - 1)
            ]

        render_pkg = self.render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.bg,
            **render_options,
        )
        return render_pkg
