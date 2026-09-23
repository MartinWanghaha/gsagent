import os
from argparse import Namespace

import hydra
import omegaconf
import wandb
from omegaconf import OmegaConf

from source.trainer import EDGSTrainer
from source.utils_aux import set_seed


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    _ = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=[cfg.wandb.tag],
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
    )
    omegaconf.OmegaConf.resolve(cfg)
    set_seed(cfg.seed)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
    with open(os.path.join(cfg.gs.dataset.model_path, "cfg_args"), "w") as cfg_log_f:
        params = {
            "sh_degree": cfg.gs.sh_degree,
            "source_path": cfg.gs.dataset.source_path,
            "model_path": cfg.gs.dataset.model_path,
            "images": cfg.gs.dataset.images,
            "depths": cfg.gs.dataset.depths,
            "resolution": cfg.gs.dataset.resolution,
            "white_background": cfg.gs.dataset.white_background,
            "train_test_exp": cfg.gs.dataset.train_test_exp,
            "data_device": cfg.gs.dataset.data_device,
            "eval": cfg.gs.dataset.eval,
            "convert_SHs_python": cfg.gs.pipe.convert_SHs_python,
            "compute_cov3D_python": cfg.gs.pipe.compute_cov3D_python,
            "debug": cfg.gs.pipe.debug,
            "antialiasing": cfg.gs.pipe.antialiasing,
        }
        cfg_log_f.write(str(Namespace(**params)))
    OmegaConf.save(cfg, os.path.join(cfg.gs.dataset.model_path, "config.yaml"))

    # Init both agents
    gs = hydra.utils.instantiate(cfg.gs)

    # Init trainer and launch training
    trainer = EDGSTrainer(
        GS=gs,
        training_config=cfg.gs.opt,
        dataset_white_background=cfg.gs.dataset.white_background,
        device=cfg.device,
        log_wandb=cfg.wandb.mode != "disabled",
    )

    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()
    if trainer.gs_step == 0:
        trainer.init_with_corr(cfg.init_wC)
    elif cfg.init_wC.use:
        trainer.configure_correspondence_resume(cfg.init_wC)
        trainer.CONSOLE.print(
            "Skipping correspondence initialization when resuming a 3DGS "
            "checkpoint.",
            style="warning",
        )
    trainer.train(cfg.train)

    # All done
    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
