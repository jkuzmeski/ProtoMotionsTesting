# ----- BEGIN ISAAC LAB BOILERPLATE (Multi-GPU aware)-----
#
# This must be at the top of the file, before any other imports.
import os

# This dictionary will hold the flags for launching the app.
# We hard-code headless=True since you are on a remote cluster.
app_launcher_flags = {"headless": True}

# Check for environment variables set by a distributed launcher (like SLURM or torchrun)
# to determine if we are in a multi-GPU run.
if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    app_launcher_flags["distributed"] = True

# We only want to import and launch the app if the simulator is Isaac Lab.
# We check sys.argv directly because this needs to happen before Hydra takes over.
import sys
simulator = None
if any("simulator=isaaclab" in arg for arg in sys.argv):
    simulator = "isaaclab"
    from isaaclab.app import AppLauncher

    # Launch the Isaac Lab app with the determined flags.
    app_launcher = AppLauncher(app_launcher_flags)
    simulation_app = app_launcher.app
elif any("simulator=isaacgym" in arg for arg in sys.argv):
    import isaacgym  # This is necessary for Isaac Gym's import-time initialization
    simulator = "isaacgym"

#
# ----- END ISAAC LAB BOILERPLATE -----


os.environ["WANDB_DISABLE_SENTRY"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"

from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate
import wandb
from lightning.pytorch.loggers import WandbLogger
import torch
from lightning.fabric import Fabric
from utils.config_utils import *
from utils.common import seeding
from protomotions.agents.ppo.agent import PPO
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    torch.set_float32_matmul_precision("high")

    save_dir = Path(config.save_dir)
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"
    if pre_existing_checkpoint.exists():
        log.info(f"Found latest checkpoint at {pre_existing_checkpoint}")
        # Load config from checkpoint folder
        if checkpoint_config_path.exists():
            log.info(f"Loading config from {checkpoint_config_path}")
            config = OmegaConf.load(checkpoint_config_path)

        # Set the checkpoint path in the config
        config.checkpoint = pre_existing_checkpoint

    # Fabric should launch AFTER loading the config. This ensures that wandb parameters are loaded correctly for proper experiment resuming.
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if config.seed is not None:
        rank = fabric.global_rank
        if rank is None:
            rank = 0
        fabric.seed_everything(config.seed + rank)
        seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)

    # The app is now launched at the top of the script.
    # We just need to pass the global 'simulation_app' variable if we are using Isaac Lab.
    if simulator == "isaaclab":
        env = instantiate(
            config.env, device=fabric.device, simulation_app=simulation_app
        )
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(config.checkpoint)

    # find out wandb id and save to config.yaml if 1st run:
    # wandb on rank 0
    if fabric.global_rank == 0 and not checkpoint_config_path.exists():
        if "wandb" in config:
            for logger in fabric.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

            # saving config with wandb id for next resumed run
            wandb_id = wandb.run.id
            log.info(f"wandb_id found {wandb_id}")
            unresolved_conf["wandb"]["wandb_id"] = wandb_id

        # only save before 1st run.
        # note, we save unresolved config for easier inference time logic
        log.info(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)

    agent.fabric.strategy.barrier()

    if config.simulator.config.record_viewer:
        agent.env.simulator._start_video_record()

    agent.fit()

    if config.simulator.config.record_viewer:
        agent.env.simulator._stop_video_record()


if __name__ == "__main__":
    main()