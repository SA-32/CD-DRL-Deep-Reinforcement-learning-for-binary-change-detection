"""
config.py
---------
Hyperparameters, taken directly from the paper's "Implementation details"
subsection.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    in_channels: int = 3
    image_size: tuple = (256, 256)     # set to (320, 320) for Cardiac, (256, 256) for Brain
    batch_size: int = 8                # "the batch size is set to 2" (GPU memory limited)

    # Optimization ("Adam ... learning rates ... 1e-3, drops by a factor of
    # 0.9 every 25 epochs")
    optimizer: str = "adam"
    lr: float = 1e-3
    lr_decay_gamma: float = 0.9
    lr_decay_every: int = 25
    max_epochs: int = 200              # "maximum epoch to 200"

    # RL / PA3C
    t_max: int = 10                    # "length of each episode t_max to 10"
    gamma: float = 0.95                # "discount rate gamma to 0.95"
    entropy_coef: float = 0.01         # standard A3C entropy bonus (not explicitly
                                        # given a value in the paper; kept small)
    value_loss_coef: float = 0.5

    # Model architecture
    use_sam: bool = True
    use_dc: bool = True
    policy_hidden: int = 128
    value_hidden: int = 128
    n_layers: int = 4
    n_actions: int = 2                 # {0: background, 1: do nothing}

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_every: int = 200
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"
