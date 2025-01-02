from dataclasses import asdict, dataclass


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    kl_weight: float = 1e-3
    gan_weight: float = 1e-3
    gan_loss_start: int = 2
    num_visualization: int = 1
    lr: float = 3e-4
    grad_clip: float = 0.0

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
