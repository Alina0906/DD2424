from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: str = "annotations"
    image_dir: str = "images"
    batch_size: int = 32
    num_workers: int = 4
    img_size: int = 224

    model_name: str = "google/vit-large-patch16-224"
    num_labels: int = 37
    task: str = "class"

    lora_rank: int = 8
    lora_alpha: int = 16
    use_lora: bool = False

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    # logging_steps: int = 50

    prune_amount: float = 0.0
    quantize: bool = False


