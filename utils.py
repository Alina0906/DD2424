import argparse
from cfg import Config
from evaluate import load
import numpy as np
import torch


def add_arg(parser, field_name, field_def):
    typ = field_def.type
    default = field_def.default

    if typ is bool:
        parser.add_argument(f"--{field_name}", action="store_true", help=f"enable {field_name}")
    else:
        parser.add_argument(f"--{field_name}", type=typ, default=default)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    
    for field_name, field_def in Config.__dataclass_fields__.items():
        add_arg(parser, field_name, field_def)

    args = parser.parse_args()
    
    return Config(**vars(args))


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    metric = load("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


