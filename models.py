import torch
import torch.nn as nn
import torch_pruning as tp
from torch.ao.quantization import quantize_dynamic

from transformers import (ViTImageProcessor, ViTForImageClassification, 
    AutoImageProcessor, AutoModelForImageClassification)
from peft import get_peft_model, LoraConfig

from cfg import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prune_model(model, prune_amount=0.2, img_size=224):
    strategy = tp.strategy.L1Strategy()  
    pruning_targets = []
    for name, module in model.named_modules():
        if "attention.attention.query" in name or "attention.attention.value" in name:
            if isinstance(module, torch.nn.Linear):
                pruning_targets.append(module)

    example_inputs = {"pixel_values": torch.randn(1, 3, img_size, img_size)}
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    for module in pruning_targets:
        n_keep = int(module.out_features * (1 - prune_amount))
        prune_indices = strategy(module.weight, n_keep=n_keep)
        plan = DG.get_pruning_plan(module, tp.prune_linear, indices=prune_indices)
        plan.exec()

    return model


def build_model(config, label2id, id2label):  
    """
    num_labels = config.num_labels if config.task == "class" else 2
    model = ViTForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    ).to(device)

    processor = ViTImageProcessor.from_pretrained(config.model_name)
    """
    
    model = AutoModelForImageClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels if config.task == "class" else 2,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
    ).to(device)

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    
    if config.use_lora:
        peft_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)

    if config.prune_amount > 0:  # need to fix
        model = prune_model(model, prune_amount=config.prune_amount, img_size=config.img_size)
        
    if config.quantize:
        model = quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    return model, processor
