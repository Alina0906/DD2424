from transformers import (ViTImageProcessor, ViTForImageClassification, 
    AutoImageProcessor, AutoModelForImageClassification,
    Trainer, TrainingArguments)
import torch
import torch_pruning as tp
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from cfg import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(config: Config):
    model = ViTForImageClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels if config.task == "class" else 2,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    ).to(device)

    processor = ViTImageProcessor.from_pretrained(config.model_name)

    if config.use_lora:
        model = AutoModelForImageClassification.from_pretrained(
            config.model_name,
            #label2id=label2id,
            #id2label=id2label,
            ignore_mismatched_sizes=True,  
        )
        processor = AutoImageProcessor.from_pretrained(config.model_name)

        peft_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            inference_mode=False
        )
        model = get_peft_model(model, peft_cfg)

    if config.prune_amount > 0:
        DG = tp.DependencyGraph()
        example_inputs = {"pixel_values": torch.randn(1, 3, config.img_size, config.img_size)}
        DG.build_dependency(model, example_inputs=example_inputs)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features > 100:
                strategy = tp.strategy.L1Strategy()
                prune_index = strategy(module.weight, amount=config.prune_amount)
                DG.prune_linear(module, prune_index)

    if config.quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    return model, processor
