import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import normalize
import torch.nn.utils.prune as prune
import transformers
from transformers import ViTForImageClassification, Trainer, TrainingArguments

from data_generation import load_data, train_test_shuffle, vit_transforms, VitDataset
from models import build_model
from utils import collate_fn, compute_metrics, parse_args


config = parse_args()

transforms = vit_transforms()
train = load_data('trainval')
test = load_data('test')

train, test = train_test_shuffle(train, test)

train_ds = VitDataset(train, transform=transforms, task=config.task)
eval_ds = VitDataset(test, transform=transforms, task=config.task)

model, processor = build_model(config, train_ds.label2id, train_ds.id2label)

prune_config = {
    "prune_layers": [
        "vit.encoder.layer.{}.attention.attention.query",
        "vit.encoder.layer.{}.attention.attention.key",
        "vit.encoder.layer.{}.attention.attention.value",
        "vit.encoder.layer.{}.output.dense"
    ],  
    "prune_rate": 0.1,  
    "prune_epoch_interval": 2,  
    "total_prune_steps": 5  
}

class PruningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = 0
        self.attention_maps = {"original": [], "pruned": []}
        
    def apply_pruning(self):
        for layer_idx in range(24):  # ViT-large
            for pattern in prune_config["prune_layers"]:
                layer_name = pattern.format(layer_idx)
                module = self.model
                for part in layer_name.split("."):
                    module = getattr(module, part)

                prune.l1_unstructured(module, name="weight", amount=prune_config["prune_rate"])
                prune.remove(module, "weight")  
                
    def log_sparsity(self):
        total_zeros = 0
        total_params = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "weight"):
                zeros = (module.weight == 0).sum().item()
                total_zeros += zeros
                total_params += module.weight.numel()
        print(f"Global Sparsity: {100*total_zeros/total_params:.2f}%")
        
    def plot_heatmap(self, tensor, title):
        plt.figure(figsize=(10, 8))
        plt.imshow(tensor.cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.savefig(f"heatmap_step{self.current_step}.png")
        plt.close()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % prune_config["prune_epoch_interval"] == 0:
            if self.current_step < prune_config["total_prune_steps"]:
                # Attention Map Before Pruning
                sample = self.train_dataset[0]
                input_tensor = sample['pixel_values']
                
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                    
                with torch.no_grad():
                    outputs = self.model(input_tensor.to(self.args.device))
                    self.attention_maps["original"].append(outputs.attentions[0][0].mean(0)[0])
                    
                # Conduct Pruning
                self.apply_pruning()
                self.log_sparsity()
                
                # Attention Map After Pruning
                with torch.no_grad():
                    outputs = self.model(input_tensor.to(self.args.device))
                    self.attention_maps["pruned"].append(outputs.attentions[0][0].mean(0)[0])
                    
                # Visualization
                original = self.attention_maps["original"][-1]
                pruned = self.attention_maps["pruned"][-1]
                self.plot_heatmap(original, f"Original Attention (Step {self.current_step})")
                self.plot_heatmap(pruned, f"Pruned Attention (Step {self.current_step})")
                self.plot_heatmap(original - pruned, f"Attention Difference (Step {self.current_step})")
                
                self.current_step += 1

training_args = TrainingArguments(
    output_dir="./vit-pruned",
    remove_unused_columns=True,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=prune_config["total_prune_steps"] * prune_config["prune_epoch_interval"] + 2,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    warmup_steps=config.warmup_steps,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,          
    metric_for_best_model="accuracy",
    label_names=["labels"],
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    logging_dir="./logs",
)


trainer = PruningTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=processor,  
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()

def plot_comparison(original, pruned, metric_name):
    plt.plot(original, label="Original")
    plt.plot(pruned, label="Pruned")
    plt.xlabel("Pruning Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f"{metric_name}_comparison.png")
    plt.close()

val_acc = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]
plot_comparison(val_acc[:len(val_acc)//2], val_acc[len(val_acc)//2:], "Validation Accuracy")
