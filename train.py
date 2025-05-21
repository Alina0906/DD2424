import torch
import transformers
from transformers import TrainingArguments, Trainer
from data_generation import load_data, train_test_shuffle, vit_transforms, VitDataset

from models import build_model
from utils import collate_fn, compute_metrics, parse_args


def main():
    config = parse_args()
    transforms = vit_transforms(cfg)
    
    train = load_data('trainval')
    test = load_data('test')
    
    train, test = train_test_shuffle(train, test)

    train_ds = VitDataset(train, transform=transforms, task=config.task)
    eval_ds = VitDataset(test, transform=transforms, task=config.task)

    model, processor = build_model(config, train_ds.label2id, train_ds.id2label)

    training_args = TrainingArguments(
        output_dir="./outputs",
        remove_unused_columns=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,  
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
