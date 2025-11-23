import argparse
import os
from functools import partial

import numpy as np
import evaluate
import inspect
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataset import load_dataset
from labels import LABELS
from model import build_model


def compute_metrics(p, label_list=LABELS):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in seq if l != -100] for seq in labels]
    true_preds = [[label_list[p_] for (p_, l) in zip(seq_p, seq_l) if l != -100]
                  for seq_p, seq_l in zip(preds, labels)]

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_preds, references=true_labels)
    # return overall f1
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--train")
    parser.add_argument("--dev")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = load_dataset(tokenizer, args.train)
    dev_ds = load_dataset(tokenizer, args.dev)

    model = build_model(args.model_name, num_labels=len(LABELS))

    data_collator = DataCollatorForTokenClassification(tokenizer)

    ta_init_kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=50,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=False,
    )

    # Filter kwargs to only those accepted by this Transformers version
    sig = inspect.signature(TrainingArguments.__init__)
    valid_kwargs = {k: v for k, v in ta_init_kwargs.items() if k in sig.parameters}
    training_args = TrainingArguments(**valid_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out_dir)


if __name__ == "__main__":
    main()
