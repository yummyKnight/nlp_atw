import os.path
from functools import partial

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.data import prepare_data
from datasets import load_metric
import numpy as np
import torch
from transformers import AutoTokenizer


def compute_metrics(p, metric, ner_labels):
    """ Computes and returns metrics during training.

    Args:
        p (tuple): tuple containing predictions, labels as lists.

    Returns:
        dict: Dictionary containing precision, recall, f1 score, accuracy.

    """

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [ner_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ner_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": metrics["overall_precision"],
        "recall": metrics["overall_recall"],
        "f1": metrics["overall_f1"],
        "accuracy": metrics["overall_accuracy"]
    }


def train(batch_size=32):
    print("Is cuda available:", torch.cuda.is_available())
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=os.path.join(root, "cached_models"))
    tokenized_dataset, data_collator, ner_labels, id2label, label2id = prepare_data(tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(ner_labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=os.path.join(root, "cached_models")
    )
    metric = load_metric("seqeval")

    args = TrainingArguments(
        model_checkpoint,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        save_strategy="no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        weight_decay=0.01,
        report_to="tensorboard"
    )
    partial_compute_metrics = partial(compute_metrics, ner_labels=ner_labels, metric=metric)
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial_compute_metrics,
    )
    trainer.train()
    trainer.save_model("best_model/")


if __name__ == '__main__':
    train(10)
