import os.path
from functools import partial

from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def align_labels_and_tokens(word_ids, labels):
    """ Aligns tokens and their respective labels

    Args:
        word_ids (list): word ids of tokens after subword tokenization.
        labels (list): original labels irrespective of subword tokenization.

    Returns:
        updated_labels (list): labels aligned with respective tokens.

    """
    updated_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            updated_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:
            updated_labels.append(-100)
        else:
            label = labels[word_id]
            # B-XXX to I-XXX for subwords (Inner entities)
            if label % 2 == 1:
                label += 1
            updated_labels.append(label)
    return updated_labels


def prepare_data(tokenizer):
    raw_dataset = load_dataset("conll2003", cache_dir=os.path.join(root, "cached_data"))
    ner_labels, id2label, label2id = get_ner_info(raw_dataset)
    print_ds_statistics(raw_dataset)
    # tokenize
    test_tokenizer(raw_dataset, tokenizer)
    tokenized_dataset = raw_dataset.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer),
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    return tokenized_dataset, data_collator, ner_labels, id2label, label2id


def tokenize_and_align_labels(dataset, tokenizer):
    """ Performs tokenization and aligns all tokens and labels
        in the dataset.

    Args:
        dataset (DatasetDict): dataset containing tokens and labels.

    Returns:
        tokenized_data (dict): contains input_ids, attention_mask, token_type_ids, labels

    """

    tokenized_data = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)
    all_labels = dataset["ner_tags"]
    updated_labels = []
    for i, labels in enumerate(all_labels):
        updated_labels.append(align_labels_and_tokens(tokenized_data.word_ids(i), labels))
    tokenized_data["labels"] = updated_labels
    return tokenized_data


def test_tokenizer(raw_dataset, tokenizer):
    sample_input = tokenizer(raw_dataset["train"][0]["tokens"], is_split_into_words=True)
    print("Start tokenizing")
    print("Tokens: ", sample_input.tokens())
    print("Word Ids: ", sample_input.word_ids())


def print_ds_statistics(raw_dataset):
    print("Upper DS struct:")
    print(raw_dataset)
    print("Dataset struct")
    print("Tokens: ", raw_dataset["train"][0]["tokens"])
    print("Stats: ", raw_dataset["train"][0]["ner_tags"])
    dataset_feature = raw_dataset["train"].features
    print("Ner tags: ", dataset_feature["ner_tags"])
    ner_labels, id2label, label2id = get_ner_info(raw_dataset)
    print("Ner labels: ", ner_labels)


def get_ner_info(raw_dataset):
    dataset_feature = raw_dataset["train"].features
    ner_labels = dataset_feature["ner_tags"].feature.names
    id2label = {str(i): label for i, label in enumerate(ner_labels)}
    label2id = {value: key for key, value in id2label.items()}
    return ner_labels, id2label, label2id