import os
import pickle
import re
from datasets import GeneratorBasedBuilder, load_dataset
import datasets
from pickle import Unpickler
import pyrootutils
from transformers import AutoTokenizer


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
from src.data import tokenize_and_align_labels


def main():
    dataset_config = {
        "LOADING_SCRIPT_FILES": os.path.join(root, "src/medical_ner_ds.py"),
        "CONFIG_NAME": "clean",
        "DATA_DIR": os.path.join(root, "data"),
        # "CACHE_DIR": os.path.join(root, "cache_crema"),
    }
    ds = load_dataset(
        dataset_config["LOADING_SCRIPT_FILES"],
        dataset_config["CONFIG_NAME"],
        data_dir=dataset_config["DATA_DIR"],
        # cache_dir=dataset_config["CACHE_DIR"]
    )
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=os.path.join(root, "cached_models"))
    # test_tokenizer(ds, tokenizer)
    print(tokenize_and_align_labels(ds['train'][:1], tokenizer))


def test_tokenizer(raw_dataset, tokenizer):
    sample_input = tokenizer(raw_dataset["train"][0]["tokens"], is_split_into_words=True)
    print("Start tokenizing")
    print("Tokens: ", sample_input.tokens())
    print("Word Ids: ", sample_input.word_ids())

if __name__ == '__main__':
    main()
