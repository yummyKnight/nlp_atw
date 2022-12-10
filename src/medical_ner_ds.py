import os.path
import pickle
import re
from datasets import GeneratorBasedBuilder
import datasets
import string


class MedicalNerDataset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="clean", description="Train Set.")]

    def __init__(self, **kwargs) -> None:
        data_dir = kwargs["data_dir"]
        self.train_pkl = os.path.join(data_dir, "../data/train.pkl")
        self.val_pkl = os.path.join(data_dir, "../data/val.pkl")
        self._ner_mapping = {'O': 0, 'B-MED': 1, 'I-MED': 2, 'E-MED': 3, 'B-MEDC': 4, 'I-MEDC': 5, 'E-MEDC': 6,
                             'B-PAT': 7, 'I-PAT': 8, 'E-PAT': 9}
        super().__init__(**kwargs)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.train_pkl}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.val_pkl}),
        ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=list(self._ner_mapping.keys())
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _find_token_by_value(self, ann_values: list, value: str, prev_tag: str, default="O"):
        for i, (ann_v, ann_tag) in enumerate(ann_values):
            if ann_v == value:
                if not ann_tag.startswith("B") and not prev_tag.startswith("B"):
                    continue
                return ann_values.pop(i)[1]
        return default

    def get_tags_tokens(self, sent, anns):
        sent = re.sub("\[\d*\]", "", sent)
        nopunc = [char for char in sent if char not in string.punctuation]
        cleared = ''.join(nopunc)
        ann_values = [(ann["value"], ann['tag_name']) for ann in anns]
        tags = []
        tokens = []
        def_tag = "O"
        prev_tag = def_tag
        for token in cleared.split(" "):
            tag = self._find_token_by_value(ann_values, token, prev_tag, def_tag)
            prev_tag = tag
            tags.append(self._ner_mapping[tag])
            tokens.append(token)
        return tags, tokens

    def _generate_examples(self, filepath):
        with open(filepath, "rb") as f:
            texts, annotations = pickle.load(f)
        guid = -1
        for sent, anns in zip(texts, annotations):
            tags, tokens = self.get_tags_tokens(sent, anns)
            guid += 1
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": tags,
            }
