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
        self.train_pkl = os.path.join(data_dir, "train.pkl")
        self.val_pkl = os.path.join(data_dir, "val.pkl")
        self._ner_mapping = {'O': 0, 'Medicine': 1, 'MedicalCondition': 2, 'Pathogen': 3}
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
                            names=sorted(list(self._ner_mapping.keys()))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def get_tags_tokens(self, sent, anns):
        sent = re.sub("\[\d*\]", "", sent)
        nopunc = [char for char in sent if char not in string.punctuation]
        cleared = ''.join(nopunc)
        value_tag = {ann["value"]: ann['tag_name'] for ann in anns}
        tags = []
        tokens = []
        print(cleared)
        for token in cleared.split(" "):
            tag = value_tag.get(token, "O")
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