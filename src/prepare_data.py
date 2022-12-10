import string

from tqdm import tqdm
import typing

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import re

def train_val_split(df_main):
    X = df_main["content"]
    y = df_main["annotations"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    return X_train, X_test, y_train, y_test

def save(X_train, X_test, y_train, y_test):
    train = X_train.tolist(), y_train.tolist()
    val = X_test.tolist(), y_test.tolist()
    with open("../data/train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("../data/val.pkl", "wb") as f:
        pickle.dump(val, f)

MODE = typing.Literal['s', 'm', 'e']
bedin_prefix = "B-"
inter_prefix = "I-"
end_prefix = "E-"
names_to_tags_map = {
    'Medicine': "MED",
    'MedicalCondition': "MEDC",
    'Pathogen': "PAT"
}
def update_tag(anno: dict, where: MODE, part: str):
    if where == "s":
        anno['tag_name'] = bedin_prefix + names_to_tags_map[anno['tag_name']]
    elif where == "e":
        anno['tag_name'] = end_prefix + names_to_tags_map[anno['tag_name']]
    else:
        anno['tag_name'] = inter_prefix + names_to_tags_map[anno['tag_name']]
    anno["value"] = part

def prepare():
    with open("../medical_data/Corona2.json", "r") as f:
        data = json.load(f)

    examples = data["examples"]
    df_main = pd.DataFrame(examples)
    df_main.head()
    tags_to_id = {"O": 0}
    max_id = 0

    new_anno = []
    for i, sample in tqdm(enumerate(df_main["annotations"])):
        new_sample = []
        for ann in sample:
            val = re.sub("\s+", " ", ann['value']).strip()
            nopunc = [char for char in val if char not in string.punctuation]
            val = ''.join(nopunc)
            if len(val) < 2:
                continue
            parts = val.split(" ")
            for i, part in enumerate(parts):
                new_ann = ann.copy()
                if i == 0:
                    update_tag(new_ann, 's', part)
                elif i == len(parts) - 1:
                    update_tag(new_ann, 'e', part)
                else:
                    update_tag(new_ann, 'm', part)
                new_sample.append(new_ann)
                if new_ann['tag_name'] not in tags_to_id:
                    max_id += 1
                    tags_to_id[new_ann['tag_name']] = max_id
        new_anno.append(new_sample)

    for i, new_sample in enumerate(new_anno):
        df_main["annotations"].at[i] = new_sample

    save(*train_val_split(df_main))
    print(tags_to_id)

if __name__ == '__main__':
    prepare()