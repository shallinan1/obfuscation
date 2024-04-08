import pickle
from utils import clean_text, save_to_jsonl
from nltk import sent_tokenize
from IPython import embed
import random
from sklearn.model_selection import train_test_split
SEED = 1
import re
"""
Converts the pickle files of AMT into jsonl files, where
each paragraph in the AMT file is split into sentences and saved.

Converts the train.pickle into a train and val split with 0.75 and 0.25 splits
"""

paths = [
    "/gscratch/xlab/jrfish/Authorship_Obfuscation_Controllable_Decoding/data/amt-3/X_train.pickle",
    "/gscratch/xlab/jrfish/Authorship_Obfuscation_Controllable_Decoding/data/amt-3/X_test.pickle"
]
labels = ["train", "test"]

for path, label in zip(paths, labels):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Length of data is {len(data)} for {label}")
    final_data = {}
    for d in data:
        cur_style = d[0].split("/")[1]
        if cur_style not in final_data:
            final_data[cur_style] = []

        d = clean_text(d[-1].strip())
        d = re.sub(r'\n\t', '\n', d)
        d = re.sub(r'\n+', '\n', d)
        paragraphs = d.split("\n")
        for p in paragraphs:
            final_data[cur_style].append(sent_tokenize(p.strip()))

    if label == "train":
        train_ratio = 0.75
        dev_ratio = 0.25
        random.seed(SEED)

        for cur_label in final_data.keys():
            sents = final_data[cur_label]

            train_sents, dev_sents = train_test_split(
                sents, 
                train_size=train_ratio, 
                test_size=dev_ratio, 
                shuffle=True, 
                random_state=SEED)
            save_to_jsonl(train_sents, f"data/train_para_amt-3-{cur_label}.jsonl")
            save_to_jsonl(dev_sents, f"data/val_para_amt-3-{cur_label}.jsonl")
    else:
        for cur_label in final_data.keys():
            sents = final_data[cur_label]
            save_to_jsonl(sents, f"data/test_para_amt-3-{cur_label}.jsonl")
