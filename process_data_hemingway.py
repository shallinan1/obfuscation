import torch
import numpy as np
import random
from utils import clean_text, save_to_jsonl
from nltk import sent_tokenize

data = torch.load("/gscratch/xlab/jrfish/Authorship_Obfuscation_Controllable_Decoding/data/hemingway/hemingway_novels")
random.seed(1)
train_p = .6
val_p = .2
test_p = .2
n_sent_paragraph = 5

train_data = []
val_data = []
test_data = []
for k in list(data.keys()):
    full_text = data[k]['text']
    # Do not include paragraphs that are too small
    split_text = [clean_text(text.replace("\n", "")) for text in full_text if len(text)>20]
    split_text = [sent_tokenize(s) for s in split_text]
    random.shuffle(split_text)
    num_train = int(np.floor(len(split_text)*train_p))
    num_val = int(np.floor(len(split_text)*val_p))
    num_test = int(np.floor(len(split_text)*test_p))
    [train_data.append(s) for s in split_text[0:num_train]]
    [val_data.append(s) for s in split_text[num_train:num_val+num_train]]
    [test_data.append(s) for s in split_text[-num_test:]]

save_to_jsonl(train_data, "data/train_para_heming.jsonl")
save_to_jsonl(val_data, "data/val_para_heming.jsonl")
save_to_jsonl(test_data, "data/test_para_heming.jsonl")