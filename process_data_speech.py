import torch
import numpy as np
import random
from nltk import sent_tokenize
from utils import clean_text, save_to_jsonl
from IPython import embed

for Name, name in zip(["Trump", "Obama"], ["trump", "obama"]):
    data = torch.load("/gscratch/xlab/jrfish/Authorship_Obfuscation_Controllable_Decoding/data/" + Name + "_data/" + name + "_speeches")
    random.seed(1)
    train_p = .6
    val_p = .2
    test_p = .2
    n_sent_paragraph = 5

    def group_text(text,n_sent_paragraph):
        # Split the text into sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into chunks of n_sent_paragraphj
        grouped_sentences = [sentences[i:i+n_sent_paragraph] for i in range(0, len(sentences), n_sent_paragraph)]
        
        return grouped_sentences

    train_data = []
    val_data = []
    test_data = []
    for k in list(data.keys()):
        full_text = data[k]['text']
        full_text = clean_text(full_text)
        split_text = group_text(full_text, n_sent_paragraph) # artifically make paragraphs by grouping sentences
        random.shuffle(split_text)
        num_train = int(np.floor(len(split_text)*train_p))
        num_val = int(np.floor(len(split_text)*val_p))
        num_test = int(np.floor(len(split_text)*test_p))
        [train_data.append(s) for s in split_text[0:num_train]]
        [val_data.append(s) for s in split_text[num_train:num_val+num_train]]
        [test_data.append(s) for s in split_text[-num_test:]]
    
    # Save train, val, and test to jsonl
    save_to_jsonl(train_data, "train_para_" + name + ".jsonl")
    save_to_jsonl(val_data, "val_para_"  + name + ".jsonl")
    save_to_jsonl(test_data, "test_para_"  + name + ".jsonl")


