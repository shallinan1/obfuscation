import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
os.environ["WANDB_DISABLED"] = "true" 

import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    set_seed
)

from IPython import embed
import random
import math
from datetime import datetime
time = datetime.now()    
date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
import json
from utils import read_from_jsonl
import itertools
import numpy as np

"""
Trains a classifier for the specified styles
# TODO add more description here

CUDA_VISIBLE_DEVICES=0 python3 train_multilabel_obf.py --use_accuracy_for_training --lr 5e-5 --batch_size 128 --seed 0 --epochs 5 --save_ratio 2
"""
def load_data(data_dir, styles):
    data = {}
    for split in ["train", "val"]:
        data[split] = {}

        for s in styles:
            path = os.path.join(data_dir, f"{split}_para_{s}.jsonl") 
            # IMPORTANT: This flattens the list of lists, meaning we are doing sentence classification
            cur_data = list(itertools.chain(*read_from_jsonl(path)))
            
            data[split][s] = cur_data

    return data

def main(args):
    # Set seed before initializing model.
    set_seed(args.seed)
    num_labels = len(args.styles)
    print(f"We have {num_labels} styles")

    if not args.evaluate: # Train model from scratch
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large", num_labels=num_labels, 
            problem_type="multi_label_classification"
        )
    else: # Load existing model for evaluation only
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_path, num_labels=num_labels, 
            problem_type="multi_label_classification"
        )
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    train_data, train_labels = [], []
    dev_data, dev_labels = [], []

    # Make a dict to map style to number and vice versa
    style_dict, rev_style_dict = {}, {}
    for i, style in enumerate(args.styles):
        style_dict[style] = i
        rev_style_dict[i] = style

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Save dictionary with label ids to style and vice versa
    with open(os.path.join(args.output_dir, "styledict.json"), "w") as k:
        k.write(json.dumps(style_dict))
    with open(os.path.join(args.output_dir, "revstyledict.json"), "w") as m:
        m.write(json.dumps(rev_style_dict))

    data = load_data(args.data_folder, args.styles)
    final_data = {}
    for split in data.keys():
        cur_data = data[split]

        min_length = 1e6 # Arbitrarily large value
        for s in cur_data.keys():
            min_length = min(len(cur_data[s]), min_length)
        print(f"Min length for split {split} is {min_length}")

        comb_data, comb_labels = [], []
        for s in cur_data.keys():
            comb_data.extend(random.sample(cur_data[s], min_length))
            comb_labels.extend([style_dict[s]] * min_length)
        final_data[split] = (comb_data, comb_labels)

    train_data, train_labels = final_data["train"]
    dev_data, dev_labels = final_data["val"]
    print(f"Length of train data is {len(train_data)}, length of val data is {len(dev_data)}")

    train_labels = torch.nn.functional.one_hot(torch.tensor(train_labels), num_classes=num_labels).type(torch.float32)
    dev_labels = torch.nn.functional.one_hot(torch.tensor(dev_labels), num_classes=num_labels).type(torch.float32)

    # Shuffle the data
    train_combo = list(zip(train_data, train_labels))
    random.shuffle(train_combo)
    train_data, train_labels = zip(*train_combo)
    train_data, train_labels = list(train_data), list(train_labels)

    dev_combo = list(zip(dev_data, dev_labels))
    random.shuffle(dev_combo)
    dev_data, dev_labels = zip(*dev_combo)
    dev_data, dev_labels = list(dev_data), list(dev_labels)
    
    # Collate function for batching tokenized texts
    def collate_tokenize(data, max_length_tok=args.max_length_tok):
        text_batch = [element["text"] for element in data]
        tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt', max_length=max_length_tok)
        label_batch = [element["label"] for element in data]
        tokenized['labels'] = torch.stack(label_batch)
        
        return tokenized

    class StyleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __getitem__(self, idx):
            item = {}
            item['text'] = self.texts[idx]
            item['label'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = StyleDataset(train_data, train_labels)
    dev_dataset = StyleDataset(dev_data, dev_labels)

    # Number of steps per epoch
    steps = len(train_labels)/(args.batch_size*torch.cuda.device_count())
    # Save every quarter epoch
    save_steps = math.ceil(steps / args.save_ratio)
    print(f"Save steps is {save_steps}")

    # Training branch
    if not args.evaluate:
        compute_metrics= None
        metric_for_best_model = None
        greater_is_better = None

        # If we want to calculate classification accuracy while we're training
        if args.use_accuracy_for_training:
            def accuracy(eval_pred):
                predictions, labels = eval_pred
                predictions = torch.argmax(torch.tensor(predictions), dim=-1).tolist()
                labels = [a[1].item() for a in torch.nonzero(torch.tensor(labels))]
                
                # Initialize dictionaries to track correct and total predictions per class
                correct_predictions_per_class = {}
                total_predictions_per_class = {}
                
                # Loop through all predictions and labels
                for pred, label in zip(predictions, labels):
                    if label not in total_predictions_per_class:
                        total_predictions_per_class[label] = 0
                        correct_predictions_per_class[label] = 0
                    total_predictions_per_class[label] += 1
                    
                    if pred == label:
                        correct_predictions_per_class[label] += 1
                
                # Calculate accuracy for each class
                accuracy_per_class = {rev_style_dict[label] + "_acc": correct / total for label, correct, total in zip(total_predictions_per_class.keys(), correct_predictions_per_class.values(), total_predictions_per_class.values())}
                
                # Calculate overall accuracy
                overall_accuracy = sum(correct_predictions_per_class.values()) / sum(total_predictions_per_class.values())
                assert overall_accuracy == sum([a == b for a, b in zip(predictions, labels)])/len(predictions)

                # Combine overall accuracy with per-class accuracies
                accuracy_results = {'overall_acc': overall_accuracy, 'acc_product': np.prod(list(accuracy_per_class.values())), **accuracy_per_class}
                return accuracy_results
            
            compute_metrics = accuracy
            metric_for_best_model = 'overall_acc'
            greater_is_better = True

        args = TrainingArguments(
            output_dir = os.path.join(args.output_dir,date_time), 
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='steps',
            num_train_epochs=args.epochs,
            eval_steps = save_steps,
            save_steps = save_steps,
            logging_steps = save_steps,
            lr_scheduler_type = 'linear',
            learning_rate=args.lr,
            seed = args.seed,
            warmup_ratio = 0.1,
            load_best_model_at_end = True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            remove_unused_columns=False
            )

        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=dev_dataset, 
            tokenizer=tokenizer,
            data_collator = collate_tokenize,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(5)]
            )

        trainer.train()
    else: # Evaluation only branch
        from torch.utils.data import DataLoader
        dataload = DataLoader(dev_dataset, collate_fn=collate_tokenize, batch_size=args.batch_size)
        truth, pred = [], []
        for d in dataload:
            true_labs = [a[1].item() for a in d["labels"].nonzero()]
            truth.extend(true_labs)
            pred.extend(torch.argmax(model(**d).logits, dim=-1).tolist())

        print(sum([a == b for a, b in zip(truth, pred)])/len(pred))

        embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder', type=str, default="data")
    parser.add_argument(
        '--styles', type=str, nargs="+", default=["amt-3-h", "amt-3-pp", "amt-3-qq", "heming", "obama", "trump"])
    parser.add_argument(
        '--save_ratio', type=int, default=4)
    parser.add_argument(
        '--lr', type=float, default=5e-5)
    parser.add_argument(
        '--epochs', type=int, default=5)
    parser.add_argument(
        '--max_length_tok', type=int, default=128, help="max length of tokenized sequences")
    parser.add_argument(
        '--batch_size', type=int, default=64)
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--output_dir', type=str, default='models/')
    parser.add_argument(
        '--pretrained_path', type=str, default=None)
    parser.add_argument(
        "--evaluate", action="store_true")
    parser.add_argument(
        "--use_accuracy_for_training", action="store_true", help = "If you want eval metric to be accuracy")
    main(parser.parse_args())
