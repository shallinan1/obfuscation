import json
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Iterable, Dict, Any
import torch
from IPython import embed
import json

# TODO Un-hardcode these?
with open("models/styledict.json") as f:
    rev_clf_label_map = json.load(f)
clf_label_map = {v: k for k, v in rev_clf_label_map.items()}

def convert_label_list(list_of_dics, 
                       cur_key='label', 
                       cur_val='score', 
                       label_map=None):
    output = {}
    for item in list_of_dics:
        output[label_map[int(item[cur_key].split("_")[1])]] = item[cur_val]
    return output

class StylePredictor:
    def __init__(self,
                 style_model_path: str, 
                 batch_size: int, 
                 device: int):
        self.batch_size = batch_size
        style_model = AutoModelForSequenceClassification.from_pretrained(style_model_path).to(device)
        style_model_tokenizer = AutoTokenizer.from_pretrained(style_model_path)
        self.style_clf = pipeline('text-classification', model=style_model, tokenizer=style_model_tokenizer, top_k=None, function_to_apply='sigmoid', device=device)

    def get_style_predictions(self, 
                         inputs: List[str],
        ) -> List[float]:
            
            # Get the style predictions with the pipeline
            # TODO add tqdm and batch size
            with torch.no_grad():
                style_predictions = self.style_clf(inputs)                

            full_predictions = []
            for style_preds in style_predictions:
                cur_dict = {}
                for style_pred in style_preds:
                    cur_idx = style_pred['label'].split("_")[-1]
                    cur_dict[clf_label_map[int(cur_idx)]] = style_pred['score']
                full_predictions.append(cur_dict)

            # Final prediction
            predictions = [max(f, key = f.get) for f in full_predictions]
            prediction_scores = [f[p] for f, p in zip(full_predictions, predictions)]

            return full_predictions, predictions, prediction_scores
            # clean_style_predictions = [convert_label_list(s ,label_map =clf_label_map) for s in style_predictions]
                

if __name__ == "__main__":
    # Small example of using reward on some generations to get labels
    inputs = [
        "Hey, where you going", 
        "Where art thou going?", 
        "He is a very bad man.",
        "I'm excited to announce a new store."
        ]
    batch_size = 4
    device = 0
    style_model_path = "/gscratch/xlab/hallisky/obfuscation/models/04-08-2024_12:39:35/checkpoint-80"
    
    style_predictor = StylePredictor(
        style_model_path=style_model_path, 
        batch_size=batch_size, 
        device=device)
    full_predictions, predictions, prediction_scores = style_predictor.get_style_predictions(inputs)
    embed()