import torch
from torch.utils.data import DataLoader
from hyperparameter import args
from processer.utils import build_dataset, collate_fn
from transformers import BertTokenizer, BertConfig
from models.model import BertCrfForNer
from processer.progressbar import ProgressBar
from processer.classes import get_entities
from torch.utils.data import RandomSampler, DistributedSampler, SequentialSampler
from loguru import logger
import os
from torch import nn
import json


def predict(model, tokenizer, device, prefix=""):
    test_dataset = build_dataset(tokenizer, args.do_eval)
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32,
                                collate_fn=collate_fn)
    # Eval!
    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module

    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": None}
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2tag, args.markup)
        json_d = {}
        json_d['id'] = step
        tokens = tokenizer.convert_ids_to_tokens(batch[0][5][1:-1])
        result = [(token, tag) for token, tag in zip(tokens, [args.id2tag[x] for x in preds])]
        print(result)
        json_d['tag_seq'] = " ".join([args.id2tag[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(None, step)
    print(results)


if __name__ == '__main__':

    num_labels = len(args.id2tag)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(args.checkpoint_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint_dir)
    model = BertCrfForNer.from_pretrained(args.checkpoint_dir, config=config).to(device)
    predict(model, tokenizer, device)
