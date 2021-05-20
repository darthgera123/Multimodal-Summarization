import collections
import os
import datasets

from param import args

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from fts_tsv.summ_data import HMTorchDataset
# from entryU import ModelU
# from entryO import ModelO
# from entryD import ModelD
# from entryX import ModelX
# from entryV import ModelV
# from entryD import ModelD
from oscar import ModelO
# from src.vilio.transformers.modeling_encoder_decoder import EncoderDecoderModel
# from src.vilio.transformers.modeling_gpt2 import GPT2Model,GPT2LMHeadModel,GPT2Config
from transformers import EncoderDecoderModel
from src.vilio.transformers.tokenization_bert import BertTokenizer
from src.vilio.modeling_bertO import BertO
# from src.vilio.modeling_bertD import BertD, BertConfig

from param import args
from src.vilio.transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.vilio.transformers.trainer import Trainer
from transformers import TrainingArguments
from transformers.tokenization_auto import AutoTokenizer





def is_backbone(n):
    if "encoder" in n:
        return True
    elif "embeddings" in n:
        return True
    elif "pooler" in n:
        return True
    print("F: ", n)
    return False

no_decay = ['bias', 'LayerNorm.weight']

# params = list(model.named_parameters())
# if args.reg:
#     optimizer_grouped_parameters = [
#         {"params": [p for n, p in params if is_backbone(n)], "lr": args.lr},
#         {"params": [p for n, p in params if not is_backbone(n)], "lr": args.lr * 500},
#     ]

#     for n, p in model.named_parameters():
#         print(n)

#     optim = AdamW(optimizer_grouped_parameters, lr=args.lr)
# else:
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
#         {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]

#     optim = AdamW(optimizer_grouped_parameters, lr=args.lr)

# if args.train is not None:
#     scheduler = get_linear_schedule_with_warmup(optim, t_total * 0.1, t_total)

# output = args.output
# os.makedirs(output, exist_ok=True)
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in tnrange(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated
if __name__ == '__main__':
    # loader = train_tuple.loader
    # ids, feats, boxes, sent, title, target = next(iter(loader))
    # feats, boxes, target = feats.cuda(0), boxes.cuda(0), target.long().cuda(0)
    # slogit,logit,dec_output = model(sent,title, (feats, boxes))
    # print(slogit.shape)
    # print(logit.shape)
    # print(dec_output)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # title = tokenizer.batch_decode(dec_output)
    # print(title)
    tset = HMTorchDataset('data/dev_data.jsonl','data/img','data/hm_vgattr3636.tsv',
                            args.tr,128,32)
    # test_set = HMTorchDataset('data/dev_data.jsonl','data/img','data/hm_vgattr3636.tsv',args.tr,
    #                             128,train=False)
    train_loader = DataLoader(
        tset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        drop_last=False, pin_memory=True
    )
    # test_loader = DataLoader(
    #     test_set, batch_size=args.batch_size,
    #     shuffle=False, num_workers=args.num_workers,
    #     drop_last=False, pin_memory=True
    # )
    
    model = ModelO(args)
    # if args.loadpre is not None:
    #     model.load(args.loadpre)
    model = model.cuda(0)
    loader = train_loader
    ids, feats, boxes, sent,title= next(iter(loader))
    feats, boxes= feats.cuda(0), boxes.cuda(0)
    tokenizer = AutoTokenizer.from_pretrained(args.tr)
    context = []
    context = torch.tensor(context, dtype=torch.long).cuda()
    context = context.unsqueeze(0)
    generated = context
    temperature=0.5
    top_k=20
    top_p=0.8
    with torch.no_grad():  
        for _ in range(32):
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            loss,logits,past_tokens = model(sent,title, (feats, boxes))
            next_token_logits = logits[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    
    text = tokenizer.decode(generated[0])
    print(text)
    exit()

    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        tokenizer = AutoTokenizer.from_pretrained(args.tr)
        print(pred_ids)
        print(labels_ids)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    training_args = TrainingArguments(
        evaluate_during_training=False,
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=1000000,
        num_train_epochs=10,
        output_dir='./',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = tset,
        eval_dataset = tset,
        compute_metrics=compute_metrics
        )
    trainer.train()

    # Generate giving wrong outputs for some reason. Need help
    # ids, feats, boxes, sent, = next(iter(test_loader))
    # feats, boxes = feats.cuda(0), boxes.cuda(0)

    # # if args.train is not None:
    # #     batch_per_epoch = len(train_loader)
    # #     t_total = int(batch_per_epoch * args.epochs // args.acc)
    # #     print("Total Iters: %d" % t_total)

    # print(sent[0])
    # p = model.generate(sent, (feats, boxes))
    # # print(p.shape)
    # print(p[0])
    

