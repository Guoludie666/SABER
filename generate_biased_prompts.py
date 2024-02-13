'''beam search for prompts that can probe the bias'''

import time
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers import BertTokenizer,BertForMaskedLM
from transformers import RobertaTokenizer,RobertaForMaskedLM
from transformers import AlbertTokenizer,AlbertForMaskedLM
import math
import torch.nn.functional as F
import pickle
import tqdm


import os



def get_pairs():
    pairs=[]
    with open('data/train_data/data_onemask/XNTmask_data.pk', 'rb') as file1:
        XTmask_data = pickle.load(file1)
    tar1_maskwords=XTmask_data["male"]["sent"][:]
    tar2_maskwords=XTmask_data["female"]["sent"][:]
    with open('data/train_data/data_onemask/XNT_data.pk', 'rb') as file:
        XT_data = pickle.load(file)
    tar1_words=XT_data["male"]["sent"][:]
    tar2_words=XT_data["female"]["sent"][:]

    for index,line in enumerate(tar1_maskwords):
        line1=line.strip()
        line2=tar2_maskwords[index].strip()
        line1_=tar1_words[index].strip()
        line2_=tar2_words[index].strip()

        pairs.append((line1,line2,line1_,line2_))
    return pairs
 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender','race','religion'],
    help="Choose from ['gender','race','religion']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="Choose from ['bert','roberta','albert']",
)


parser.add_argument(
    "--K",
    default=0.5,
    type=int,
    help="top K-ratio prompts to be selected in the beam search",
)



def send_to_cuda(tar1_tokenized,tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized,tar2_tokenized

#change
def get_tokenized_ith_generate_prompt(tar1_words,tar2_words,tokenizer):
    tar1_sen=[]
    tar2_sen=[]
    for index,i in enumerate(tar1_words):
        tar1_sen.append(i)
        tar2_sen.append(tar2_words[index])

    tar1_tokenized = tokenizer(tar1_sen,padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen,padding=True, truncation=True, return_tensors="pt")

    tar1_mask_index = np.where(tar1_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    tar2_mask_index = np.where(tar2_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    assert tar1_mask_index.shape[0]==tar1_tokenized['input_ids'].shape[0]
    return tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index,tar1_sen


def run_model(model,inputs,mask_index,ster_words):
    predictions = model(**inputs)
    predictions_logits = predictions.logits[np.arange(inputs['input_ids'].size(0)), mask_index][:,ster_words]
    return predictions_logits


def get_JSD(tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index,model,ster_words):
    jsd_list=[]
    tar1_tokenized,tar2_tokenized = send_to_cuda(tar1_tokenized,tar2_tokenized)
    for k in range(tar1_tokenized['input_ids'].shape[0]//args.BS + 1):
        tar1_inputs={}
        tar2_inputs={}
        try:  
            for key in tar1_tokenized.keys():
                tar1_inputs[key]=tar1_tokenized[key][args.BS*k:args.BS*(k+1)]
                tar2_inputs[key]=tar2_tokenized[key][args.BS*k:args.BS*(k+1)]
                            
            tar1_local_mask_index = tar1_mask_index[args.BS*k:args.BS*(k+1)]
            tar2_local_mask_index = tar2_mask_index[args.BS*k:args.BS*(k+1)]
        except IndexError:
            for key in tar1_tokenized.keys():
                tar1_inputs[key]=tar1_tokenized[key][args.BS*(k+1):]
                tar2_inputs[key]=tar2_tokenized[key][args.BS*(k+1):]
                            
            tar1_local_mask_index = tar1_mask_index[args.BS*(k+1):]
            tar2_local_mask_index = tar2_mask_index[args.BS*(k+1):]

        if tar1_inputs['input_ids'].size(0)>0:
            tar1_predictions_logits = run_model(model,tar1_inputs,tar1_local_mask_index,ster_words)
            tar2_predictions_logits = run_model(model,tar2_inputs,tar2_local_mask_index,ster_words)
    
            jsd = jsd_model(tar1_predictions_logits,tar2_predictions_logits)
            jsd_np = jsd.detach().cpu().numpy()
            jsd_np = np.sum(jsd_np,axis=1)
            jsd_list += list(jsd_np)
            del tar1_predictions_logits, tar2_predictions_logits, jsd
    return jsd_list 


def get_prompt_jsd(tar1_words, tar2_words, model, ster_words):
    current_prompts=[]
    jsd_word_list = []
    assert len(tar1_words)==len(tar2_words)
    tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index,tar1_sen= get_tokenized_ith_generate_prompt(tar1_words,tar2_words,tokenizer)
    print("tokenized input shape",tar1_tokenized['input_ids'].shape)        
    jsd_list = get_JSD(tar1_tokenized,tar2_tokenized, tar1_mask_index, tar2_mask_index, model, ster_words)
    jsd_word_list.append(jsd_list)
    current_prompts.extend(tar1_sen)
    jsd_word_list = np.array(jsd_word_list)
    # assert jsd_word_list.shape == (len(tar1_words),len(prompts))    
    return current_prompts,np.mean(jsd_word_list, axis=0)            


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForMaskedLM.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()
    
    jsd_model = JSD(reduction='none')

    tar_pairs=get_pairs()
    male_words=[]
    female_words=[]
    male_words_=[]
    female_words_=[]
    for i in tar_pairs:
        male_words.append(i[0])
        female_words.append(i[1])
        male_words_.append(i[2])
        female_words_.append(i[3])

    ster_words_ = clean_word_list(load_word_list("data/stereotype.txt"),tokenizer)
    ster_words = tokenizer.convert_tokens_to_ids(ster_words_)   #stereotype words   

    ff=open('data/prompts_{}_{}'.format(args.model_name_or_path,args.debias_type),'w')
    if args.debias_type == 'gender':
        current_prompts,current_prompts_jsd = get_prompt_jsd(male_words, female_words, model,ster_words)
    current_prompts=np.array(current_prompts)
    index_lists=np.argsort(current_prompts_jsd)[::-1][:math.ceil(args.K*len(current_prompts))]
    tar_sen={"male":{"sent":[]},"female":{"sent":[]}}
    tar_masksen={"male":{"sent":[]},"female":{"sent":[]}}
    top_k_prompts = np.array(current_prompts)[index_lists]
    for index in index_lists:
        tar_masksen["male"]["sent"].append(male_words[index])
        tar_masksen["female"]["sent"].append(female_words[index])
        tar_sen["male"]["sent"].append(male_words_[index])
        tar_sen["female"]["sent"].append(female_words_[index])
    with open('data/train_data/bias_data/XNT_data.pk', 'wb') as f,open('data/train_data/bias_data/XNTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)  
    for p in top_k_prompts:
        ff.write(p)
        ff.write("\n")     
    print("search space size:",len(current_prompts))
    ff.close()