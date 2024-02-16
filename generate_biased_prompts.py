import argparse
import torch
import numpy as np
import torch.nn.functional as F
from utils import *
from transformers import BertTokenizer,BertForMaskedLM
from transformers import RobertaTokenizer,RobertaForMaskedLM
from transformers import AlbertTokenizer,AlbertForMaskedLM
import math
import torch.nn.functional as F
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender','race','religion'],
    help="Choose from ['gender','race','religion']",
)
parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="Choose from ['bert','albert','roberta',]",
)
parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--BS",
    default=50,
    type=int,
    help="batch size of the data fed into the model",
)
parser.add_argument(
    "--K",
    default=0.55,
    type=int,
    help="top K-ratio prompts to be selected",
)

def get_pairs(debias_type):
    pairs=[]
    with open('data/train_data/one_mask/XNT_data.pk', 'rb') as file:
        XNT_data = pickle.load(file)
    keys = list(XNT_data.keys())
    tar1_words=XNT_data[keys[0]]["sent"][:]
    tar2_words=XNT_data[keys[1]]["sent"][:]
    with open('data/train_data/one_mask/XNTmask_data.pk', 'rb') as file1:
        XNTmask_data = pickle.load(file1)
    tar1_maskwords=XNTmask_data[keys[0]]["sent"][:]
    tar2_maskwords=XNTmask_data[keys[1]]["sent"][:]
    if debias_type=='religion':
        tar3_words=XNT_data[keys[2]]["sent"][:]
        tar3_maskwords=XNTmask_data[keys[2]]["sent"][:]
    for index,line in enumerate(tar1_maskwords):
        line1=line.strip()
        line2=tar2_maskwords[index].strip()
        line1_=tar1_words[index].strip()
        line2_=tar2_words[index].strip()
        if debias_type=='religion':
            line3=tar3_maskwords[index].strip()
            line3_=tar3_words[index].strip()
            pairs.append((line1,line2,line1_,line2_,line3,line3_))
        else:
            pairs.append((line1,line2,line1_,line2_))
    return pairs
 
def send_to_cuda(tar1_tokenized,tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized,tar2_tokenized

def get_tokenized_ith_generate_prompt(tar1_words,tar2_words,tar3_words,tokenizer):
    tar1_sen=[]
    tar2_sen=[]
    tar3_sen=[]
    for index,i in enumerate(tar1_words):
        tar1_sen.append(i)
        tar2_sen.append(tar2_words[index])
        if tar3_words!=[]:
            tar3_sen.append(tar3_words[index])

    tar1_tokenized = tokenizer(tar1_sen,padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen,padding=True, truncation=True, return_tensors="pt")

    tar1_mask_index = np.where(tar1_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    tar2_mask_index = np.where(tar2_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    if tar3_words!=[]:
        tar3_tokenized = tokenizer(tar3_sen,padding=True, truncation=True, return_tensors="pt")
        tar3_mask_index = np.where(tar3_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]

    assert tar1_mask_index.shape[0]==tar1_tokenized['input_ids'].shape[0]
    if tar3_words!=[]:
        return tar1_tokenized,tar2_tokenized,tar3_tokenized,tar1_mask_index,tar2_mask_index,tar3_mask_index,tar1_sen
    else:
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


def get_prompt_jsd(tar1_words, tar2_words,tar3_words, model, ster_words):
    current_prompts=[]
    jsd_word_list = []
    assert len(tar1_words)==len(tar2_words)
    if tar3_words!=[]:
        assert len(tar1_words)==len(tar2_words)==len(tar3_words)
        tar1_tokenized,tar2_tokenized,tar3_tokenized,tar1_mask_index,tar2_mask_index,tar3_mask_index,tar1_sen=get_tokenized_ith_generate_prompt(tar1_words,tar2_words,tar3_words,tokenizer)
    else:
        tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index,tar1_sen= get_tokenized_ith_generate_prompt(tar1_words,tar2_words,tar3_words,tokenizer)

    print("tokenized input shape",tar1_tokenized['input_ids'].shape)        
    jsd_list = get_JSD(tar1_tokenized,tar2_tokenized, tar1_mask_index, tar2_mask_index, model, ster_words)
    if tar3_words!=[]:
        jsd_list_ = get_JSD(tar1_tokenized,tar3_tokenized, tar1_mask_index, tar3_mask_index, model, ster_words)
        for index in range(len(jsd_list)):
            jsd_list[index]=jsd_list[index]+jsd_list_[index]
    jsd_word_list.append(jsd_list)    
    current_prompts.extend(tar1_sen)
    jsd_word_list = np.array(jsd_word_list)
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

    tar_pairs=get_pairs(args.debias_type)
    
    attribute0_words=[]
    attribute1_words=[]
    attribute2_words=[]
    attribute0_words_=[]
    attribute1_words_=[]
    attribute2_words_=[]
    for i in tar_pairs:
        attribute0_words.append(i[0])
        attribute1_words.append(i[1])
        attribute0_words_.append(i[2])
        attribute1_words_.append(i[3])
        if args.debias_type=='religion':
            attribute2_words.append(i[4])
            attribute2_words_.append(i[5])

    ster_words_ = clean_word_list(load_word_list("data/stereotype.txt"),tokenizer)
    ster_words = tokenizer.convert_tokens_to_ids(ster_words_)   #stereotype words   

    ff=open('data/train_data/bias_data/prompts_{}_{}'.format(args.model_name_or_path,args.debias_type),'w')
    current_prompts,current_prompts_jsd = get_prompt_jsd(attribute0_words, attribute1_words, attribute2_words,model,ster_words)
    current_prompts=np.array(current_prompts)
    index_lists=np.argsort(current_prompts_jsd)[::-1][:math.ceil(args.K*len(current_prompts))]
    print(len(index_lists))
    if args.debias_type=='religion':
        tar_sen={"attribute0":{"sent":[]},"attribute1":{"sent":[]},"attribute2":{"sent":[]}}
        tar_masksen={"attribute0":{"sent":[]},"attribute1":{"sent":[]},"attribute2":{"sent":[]}}
    else:
        tar_sen={"attribute0":{"sent":[]},"attribute1":{"sent":[]}}
        tar_masksen={"attribute0":{"sent":[]},"attribute1":{"sent":[]}}
    top_k_prompts = np.array(current_prompts)[index_lists]
    for index in index_lists:
        tar_masksen["attribute0"]["sent"].append(attribute0_words[index])
        tar_masksen["attribute1"]["sent"].append(attribute1_words[index])
        tar_sen["attribute0"]["sent"].append(attribute0_words_[index])
        tar_sen["attribute1"]["sent"].append(attribute1_words_[index])
        if args.debias_type=='religion':
            tar_masksen["attribute2"]["sent"].append(attribute2_words[index])
            tar_sen["attribute2"]["sent"].append(attribute2_words_[index])
    with open('data/train_data/bias_data/XNT_data.pk', 'wb') as f,open('data/train_data/bias_data/XNTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)  
    for p in top_k_prompts:
        ff.write(p)
        ff.write("\n")     
    print("search space size:",len(current_prompts))
    ff.close()
