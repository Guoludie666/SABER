from transformers import BertTokenizer,BertModel
from transformers import RobertaTokenizer,RobertaModel
from transformers import AlbertTokenizer,AlbertModel
import torch
from tqdm import tqdm
import time
import argparse

 
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="Choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender','race','religion'],
    help="Choose from ['gender','race','religion']",
)

parser.add_argument(
    "--BS",
    default=8,
    type=int,
    help="batch size of the data fed into the model",
)

# -*- coding: utf - 8 -*-

import json
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def KNN_split(embeddings_file):
    """
    Calculate similarity using Faiss.
    :return:
    """
    dim = 768
    # The method for calculating similarity METRIC_INNER_PRODUCT => inner product (cosine similarity)
    measure = faiss.METRIC_INNER_PRODUCT
    param = "HNSW64"
    # use a single GPU
    #res = faiss.StandardGpuResources()
    # index = faiss.index_factory(dim, param, measure)
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    embeddings = []

    with open(embeddings_file, "r", encoding='utf8') as reader:
        lines=reader.readlines()
        count=0
        for line in lines[:100000]:
            line.strip()
            parts = line.split(' ')
            assert len(parts) == dim
            v = list(map(lambda x: float(x), parts[0:]))
            embeddings.append(v)

        # faiss index
        index = faiss.index_factory(dim, param, measure)
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index=index
        embeddings = np.array(embeddings, dtype=np.float32)
        gpu_index.train(embeddings)
        print("Training completed")
        gpu_index.add(embeddings)
        #print(gpu_index.ntotal)
        # KNN_6
        D, I = gpu_index.search(embeddings, 6)
        threshold=227
        #ow_indices, col_indices = np.where(D > threshold)
        T_MASK = np.where(D > threshold,I,0)
        T_MASK[:, 0] = 0
        T_MASK_ = np.where(T_MASK > 0,1,0)
        #Count the number of elements in the row that satisfy the condition except themselves
        count_per_row = np.sum(T_MASK_, axis=1)
        #Basis for dividing X_T
        print(np.sum(count_per_row > 0, axis=0),len(lines))
        XT_idx=np.where(count_per_row>0)
        XNT_idx=np.where(count_per_row==0)

        return XT_idx,T_MASK,count_per_row,XNT_idx
#Modify these files(In the case of gender, to take only 100,000 pieces of data for the experiment.)
def save_pk(embeddings_file):
    with open("data/gpt_data/gender/generate_male.txt", "r", encoding='utf8') as f1:
        male_lines=f1.readlines()[:100000]
    with open("data/gpt_data/gender/male.txt", "r", encoding='utf8') as f1_:
        malemask_lines=f1_.readlines()[:100000]
    with open("data/gpt_data/gender/generate_female.txt", "r", encoding='utf8') as f2:
        female_lines=f2.readlines()[:100000]
    with open("data/gpt_data/gender/female.txt", "r", encoding='utf8') as f2_:
        femalemask_lines=f2_.readlines()[:100000]

    XT_data={'male':{},'female':{}}
    XTmask_data={'male':{},'female':{}}
    XNT_data={'male':{},'female':{}}
    XNTmask_data={'male':{},'female':{}}

    XT_idx,T_MASK,count_per_row,XNT_idx=KNN_split(embeddings_file)
    XT_idx,T_MASK,count_per_row,XNT_idx=XT_idx[0].tolist(),T_MASK.tolist(),count_per_row.tolist(),XNT_idx[0].tolist()
    T_male_sent=[]
    T_female_sent=[]
    for i in XT_idx:
        # print(i,NT_MASK[i])
        T_male_sent.append([male_lines[int(index)] if index != 0 else None for index in T_MASK[i]])
        T_female_sent.append([female_lines[int(index)] if index != 0 else None for index in T_MASK[i]])
    neighbor_num=count_per_row
 
    XT_data['male']['sent']=[male_lines[int(i)] for i in XT_idx]
    XTmask_data['male']['sent']=[malemask_lines[int(i)] for i in XT_idx]

    XNT_data['male']['sent']=[male_lines[int(i)] for i in XNT_idx]
    XNTmask_data['male']['sent']=[malemask_lines[int(i)] for i in XNT_idx]

    
    XT_data['female']['sent']=[female_lines[int(i)] for i in XT_idx]
    XTmask_data['female']['sent']=[femalemask_lines[int(i)] for i in XT_idx]

    XNT_data['female']['sent']=[female_lines[int(i)] for i in XNT_idx]
    XNTmask_data['female']['sent']=[femalemask_lines[int(i)] for i in XNT_idx]


    XT_data['male']['neighbor_sent']=[]
    XT_data['female']['neighbor_sent']=[]

    for i in range(len(XT_idx)):
        XT_data['male']['neighbor_sent'].append(T_male_sent[i])
        XT_data['female']['neighbor_sent'].append(T_female_sent[i])
    XT_data['male']['neighbor_num']=neighbor_num
    XT_data['female']['neighbor_num']=neighbor_num
    with open('data/train_data/XT_data.pk', 'wb') as file1:
        pickle.dump(XT_data, file1)
    with open('data/train_data/XTmask_data.pk', 'wb') as file4:
        pickle.dump(XTmask_data, file4)
    with open('data/train_data/XNT_data.pk', 'wb') as file2:
        pickle.dump(XNT_data, file2)
    with open('data/train_data/XNTmask_data.pk', 'wb') as file3:
        pickle.dump(XNTmask_data, file3)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertModel.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    if args.debias_type=='gender':
        gpt_sentences_file='data/gpt_data/gender/generate_male.txt'
    elif args.debias_type=='race':
        gpt_sentences_file='data/gpt_data/race/generate_race1.txt'
    elif args.debias_type=='religion':
        gpt_sentences_file='data/gpt_data/religion/generate_religion1.txt'

    with open(gpt_sentences_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)

    embeddings = []
    start_time=time.time()
    embeddings_file=gpt_sentences_file.split('.txt')[0]+'_embedding.txt'
    with open(embeddings_file, 'a', encoding='utf-8') as emb_file:
        for i in tqdm(range(0, len(sentences), args.BS),desc="Converting data", dynamic_ncols=True):
            batch_inputs = {k: v[i:i+args.BS] for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**batch_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            for i in range(len(batch_embeddings)):
                sentence_embedding = outputs.last_hidden_state[i, 0, :].numpy()  
                emb_file.write(' '.join(str(x) for x in sentence_embedding) + '\n')
            embeddings.append(batch_embeddings)
    embeddings_file='data/gpt_data/gender/uni_male_embeddings.txt'
    save_pk(embeddings_file)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time consumed for data dividing:", total_time, "seconds")
