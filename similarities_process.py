# -*- coding: utf - 8 -*-

import json
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def KNN_split():
    dim = 768
    # METRIC_INNER_PRODUCT
    measure = faiss.METRIC_INNER_PRODUCT
    #IVFxPQy ('IVFx,PQy')
    param = "HNSW64"
    res = faiss.StandardGpuResources()  # use a single GPU
    embeddings = []
    with open("uni_male_embeddings.txt", "r", encoding='utf8') as reader:
        lines=reader.readlines()
        count=0
        for line in lines[:20]:
            line.strip()
            parts = line.split(' ')
            assert len(parts) == dim
            v = list(map(lambda x: float(x), parts[0:]))
            embeddings.append(v)

        # faiss index
        index = faiss.index_factory(dim, param, measure)
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index=index
        print(gpu_index.is_trained)
        embeddings = np.array(embeddings, dtype=np.float32)
        gpu_index.train(embeddings)
        print("train OK")
        gpu_index.add(embeddings)
        print(gpu_index.ntotal)
        # KNN
        D, I = gpu_index.search(embeddings, 6)
        print(I)

        threshold=206
        #ow_indices, col_indices = np.where(D > threshold)
        NT_MASK = np.where(D > threshold,I,0)
        NT_MASK[:, 0] = 0
        print(NT_MASK)
        NT_MASK_ = np.where(NT_MASK > 0,1,0)
        count_per_row = np.sum(NT_MASK_, axis=1)
        print(count_per_row)
        print(np.sum(count_per_row > 0, axis=0),len(lines))
        XNT_idx=np.where(count_per_row>0)
        XT_idx=np.where(count_per_row==0)

        return XNT_idx,NT_MASK,count_per_row,XT_idx

def save_pk():
    with open("generate_male.txt", "r", encoding='utf8') as f1:
        male_lines=f1.readlines()[:5]
    with open("male.txt", "r", encoding='utf8') as f1_:
        malemask_lines=f1_.readlines()[:5]
    with open("generate_female.txt", "r", encoding='utf8') as f2:
        female_lines=f2.readlines()[:5]
    with open("female.txt", "r", encoding='utf8') as f2_:
        femalemask_lines=f2_.readlines()[:5]

    XNT_data={'male':{},'female':{}}
    XT_data={'male':{},'female':{}}
    XTmask_data={'male':{},'female':{}}


    XNT_idx,NT_MASK,count_per_row,XT_idx=KNN_split()
    #print(XNT_idx[0].shape,NT_MASK.shape,count_per_row.shape,XT_idx[0].shape)

    XNT_idx,NT_MASK,count_per_row,XT_idx=XNT_idx[0].tolist(),NT_MASK.tolist(),count_per_row.tolist(),XT_idx[0].tolist()
    NT_male_sent=[]
    NT_female_sent=[]
    for i in XNT_idx:
        NT_male_sent.append([male_lines[int(index)] if index != 0 else None for index in NT_MASK[i]])
        NT_female_sent.append([female_lines[int(index)] if index != 0 else None for index in NT_MASK[i]])
    neighbor_num=count_per_row
 
    XNT_data['male']['sent']=[male_lines[int(i)] for i in XNT_idx]
    XT_data['male']['sent']=[male_lines[int(i)] for i in XT_idx]
    XTmask_data['male']['sent']=[malemask_lines[int(i)] for i in XT_idx]

    
    XNT_data['female']['sent']=[female_lines[int(i)] for i in XNT_idx]
    XT_data['female']['sent']=[female_lines[int(i)] for i in XT_idx]
    XTmask_data['female']['sent']=[femalemask_lines[int(i)] for i in XT_idx]


    XNT_data['male']['neighbor_sent']=[]
    XNT_data['female']['neighbor_sent']=[]

    for i in range(len(XNT_idx)):
        XNT_data['male']['neighbor_sent'].append(NT_male_sent[i])
        XNT_data['female']['neighbor_sent'].append(NT_female_sent[i])
    XNT_data['male']['neighbor_num']=neighbor_num
    XNT_data['female']['neighbor_num']=neighbor_num

    # 将字典数据保存到.pk文件
    with open('data/XNT_data.pk', 'wb') as file1:
        pickle.dump(XNT_data, file1)
    with open('data/XT_data.pk', 'wb') as file2:
        pickle.dump(XT_data, file2)
    with open('data/XTmask_data.pk', 'wb') as file3:
        pickle.dump(XTmask_data, file3)




def main():
    with open('data/XNT_data.pk', 'rb') as file,open('data/XT_data.pk', 'rb') as file_,open('data/XTmask_data.pk', 'rb') as file__:
        data = pickle.load(file)
        count=0
        print(data["male"]['sent'][0])
        for index,i in enumerate (data["male"]['sent'][:20]):
            for j in range(6):
                print(data["male"]['neighbor_sent'][index][j])
   
if __name__ == '__main__':
    #KNN_split()
    main()
