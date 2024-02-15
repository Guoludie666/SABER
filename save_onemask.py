import pickle
import tqdm
with open('data/train_data/XNT_data.pk', 'rb') as file,open('data/train_data/XNTmask_data.pk', 'rb') as file1:
    XNT_data = pickle.load(file)
    XNTmask_data = pickle.load(file1)

tar1_words=XNT_data["male"]["sent"]
tar2_words=XNT_data["female"]["sent"]
tar1_maskwords=XNTmask_data["male"]["sent"]
tar2_maskwords=XNTmask_data["female"]["sent"]

def clean_and_save(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords):
    tar_sen={"male":{"sent":[]},"female":{"sent":[]}}
    tar_masksen={"male":{"sent":[]},"female":{"sent":[]}}
    for index,i in enumerate(tar1_maskwords):
        if i.count("[MASK]")==1 and tar2_maskwords[index].count("[MASK]")==1:
            tar_sen["male"]["sent"].append(tar1_words[index])
            tar_sen["female"]["sent"].append(tar2_words[index])
            tar_masksen["male"]["sent"].append(tar1_maskwords[index])
            tar_masksen["female"]["sent"].append(tar2_maskwords[index])
    print(len(tar_sen["male"]["sent"]))
    with open('data/train_data/one_mask/XNT_data.pk', 'wb') as f,open('data/train_data/one_mask/XNTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)

clean_and_save(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords)

