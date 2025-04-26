# Updated: April 26, 2025
import pickle
import tqdm

def clean_and_save_double(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords):
    tar_sen={"attribute0":{"sent":[]},"attribute1":{"sent":[]}}
    tar_masksen={"attribute0":{"sent":[]},"attribute1":{"sent":[]}}
    for index,i in enumerate(tar1_maskwords):
        if i.count("[MASK]")==1 and tar2_maskwords[index].count("[MASK]")==1:
            tar_sen["attribute0"]["sent"].append(tar1_words[index])
            tar_sen["attribute1"]["sent"].append(tar2_words[index])
            tar_masksen["attribute0"]["sent"].append(tar1_maskwords[index])
            tar_masksen["attribute1"]["sent"].append(tar2_maskwords[index])
    print(len(tar_sen["attribute0"]["sent"]))
    with open('data/train_data/one_mask/XNT_data.pk', 'wb') as f,open('data/train_data/one_mask/XNTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)
def clean_and_save_triple(tar1_words,tar2_words,tar3_words,tar1_maskwords,tar2_maskwords,tar3_maskwords):
    tar_sen={"attribute0":{"sent":[]},"attribute1":{"sent":[]},"attribute2":{"sent":[]}}
    tar_masksen={"attribute0":{"sent":[]},"attribute1":{"sent":[]},"attribute2":{"sent":[]}}
    for index,i in enumerate(tar1_maskwords):
        if i.count("[MASK]")==1:
            tar_sen["attribute0"]["sent"].append(tar1_words[index])
            tar_sen["attribute1"]["sent"].append(tar2_words[index])
            tar_sen["attribute2"]["sent"].append(tar3_words[index])

            tar_masksen["attribute0"]["sent"].append(tar1_maskwords[index])
            tar_masksen["attribute1"]["sent"].append(tar2_maskwords[index])
            tar_masksen["attribute2"]["sent"].append(tar3_maskwords[index])
    print(len(tar_sen["attribute0"]["sent"]))
    with open('data/train_data/one_mask/XNT_data.pk', 'wb') as f,open('data/train_data/one_mask/XNTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)

with open('data/train_data/XNT_data.pk', 'rb') as file,open('data/train_data/XNTmask_data.pk', 'rb') as file1:
    XNT_data = pickle.load(file)
    XNTmask_data = pickle.load(file1)
keys=XNT_data.keys()
tar1_words=XNT_data["attribute0"]["sent"]
tar2_words=XNT_data["attribute1"]["sent"]
tar1_maskwords=XNTmask_data["attribute0"]["sent"]
tar2_maskwords=XNTmask_data["attribute1"]["sent"]
if len(keys)==3:
    tar3_words=XNT_data["attribute2"]["sent"]
    tar3_maskwords=XNTmask_data["attribute2"]["sent"]
    clean_and_save_triple(tar1_words,tar2_words,tar3_words,tar1_maskwords,tar2_maskwords,tar3_maskwords)
else:
    clean_and_save_double(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords)

