import pickle
import tqdm
with open('data/train_data/XNT_data.pk', 'rb') as file,open('data/train_data/XNTmask_data.pk', 'rb') as file1:
    XT_data = pickle.load(file)
    XTmask_data = pickle.load(file1)

tar1_words=XT_data["male"]["sent"]
tar2_words=XT_data["female"]["sent"]
tar1_maskwords=XTmask_data["male"]["sent"]
tar2_maskwords=XTmask_data["female"]["sent"]

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
    with open('data/train_data/data_onemask/XNT_data.pk', 'wb') as f,open('data/train_data/data_onemask/XTmask_data.pk', 'wb') as f1:
        pickle.dump(tar_sen, f)
        pickle.dump(tar_masksen, f1)

def check(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords):
    for index,i in enumerate(tar1_maskwords[:10000]):
        l2=tar2_maskwords[index]
        if i.count("MASK")!=1:
            print(i,l2)
clean_and_save(tar1_words,tar2_words,tar1_maskwords,tar2_maskwords)

