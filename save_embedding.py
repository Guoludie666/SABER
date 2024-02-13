from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import time
# 读取txt文件中的所有句子
with open('file2.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# 将句子转换为BERT模型的输入
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)

# 定义批处理大小
batch_size = 8

# 使用批处理计算嵌入
embeddings = []
# 保存句子的隐藏层状态到文件中
start_time=time.time()
with open('uni4_male_embeddings.txt', 'a', encoding='utf-8') as emb_file:
    for i in tqdm(range(0, len(sentences), batch_size),desc="Processing data", dynamic_ncols=True):
        batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        for i in range(len(batch_embeddings)):
            sentence_embedding = outputs.last_hidden_state[i, 0, :].numpy()  
            emb_file.write(' '.join(str(x) for x in sentence_embedding) + '\n')
        embeddings.append(batch_embeddings)

end_time = time.time()
total_time = end_time - start_time
print("Total time:", total_time, "seconds")
