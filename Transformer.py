import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import Trainer,TrainingArguments
from transformers import BertForSequenceClassification
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback
from sklearn.metrics import mean_squared_error,mean_absolute_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for training.")

doc = []
labels = []

for filename in os.listdir('business'):
    with open(os.path.join('business',filename),'r',encoding='utf-8') as f:
        doc.append(f.read())
        labels.append(0)
for filename in os.listdir('entertainment'):
    with open(os.path.join('entertainment',filename),'r',encoding='utf-8') as f:
        doc.append(f.read())
        labels.append(1)
for filename in os.listdir('politics'):
    with open(os.path.join('politics',filename),'r',encoding='utf-8') as f:
        doc.append(f.read())
        labels.append(2)
for filename in os.listdir('sport'):
    with open(os.path.join('sport',filename),'r',encoding='utf-8') as f:
        doc.append(f.read())
        labels.append(3)
for filename in os.listdir('tech'):
    with open(os.path.join('tech',filename),'r',encoding='utf-8') as f:
        doc.append(f.read())
        labels.append(4)

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\n\n',' ',text)
    text = text.strip().lower()
    return text

doc_clean = [clean_text(docu) for docu in doc]

#words of the article
'''doc_len = [len(docu.split()) for docu in doc]
bins = range(0,1200,50)
hist, bin_edges = np.histogram(doc_len, bins=bins)
intervals = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)]
fig = plt.figure(figsize=(12,6))
plt.bar(intervals, hist)

plt.title('words of the article')
plt.xticks(rotation=30)
plt.ylabel('count')
plt.show()'''

#tokenize and encode
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_encode(doc):
    return tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors='pt')

encoded_doc = tokenize_and_encode(doc_clean)

#show cleaned article and encoded article
'''print(doc_clean[0])
print(encoded_doc['input_ids'][0])'''

dataset = Dataset.from_dict({'input_ids':encoded_doc['input_ids'],'label':labels}).shuffle()

train_test = dataset.train_test_split(test_size=0.3)
train_eval = train_test['train'].train_test_split(test_size=0.1)

train_dataset = train_eval['train']
eval_dataset = train_eval['test']
test_dataset = train_test['test']

model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained('D:\\bert',num_labels=5)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
test_res = trainer.predict(test_dataset)
y_pred = test_res.predictions
y_pred_label = np.argmax(y_pred,axis=1)
y_true = test_dataset["label"]


train_loss = []
eval_loss = []
for log in trainer.state.log_history:
    loss_value = log.get('loss')
    if loss_value is not None:
        train_loss.append(loss_value)
    eval_los = log.get('eval_loss')
    if eval_los is not None:
        eval_loss.append(eval_los)

plt.figure()
plt.plot(train_loss,label='Training Loss',c='r')
plt.plot([70,140,210],eval_loss,label='Validation Loss',c='b')
plt.title('Loss Curve')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


