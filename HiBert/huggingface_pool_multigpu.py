import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score
import logging
import numpy as np
import pickle
from huggingface_dataloader import BertDatasetFromDiskMultiGPU

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

#data settings
tasks = ['site','histology']
with open('data/bert/num_classes.pkl','rb') as f:
    num_classes = pickle.load(f)
task_idx = 1
print(num_classes)
num_classes = num_classes[task_idx]
task = tasks[task_idx]

class HiBERT(nn.Module):

    def __init__(self,bert_load_path,num_classes):
        
        super(HiBERT,self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_load_path,
                    num_labels=num_classes)

    def forward(self,input_ids,input_mask,segment_ids,n_segs):
        
        n_segs = n_segs.view(-1)
        input_ids_ = input_ids.view(-1,512)[:n_segs]
        input_mask_ = input_mask.view(-1,512)[:n_segs]
        segment_ids_ = segment_ids.view(-1,512)[:n_segs]

        logits = self.bert(input_ids_,input_mask_,segment_ids_,labels=None)[0]
        logits = torch.max(logits,0)[0]

        return logits

    def save_bert(self,savepath):

        self.bert.save_pretrained(savepath)

#load datasets
train_data = BertDatasetFromDiskMultiGPU('data/bert/train/')
val_data = BertDatasetFromDiskMultiGPU('data/bert/val/')
test_data = BertDatasetFromDiskMultiGPU('data/bert/test/')

train_loader = DataLoader(train_data,batch_size=n_gpu,shuffle=True)
val_loader = DataLoader(val_data,batch_size=n_gpu,shuffle=False)
test_loader = DataLoader(test_data,batch_size=n_gpu,shuffle=False)

#train bert
model = HiBERT('savedmodels/ncbi_bert_base_pubmed_mimic_uncased',num_classes)
model = torch.nn.DataParallel(model)
model.to(device)

loss_fct = torch.nn.CrossEntropyLoss()
params = [{'params':[p for n,p in model.named_parameters()],'weight_decay':0.0}]
#optimizer = AdamW(params,lr=2e-5,eps=1e-8)
optimizer = torch.optim.Adam(params,lr=5e-5,eps=1e-8)

best_score = 0
patience = 5
batch_size = 16 #128 
opt_every = np.ceil(batch_size/n_gpu)
pat_count = 0

if not os.path.exists('savedmodels/biobert_pool_%s' % task):
    os.makedirs('savedmodels/biobert_pool_%s' % task)

for ep in range(100):

    model.train()

    for i,batch in enumerate(train_loader):

        input_ids = batch['tokens'].view(-1,512).to(device)
        segment_ids = batch['seg_ids'].view(-1,512).to(device)
        input_mask = batch['masks'].view(-1,512).to(device)
        n_segs = batch['n_segs'].to(device)
        logits = model(input_ids,input_mask,segment_ids,n_segs)

        label = batch['labels_%s' % task].to(device)
        loss = loss_fct(logits.view(-1,num_classes),label.view(-1))/opt_every
        loss.mean().backward()

        if (i+1) % opt_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss = loss.cpu().detach().numpy()
        sys.stdout.write('epoch %i batch %i loss: %.6f      \r' % (ep+1,i+1,loss))
        sys.stdout.flush()

    print()

    model.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():

        val_preds = []
        val_labels = []

        for j,batch in enumerate(val_loader):

            input_ids = batch['tokens'].view(-1,512).to(device)
            segment_ids = batch['seg_ids'].view(-1,512).to(device)
            input_mask = batch['masks'].view(-1,512).to(device)
            n_segs = batch['n_segs'].to(device)
            logits = model(input_ids,input_mask,segment_ids,n_segs).view(-1,num_classes)

            label = batch['labels_%s' % task].to(device)
            preds = torch.nn.Softmax(-1)(logits)
            preds = preds.cpu().data.numpy()
            val_preds.append(np.argmax(preds,1))
            val_labels.append(label.cpu().data.numpy())

            sys.stdout.write('predicting batch %i       \r' % (j+1))
            sys.stdout.flush()

        val_preds = np.hstack(val_preds)
        val_labels = np.hstack(val_labels)
        val_micro = f1_score(val_labels,val_preds,average='micro')
        val_macro = f1_score(val_labels,val_preds,average='macro')
        print('epoch %i %s val micro/macro: %.4f, %.4f' % (ep+1,task,val_micro,val_macro))

    if val_micro > best_score:
        best_score = val_micro
        pat_count = 0
        model.module.save_bert('savedmodels/biobert_pool_%s' % task)
        print('saving model')
        with open('savedmodels/biobert_pool_%s/optimizer.pkl' % task,'wb') as f:
           pickle.dump(optimizer.state_dict(),f)

    else:
        pat_count += 1
        if pat_count >= patience:
            break

model = HiBERT('savedmodels/biobert_pool_%s' % task,num_classes)
model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for i,batch in enumerate(test_loader):

        input_ids = batch['tokens'].view(-1,512).to(device)
        segment_ids = batch['seg_ids'].view(-1,512).to(device)
        input_mask = batch['masks'].view(-1,512).to(device)
        n_segs = batch['n_segs'].to(device)
        logits = model(input_ids,input_mask,segment_ids,n_segs).view(-1,num_classes)

        label = batch['labels_%s' % task].to(device)
        preds = torch.nn.Softmax(-1)(logits)
        preds = preds.cpu().data.numpy()
        test_preds.append(np.argmax(preds,1))
        test_labels.append(label.cpu().data.numpy())

        sys.stdout.write('predicting batch %i       \r' % (i+1))
        sys.stdout.flush()

test_preds = np.hstack(test_preds)
test_labels = np.hstack(test_labels)
test_micro = f1_score(test_labels,test_preds,average='micro')
test_macro = f1_score(test_labels,test_preds,average='macro')
print('%s test micro/macro: %.4f, %.4f' % (task,test_micro,test_macro))
