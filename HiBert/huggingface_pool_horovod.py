import os
import sys
import torch
import torch.nn as nn
import horovod.torch as hvd
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
device = torch.device("cuda")
hvd.init()
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

#data settings
tasks = ['site','histology']
with open('data/bert/num_classes.pkl','rb') as f:
    num_classes = pickle.load(f)
task_idx = 1
print(num_classes)
num_classes = num_classes[task_idx]
task = tasks[task_idx]

class HiBERT(nn.Module):
    '''
    hierarchical BERT with max pooling across segments
    designed for single task, single label document classification
    splits long documents into n segments of 512 each
    predicts on each individual segment, then uses maxpool on logits to combine results from each segment
    returns logits across possible classes for given input document 
    '''
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

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data,num_replicas=hvd.size(),rank=hvd.rank(),shuffle=True)
val_sampler = torch.utils.data.distributed.DistributedSampler(
              val_data,num_replicas=hvd.size(),rank=hvd.rank(),shuffle=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(
               test_data,num_replicas=hvd.size(),rank=hvd.rank(),shuffle=False)

#data loaders
train_loader = DataLoader(train_data,batch_size=1,sampler=train_sampler)
val_loader = DataLoader(val_data,batch_size=1,sampler=val_sampler)
test_loader = DataLoader(test_data,batch_size=1,sampler=test_sampler)

#init bert
model = HiBERT('/gpfs/wolf/proj-shared/gen149/ncbi_bert_base_pubmed_mimic_uncased',num_classes)
model.cuda()

#init loss and optimizer
loss_fct = torch.nn.CrossEntropyLoss()
params = [{'params':[p for n, p in model.named_parameters()],'weight_decay':0.0}]
if hvd.rank() == 0:
    print([n for n,p in model.named_parameters()])
#optimizer = BertAdam(params,lr=2e-5,e=1e-8,weight_decay=0.0)
optimizer = torch.optim.Adam(params,lr=5e-5,eps=1e-8)
optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=model.named_parameters())

#variables to track patience and validation performance
best_score = 0
patience = 5
pat_count = 0

#broadcast params
hvd.broadcast_parameters(model.state_dict(),root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

if hvd.rank() == 0:
    if not os.path.exists('savedmodels/biobert_pool_%s' % task):
        os.makedirs('savedmodels/biobert_pool_%s' % task)

#train loop
for ep in range(100):

    model.train()

    for i,batch in enumerate(train_loader):

        input_ids = batch['tokens'].view(-1,512).to(device)
        segment_ids = batch['seg_ids'].view(-1,512).to(device)
        input_mask = batch['masks'].view(-1,512).to(device)
        n_segs = batch['n_segs'].to(device)
        logits = model(input_ids,input_mask,segment_ids,n_segs)

        label = batch['labels_%s' % task].to(device)
        loss = loss_fct(logits.view(-1,num_classes),label.view(-1))
        loss.backward()
        optimizer.step()

        loss = loss.cpu().detach().numpy()

    print()

    #validation loop
    model.eval()
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

            print(f'predicting batch {j+1}')

        val_preds = torch.tensor(np.hstack(val_preds))
        val_labels = torch.tensor(np.hstack(val_labels))
        val_preds_all = hvd.allgather(val_preds, name='val_preds_all').cpu().data.numpy()
        val_labels_all = hvd.allgather(val_labels, name='val_labels_all').cpu().data.numpy()
        val_micro = f1_score(val_labels_all,val_preds_all,average='micro')
        val_macro = f1_score(val_labels_all,val_preds_all,average='macro')
        if hvd.rank() == 0:
            print('epoch %i %s val micro/macro: %.4f, %.4f' % (ep+1,task,val_micro,val_macro))

    if hvd.rank() == 0:
        if val_micro > best_score:
            best_score = val_micro
            pat_count = 0
            model.save_bert('savedmodels/biobert_pool_%s' % task)
            print('saving model')
            with open('savedmodels/biobert_pool_%s/optimizer.pkl' % task,'wb') as f:
                pickle.dump(optimizer.state_dict(),f)

    else:
        pat_count += 1
        if pat_count >= patience:
            break

#load best model to predict on test set
model = HiBERT('savedmodels/biobert_pool_%s' % task,num_classes)
model.cuda()
hvd.broadcast_parameters(model.state_dict(),root_rank=0)
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

        print(f'predicting batch {j+1}')

test_preds = torch.tensor(np.hstack(test_preds))
test_labels = torch.tensor(np.hstack(test_labels))
test_preds_all = hvd.allgather(test_preds, name='test_preds_all').cpu().data.numpy()
test_labels_all = hvd.allgather(test_labels, name='test_labels_all').cpu().data.numpy()
test_micro = f1_score(test_labels_all,test_preds_all,average='micro')
test_macro = f1_score(test_labels_all,test_preds_all,average='macro')
if hvd.rank() == 0:
    print('%s test micro/macro: %.4f, %.4f' % (task,test_micro,test_macro))
