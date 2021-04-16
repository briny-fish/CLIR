from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction,BertModel
from transformers import AdamW
import pickle
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys
import pickle
import os
import Metrics as mc
import torch
import dataLoader
import torch.nn as nn
import dataProc as dp
import torch.nn.functional as F
from collections import OrderedDict
import random
torch.cuda.set_device(0)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bertmodel = BertModel.from_pretrained('bert-base-multilingual-uncased', return_dict=True)
#datas = dataLoader.readTrainDatas()
#Datas = pd.DataFrame(datas)
#tests= dataLoader.readTestDatas()
#Tests = pd.DataFrame(tests)
#devs = dataLoader.readDevDatas()
#Devs = pd.DataFrame(devs)
MAX_SEQUENCE_LENGTH = 200
train_batch_size = 5
test_batch_size = 200
epoch_num = 6
#Datas['label']=Datas['label'].astype(int)
#Tests['label']=Tests['label'].astype(int)
#print(Tests.head(201))
#Devs['label']=Devs['label'].astype(int)
input_categories = ['query','doc']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
query = dataLoader.readQuery()
doc = dataLoader.readDoc()
label = dataLoader.readLabel()
Query = pd.DataFrame(query)
Doc = pd.DataFrame(doc)
'''

out1 = open('/home/data_ti4_c/zongwz/data_en_de/train_inputs50000','rb')
out2 = open('/home/data_ti4_c/zongwz/data_en_de/test_inputs50000','rb')
#out3 = open('/home/data_ti4_c/zongwz/data_en_de/dev_inputs50000','rb')
out4 = open('/home/data_ti4_c/zongwz/data_en_de/train_outputs50000','rb')
out5 = open('/home/data_ti4_c/zongwz/data_en_de/200test_outputs200000','rb')
#out6 = open('/home/data_ti4_c/zongwz/data_en_de/200dev_outputs200000','rb')
#train_outputs = dp.compute_output_arrays(Datas, 'label')
#train_inputs = dp.compute_input_arrays(Datas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#test_outputs = dp.compute_output_arrays(Tests, 'label')
#test_inputs = dp.compute_input_arrays(Tests, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#dev_outputs = dp.compute_output_arrays(Devs, 'label')
#dev_inputs = dp.compute_input_arrays(Devs, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = pickle.load(out2)
test_outputs = pickle.load(out5)

train_inputs = pickle.load(out1)
#test_inputs = pickle.load(out2)
#dev_inputs = pickle.load(out3)
train_outputs = pickle.load(out4)
#test_outputs = pickle.load(out5)
#dev_outputs = pickle.load(out6)
print('success')
print(test_outputs[:10])
out1.close()
#out6.close()
out5.close()
out4.close()
#out3.close()
out2.close()

class BertMatch(nn.Module):
    def __init__(self):
        super(BertMatch,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased', return_dict=True)
        self.cls0 = nn.Linear(768,768)
        self.cls1 = nn.Linear(768,1)

    def forward(self,input_ids,attention_mask,token_type_ids):
        bertout = self.bert(input_ids,attention_mask,token_type_ids)['pooler_output']
        bertout = self.cls0(bertout)
        #[batchsize,seq_size,hid_size]
        bertout = F.sigmoid(bertout)
        out = self.cls1(bertout)
        out = F.sigmoid(out)
        #[batch_size,4]
        return out
model = BertMatch()
model = model.to(device)
learning_rate = 3e-7
optim = torch.optim.AdamW(model.parameters(),lr = learning_rate)
criterion = nn.MarginRankingLoss()
train_metric = mc.Metrics()
def run_model():
    for epoch in range(epoch_num):
        print('epoch:{}'.format(epoch))
        train_metric = mc.Metrics()
        test_metric = mc.Metrics()
        dev_metric = mc.Metrics()
        #print(train_inputs[0])
        one_epoch(train_metric,0,train_inputs,train_outputs,train_batch_size)
        #one_epoch(dev_metric, 1, dev_inputs, dev_outputs,test_batch_size)
        one_epoch(test_metric,1,test_inputs,test_outputs,test_batch_size)
        #valid_acc,valid_loss = dev_metric.compute()
        train_acc,  train_loss = train_metric.compute()
        test_acc,  test_loss = test_metric.compute()
        print('train:map1:%s  loss:%s' % (train_acc, train_loss))
        #print('valid:map1:%s  loss:%s' % (valid_acc,  valid_loss))
        print('test:map1:%s  loss:%s' % (test_acc,  test_loss))


def one_epoch(metric,state,inputs,labels,batch_size):
    if state == 0:model.train()
    else:model.eval()
    loss = ''
    #print(inputs[:10])
    #print(inputs[0])
    model_output = ''
    for cnt in tqdm(range(0, len(inputs[0]), batch_size)):
        #if cnt>400:break
        torch.cuda.empty_cache()
        idx = []
        gd = 0
        if (cnt + batch_size < len(inputs[0])):
            idx = list(range(cnt, cnt + batch_size))
            random.shuffle(idx)
            for i in range(len(idx)):
                if(idx[i]==cnt):gd = i
        else:
            break

        simples = [[inputs[i][id] for id in idx] for i in range(3)]
        if(labels[cnt] == 0):
            print('some postive sample loaded failure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            break
        simples = torch.LongTensor(simples).to(device)
        loss0 = ''
        if state == 0:
            
                
            model_output = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2])
            #print(model_output.shape)
            loss0 = criterion(model_output[gd],model_output[(gd+1)%batch_size],torch.LongTensor([1]).to(device))
            for i in range(batch_size-2):
                loss0 += criterion(model_output[gd],model_output[(gd+i+2)%batch_size],torch.LongTensor([1]).to(device))
            if(loss == ''):loss = loss0
            else: loss += loss0
            
            if(cnt%20 == 0 or len(inputs[0])-cnt < 20):
                loss.backward()
                optim.step()
                optim.zero_grad()
                loss = ''
        else:
            with torch.no_grad():
                model_output = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2])
                loss0 = criterion(model_output[gd],model_output[(gd+1)%batch_size],torch.LongTensor([1]).to(device))
                for i in range(batch_size-2):
                    loss0 += criterion(model_output[gd],model_output[(gd+i+2)%batch_size],torch.LongTensor([1]).to(device))
        model_output = model_output
        #print(model_output)
        pred = torch.argmax(model_output.squeeze(),dim = -1)
        target = gd
        true_num = float(pred == target)
        metric.add_arg(1.0, true_num, loss0)

run_model()
torch.save(model.state_dict(), r'/home/data_ti4_c/zongwz/data_en_de/parameter.pkl')