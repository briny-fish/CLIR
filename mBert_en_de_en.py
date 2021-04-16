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
import gensim
import os
import Metrics as mc
import torch
import dataLoader_en_de_en as dataLoader
import torch.nn as nn
import dataProc as dp
import torch.nn.functional as F
from collections import OrderedDict
import random
torch.cuda.set_device(1)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bertmodel = BertModel.from_pretrained('bert-base-multilingual-uncased', return_dict=True)
datas = dataLoader.readTrainDatas()
Datas = pd.DataFrame(datas)
tests= dataLoader.readTestDatas()
Tests = pd.DataFrame(tests)
#devs = dataLoader.readDevDatas()
#Devs = pd.DataFrame(devs)
MAX_SEQUENCE_LENGTH = 200
train_batch_size = 5
test_batch_size = 200
bi_train_batch_size = train_batch_size * 2
bi_test_batch_size = test_batch_size * 2
epoch_num = 20
#Datas['label']=Datas['label'].astype(int)
#Tests['label']=Tests['label'].astype(int)
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

out1 = open('/home/data_ti4_c/zongwz/data_en_de/bi_train_inputs200000','rb')
out2 = open('/home/data_ti4_c/zongwz/data_en_de/bi_test_inputs200000','rb')
#out3 = open('/home/data_ti4_c/zongwz/data_en_de/bi_dev_inputs200000','wb')
out4 = open('/home/data_ti4_c/zongwz/data_en_de/bi_train_outputs200000','rb')
out5 = open('/home/data_ti4_c/zongwz/data_en_de/bi_test_outputs200000','rb')
#out6 = open('/home/data_ti4_c/zongwz/data_en_de/bi_dev_outputs200000','wb')
#train_outputs = dp.compute_output_arrays(Datas, 'label')
#train_inputs = dp.compute_input_arrays(Datas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#test_outputs = dp.compute_output_arrays(Tests, 'label')
#test_inputs = dp.compute_input_arrays(Tests, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#pickle.dump(test_outputs,out5)
#pickle.dump(test_inputs,out2)
#pickle.dump(train_outputs,out4)
#pickle.dump(train_inputs,out1)
#dev_outputs = dp.compute_output_arrays(Devs, 'label')
#dev_inputs = dp.compute_input_arrays(Devs, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
train_langs = list(Datas['lang'])
#dev_langs = list(Devs['lang'])
test_langs = list(Tests['lang'])
train_inputs = pickle.load(out1)
test_inputs = pickle.load(out2)
#dev_inputs = pickle.load(out3)
train_outputs = pickle.load(out4)
test_outputs = pickle.load(out5)
#dev_outputs = pickle.load(out6)
print('success')
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

    def forward(self,input_ids,attention_mask,token_type_ids,batch_size = 0):
        if batch_size == 0:
            bertout = self.bert(input_ids,attention_mask,token_type_ids)['pooler_output']
            bertout = self.cls0(bertout)
            #[batchsize,seq_size,hid_size]
            bertout = F.sigmoid(bertout)
            out = self.cls1(bertout)
            out = F.sigmoid(out)
            #[batch_size,4]
            return out
        else:
            bertout2 = self.bert(input_ids[:batch_size // 2],attention_mask[:batch_size // 2],token_type_ids[:batch_size // 2])['pooler_output']
            bertout0 = self.bert(input_ids[batch_size // 2:],attention_mask[batch_size // 2:],token_type_ids[batch_size // 2:])['pooler_output']
            bertout1 = self.cls0(bertout0)
            #print(input_ids[batch_size // 2 + 1:].size())
            #print(input_ids.size())
            #[batchsize,seq_size,hid_size]
            bertout1 = F.sigmoid(bertout1)
            out1 = self.cls1(bertout1)
            out1 = F.sigmoid(out1)
            #print(out1)
            max_out = bertout0[torch.argmax(out1)]
            #print(out1)
            #print(torch.argmax(out1,dim = -1))
            #print(bertout2.size())
            bertout2 = bertout2 + max_out
            out2 = self.cls0(bertout2)
            out2 = F.sigmoid(out2)
            out2 = self.cls1(out2)
            out2 = F.sigmoid(out2)
            #[batch_size,4]
            return out1,out2

model = BertMatch()
model = model.to(device)
learning_rate = 2e-7
optim = torch.optim.AdamW(model.parameters(),lr = learning_rate)
criterion = nn.MarginRankingLoss()
train_metric = mc.Metrics()
def run_model():
    for epoch in range(epoch_num):
        en_train_metric = mc.Metrics()
        en_test_metric = mc.Metrics()
        en_dev_metric = mc.Metrics()
        de_train_metric = mc.Metrics()
        de_test_metric = mc.Metrics()
        de_dev_metric = mc.Metrics()
        
        print(train_inputs[0])
        #one_epoch(de_train_metric,en_train_metric,0,train_inputs,train_outputs,train_batch_size,train_langs)
        #one_epoch(de_dev_metric,en_dev_metric, 1, dev_inputs, dev_outputs,test_batch_size,dev_langs)
        #one_epoch(de_test_metric,en_test_metric,1,test_inputs,test_outputs,test_batch_size,test_langs)
        bi_one_epoch(de_train_metric,en_train_metric,0,train_inputs,train_outputs,bi_train_batch_size,train_langs)
        #one_epoch(de_dev_metric,en_dev_metric, 1, dev_inputs, dev_outputs,test_batch_size,dev_langs)
        bi_one_epoch(de_test_metric,en_test_metric,1,test_inputs,test_outputs,bi_test_batch_size,test_langs)

        #valid_acc,valid_loss = dev_metric.compute()
        en_train_acc,  en_train_loss = en_train_metric.compute()
        en_test_acc,  en_test_loss = en_test_metric.compute()
        print('en_train:map1:%s  loss:%s' % (en_train_acc, en_train_loss))
        #print('valid:map1:%s  loss:%s' % (valid_acc,  valid_loss))
        print('en_test:map1:%s  loss:%s' % (en_test_acc,  en_test_loss))

        de_train_acc,  de_train_loss = de_train_metric.compute()
        de_test_acc,  de_test_loss = de_test_metric.compute()
        print('de_train:map1:%s  loss:%s' % (de_train_acc, de_train_loss))
        #print('valid:map1:%s  loss:%s' % (valid_acc,  valid_loss))
        print('de_test:map1:%s  loss:%s' % (de_test_acc,  de_test_loss))


def one_epoch(demetric,enmetric,state,inputs,labels,batch_size,langs):
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
        lang = langs[cnt]
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
            
            if(cnt%30 == 0 or len(inputs[0])-cnt < 30):
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
        #model_output = model_output
        #print(model_output)
        pred = torch.argmax(model_output.squeeze(),dim = -1)
        target = gd
        true_num = float(pred == target)
        if(lang == 'en'):
            enmetric.add_arg(1.0, true_num, loss0)
        else:
            demetric.add_arg(1.0, true_num, loss0)
    if(state==0):torch.save(model.state_dict(), r'/home/data_ti4_c/zongwz/data_en_de/bi-parameter.pkl')

def bi_one_epoch(demetric,enmetric,state,inputs,labels,batch_size,langs):
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
        gd_en = 0
        gd_de = 0
        if (cnt + batch_size < len(inputs[0])):
            idx1 = list(range(cnt, cnt + batch_size // 2))
            idx2 = list(range(cnt + batch_size // 2, cnt + batch_size))
            random.shuffle(idx1)
            for i in range(len(idx1)):
                if(idx1[i]==cnt):gd_de = i
            random.shuffle(idx2)
            for i in range(len(idx2)):
                if(idx2[i]==cnt + batch_size // 2):gd_en = i
            idx = idx1 + idx2
        else:
            break

        simples = [[inputs[i][id] for id in idx] for i in range(3)]
        lang_1 = langs[cnt]
        lang_2 = langs[cnt + batch_size // 2]
        if(labels[cnt] == 0):
            print('some postive sample loaded failure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            break
        simples = torch.LongTensor(simples).to(device)
        loss0 = ''
        loss_en = ''
        loss_de = ''
        if state == 0:
            
                
            out_de,out_en = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2], batch_size = batch_size)
            #print(model_output.shape)
            
            for i in range(batch_size//2):
                    if(i == gd_de):continue
                    if(loss_de==''):
                        loss_de = criterion(out_de[gd_de],out_de[i],torch.LongTensor([1]).to(device)) 
                    else:
                        loss_de += criterion(out_de[gd_de],out_de[i],torch.LongTensor([1]).to(device)) 
            
            for i in range(batch_size//2):
                if(i + batch_size/2 == gd_en):continue
                if(loss_en==''):
                    loss_en = criterion(out_en[gd_en - batch_size//2],out_en[i],torch.LongTensor([1]).to(device)) 
                else:
                    loss_en += criterion(out_en[gd_en - batch_size//2],out_en[i],torch.LongTensor([1]).to(device))

            if(loss == ''):loss = loss_en+loss_de
            else: loss += loss_en+loss_de
            
            if(cnt%30 == 0 or len(inputs[0])-cnt < 30):
                loss.backward()
                optim.step()
                optim.zero_grad()
                loss = ''
        else:
            with torch.no_grad():
                out_de,out_en = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2], batch_size = batch_size)
                for i in range(batch_size//2):
                    if(i == gd_de):continue
                    if(loss_de==''):
                        loss_de = criterion(out_de[gd_de],out_de[i],torch.LongTensor([1]).to(device)) 
                    else:
                        loss_de += criterion(out_de[gd_de],out_de[i],torch.LongTensor([1]).to(device)) 
            
                for i in range(batch_size//2):
                    if(i + batch_size/2 == gd_en):continue
                    if(loss_en==''):
                        loss_en = criterion(out_en[gd_en - batch_size//2],out_en[i],torch.LongTensor([1]).to(device)) 
                    else:
                        loss_en += criterion(out_en[gd_en - batch_size//2],out_en[i],torch.LongTensor([1]).to(device))
        #print(model_output)
        pred_de = int(torch.argmax(out_de))
        pred_en = int(torch.argmax(out_en) + batch_size // 2)
        target_de = gd_de
        target_en = gd_en
        true_num_de = float(pred_de == target_de)
        true_num_en = float(pred_en == target_en)
        enmetric.add_arg(1.0, true_num_en, loss_en)
        demetric.add_arg(1.0, true_num_de, loss_de)
    if(state==0):torch.save(model.state_dict(), r'/home/data_ti4_c/zongwz/data_en_de/idea-bi-parameter.pkl')
run_model()
