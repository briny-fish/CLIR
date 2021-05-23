from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction, BertModel
from transformers import AdamW
import pickle
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
import sys
import Models 
import dataProc
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
paramdict = {}
paramdict['other'] = 'fixed with 0'
paramdict['model'] = 'base'
paramdict['exout_num'] = 5
paramdict['lr'] = 5e-6
param = 'MSE' #ranking or MSE
paramdict['train_size'] = 100000 #100-1000000 or -1
paramdict['test_size'] = 30000#100-100000 or -1
learning_rate = paramdict['lr']
outfile = open('/home/zongwz/%s%sq&docsBertlogen-zh-%s.txt'% (paramdict['model'],param,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), 'w+', encoding='utf-8')
outfile.write(str(paramdict)+'\n')
outfile.write(param)
torch.cuda.set_device(2)
tokenizer = BertTokenizer.from_pretrained('/home/data_ti4_c/zongwz/bert-base-multilingual-cased')
bertmodel = BertModel.from_pretrained('/home/data_ti4_c/zongwz/bert-base-multilingual-cased', return_dict=True)
datas = dataLoader.read_enzh('en-zh.train')
Datas = pd.DataFrame(datas)[:paramdict['train_size']]
test1 = dataLoader.read_enzh('en-zh.test1')
Test1 = pd.DataFrame(test1)[:paramdict['test_size']]
test2 = dataLoader.read_enzh('en-zh.test2')
Test2 = pd.DataFrame(test2)[:paramdict['test_size']]
dev = dataLoader.read_enzh('en-zh.dev')
Dev = pd.DataFrame(dev)[:paramdict['test_size']]
MAX_SEQUENCE_LENGTH = 200
train_batch_size = 4
test_batch_size = 4
epoch_num = 20
Datas['label'] = Datas['label'].astype(float)
Test1['label'] = Test1['label'].astype(float)
Test2['label'] = Test2['label'].astype(float)
Dev['label'] = Dev['label'].astype(float)
input_categories = ['query', 'doc']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def convert_inputs(file,inDatas = ''):
    if os.path.exists(file):
        _inputs = pickle.load(open(file, 'rb'))
    else:
        _inputs = dataProc.compute_input_arrays(inDatas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
        pickle.dump(_inputs, open(file, 'wb'))
    return _inputs
def convert_docinputs(file,inDatas = ''):
    if os.path.exists(file):
        _inputs = pickle.load(open(file, 'rb'))
    else:
        _inputs = dataProc.compute_input_arrays_single(inDatas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
        pickle.dump(_inputs, open(file, 'wb'))
    return _inputs
def get_split(Inp,size):
    tmp = [[x[i] for i in range(size)] for x in Inp]
    return np.array(tmp)
train_docs = get_split(convert_docinputs('/home/zongwz/en-zh_train_docinputs.pkl',Datas),paramdict['train_size'])
test1_docs = get_split(convert_docinputs('/home/zongwz/en-zh_test1_docinputs.pkl',Test1),paramdict['test_size'])
test2_docs = get_split(convert_docinputs('/home/zongwz/en-zh_test2_docinputs.pkl',Test2),paramdict['test_size'])
dev_docs = get_split(convert_docinputs('/home/zongwz/en-zh_dev_docinputs.pkl',Dev),paramdict['test_size'])
train_inputs = get_split(convert_inputs('/home/data_ti4_c/zongwz/CLIR/en-zh_train_inputs.pkl'),paramdict['train_size'])
train_outputs = dataProc.compute_output_arrays(Datas, 'label')
test1_inputs = get_split(convert_inputs('/home/data_ti4_c/zongwz/CLIR/en-zh_test1_inputs.pkl'),paramdict['test_size'])
test1_outputs = dataProc.compute_output_arrays(Test1, 'label')
test2_inputs = get_split(convert_inputs('/home/data_ti4_c/zongwz/CLIR/en-zh_test2_inputs.pkl'),paramdict['test_size'])
test2_outputs = dataProc.compute_output_arrays(Test2, 'label')
dev_inputs = get_split(convert_inputs('/home/data_ti4_c/zongwz/CLIR/en-zh_dev_inputs.pkl'),paramdict['test_size'])
dev_outputs = dataProc.compute_output_arrays(Dev, 'label')

exfile = '/home/data_ti4_c/zongwz/CLIR/q_d_dict100.pkl'
exfilepkl = '/home/zongwz/ex_q&docs_inputs.pkl'
ex_docs = pd.DataFrame(dataLoader.read_q_d(exfile))
ex_docs['label'] = ex_docs['label'].astype('float')
ex_docs_inputs = ''
if os.path.exists(exfilepkl):
    ex_docs_inputs = pickle.load(open(exfilepkl,'rb'))
else:
    ex_docs_inputs = dataProc.compute_input_arrays(ex_docs, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    pickle.dump(ex_docs_inputs, open(exfilepkl, 'wb'))
ex_query2idx = {}
ex_query2idx_end = {}
queryslist = list(ex_docs['query'])
lastquery = queryslist[0]
for idx in range(len(queryslist)-1):
    if(queryslist[idx]!=queryslist[idx+1]):
        ex_query2idx[queryslist[idx]] = idx+1
        ex_query2idx_end[lastquery] = idx
        lastquery = queryslist[idx+1]
ex_query2idx_end[lastquery] = len(queryslist)-1



if(paramdict['model']=='base'):
    model = Models.BertMatch()
elif(paramdict['model']=='model1'):
    model = Models.BertMatch1()
elif(paramdict['model']=='model2'):
    model = Models.BertMatch2()
model = model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
train_metric = mc.Metrics()
ranking_criterion = nn.MarginRankingLoss()


def run_model():
    for epoch in range(epoch_num):
        print('epoch:{}'.format(epoch))
        train_metric = mc.Metrics()
        test1_metric = mc.Metrics()
        test2_metric = mc.Metrics()
        dev_metric = mc.Metrics()
        if param == 'MSE':
            one_epoch(train_metric, 0, train_inputs, train_outputs, train_batch_size,train_docs)
            one_epoch(dev_metric, 1, dev_inputs, dev_outputs, test_batch_size,test1_docs)
            one_epoch(test1_metric, 1, test1_inputs, test1_outputs, test_batch_size,test2_docs)
            one_epoch(test2_metric, 1, test2_inputs, test2_outputs, test_batch_size,dev_docs)

        elif param == 'ranking':
            ranking_one_epoch(train_metric, 0, train_inputs, train_outputs, train_batch_size,train_docs)
            ranking_one_epoch(dev_metric, 1, dev_inputs, dev_outputs,test_batch_size,test1_docs)
            ranking_one_epoch(test1_metric, 1, test1_inputs, test1_outputs, test_batch_size,test2_docs)
            ranking_one_epoch(test2_metric,1,test2_inputs,test2_outputs,test_batch_size,dev_docs)

        valid_acc, valid_loss = dev_metric.compute()
        train_acc, train_loss = train_metric.compute()
        test1_acc, test1_loss = test1_metric.compute()
        test2_acc, test2_loss = test2_metric.compute()

        s = '\t'.join(
            [str(epoch),str(train_acc), str(train_loss), str(valid_acc), str(valid_loss), str(test1_acc), str(test1_loss),
             str(test2_acc), str(test2_loss)])
        outfile.write(s + '\n')
        print('train:ndcg100:%s  loss:%s' % (train_acc, train_loss))
        print('valid:ndcg100:%s  loss:%s' % (valid_acc, valid_loss))
        print('test1:ndcg100:%s  loss:%s' % (test1_acc, test1_loss))
        print('test2:ndcg100:%s  loss:%s' % (test2_acc, test2_loss))

def get_ex_docs(queryidl,queryidr,batch_size):
    
    tmp = [[[]for j in range(3)]for i in range(paramdict['exout_num'])]#(exout_num*3*batchsize)
    for x in range(paramdict['exout_num']):
        curid = queryidl + int((queryidr - queryidl) / paramdict['exout_num'] * x)
        tmp[x][0] = [ex_docs_inputs[0][curid] for i in range(batch_size)]
        tmp[x][1] = [ex_docs_inputs[1][curid] for i in range(batch_size)]
        tmp[x][2] = [ex_docs_inputs[2][curid] for i in range(batch_size)]
    return tmp

def one_epoch(metric, state, inputs, labels, batch_size, docs_in):
    if state == 0:
        model.train()
    else:
        model.eval()
    loss = ''
    # print(inputs[:10])
    # print(inputs[0])
    model_output_tot = []
    querys = list(Datas['query'])
    for cnt in tqdm(range(0, len(inputs[0]), 100)):
        idx = range(cnt, cnt + 100)
        queryidl = ex_query2idx[querys[cnt]]
        queryidr = ex_query2idx_end[querys[cnt]]
        nidx = cnt
        while labels[nidx] > 0: nidx += 1
        nidx = range(nidx, cnt + 100)
        loss = ''
        losssum = 0.0
        for cnt1 in range(cnt, cnt + 100, batch_size):
            # print(cnt1)
            sub_idx = list(range(cnt1, cnt1 + batch_size))
            target = torch.FloatTensor(labels[sub_idx]).to(device)
            simples = torch.LongTensor([inputs[i][sub_idx] for i in range(3)]).to(device)
            ex_docs_In = torch.LongTensor(get_ex_docs(queryidl,queryidr,batch_size)).to(device)
            doc_In = torch.LongTensor([docs_in[i][sub_idx] for i in range(3)]).to(device)
            if state == 0:

                model_output = 6.0 * model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2],ex_docs_In=ex_docs_In,doc_In = doc_In)
                loss = criterion(model_output.view(-1), target)
                losssum += loss
                loss.backward()
                optim.step()
                optim.zero_grad()
            else:
                with torch.no_grad():
                    model_output = 6.0 * model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2],ex_docs_In=ex_docs_In,doc_In = doc_In)
                    loss = criterion(model_output.view(-1), target)
                    losssum += loss
            # print(model_output)

            model_output_tot += model_output.view(-1).tolist()
        model_output_tot = [[labels[x], model_output_tot[x - cnt]] for x in idx]
        tmplist = sorted(model_output_tot, key=lambda x: x[1], reverse=True)
        tmplist = [tmplist[x][0] for x in range(len(tmplist))]
        metric.add_arg(1.0, losssum, labels[idx], np.array(tmplist))
        model_output_tot = []


def get_rand(l, r):
    return random.randint(l, r)


def ranking_one_epoch(metric, state, inputs, labels, batch_size,docs_in):
    if state == 0:
        model.train()
    else:
        model.eval()
    loss = ''
    querys = list(Datas['query'])
    # print(inputs[:10])
    # print(inputs[0])
    model_output_tot = []
    label_output_tot = []
    for cnt in tqdm(range(0, len(inputs[0]), 100)):
        idx = range(cnt, cnt + 100)
        queryidl = ex_query2idx[querys[cnt]]
        queryidr = ex_query2idx_end[querys[cnt]]
        nidx = cnt
        while labels[nidx] > 0 and nidx < cnt + 100: nidx += 1
        if nidx == cnt + 100: nidx -= 1
        Pidx = range(cnt, nidx)
        Nidx = range(nidx, cnt + 100)
        totnum = 0
        loss = ''
        losssum = 0.0
        if state == 0:
            for cnt1 in Pidx:
                totnum += 2
                
                sub_idx = [cnt1, get_rand(nidx, cnt + 99 if nidx < cnt + 99 else cnt + 199)]
                target = torch.FloatTensor(labels[sub_idx]).to(device)
                simples = torch.LongTensor([inputs[i][sub_idx] for i in range(3)]).to(device)
                doc_In = torch.LongTensor([docs_in[i][sub_idx] for i in range(3)]).to(device)
                ex_docs_In = torch.LongTensor(get_ex_docs(queryidl,queryidr,2)).to(device)
                model_output = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2],ex_docs_In=ex_docs_In,doc_In = doc_In)
                loss = ranking_criterion(model_output[0].view(-1) / target[0], model_output[1].view(-1),
                                         torch.LongTensor([1]).to(device))
                losssum += loss
                loss.backward()
                optim.step()
                optim.zero_grad()
                model_output_tot += model_output.view(-1).tolist()
                label_output_tot += target.view(-1).tolist()
        else:
            for cnt1 in range(cnt, cnt + 100, batch_size):
                with torch.no_grad():
                    sub_idx = list(range(cnt1, cnt1 + batch_size))
                    target = torch.FloatTensor(labels[sub_idx]).to(device)
                    simples = torch.LongTensor([inputs[i][sub_idx] for i in range(3)]).to(device)
                    doc_In = torch.LongTensor([docs_in[i][sub_idx] for i in range(3)]).to(device)
                    ex_docs_In = torch.LongTensor(get_ex_docs(queryidl,queryidr,batch_size)).to(device)
                    model_output = model(input_ids=simples[0], attention_mask=simples[1], token_type_ids=simples[2],ex_docs_In=ex_docs_In,doc_In = doc_In)
                    loss = ranking_criterion(model_output[0].view(-1) / target[0], model_output[1].view(-1),
                                             torch.LongTensor([1]).to(device))
                    losssum += loss
                    model_output_tot += model_output.view(-1).tolist()
                    label_output_tot += target.view(-1).tolist()
        # print(model_output)

        model_output_tot = [[label_output_tot[x], model_output_tot[x]] for x in range(len(model_output_tot))]
        tmplistT = sorted(model_output_tot, key=lambda x: x[0], reverse=True)
        tmplistT = [x[0] for x in tmplistT]
        tmplistP = sorted(model_output_tot, key=lambda x: x[1], reverse=True)
        tmplistP = [x[0] for x in tmplistP]
        metric.add_arg(1.0, losssum, np.array(tmplistT), np.array(tmplistP))
        # print(tmplistT)
        # print(tmplistP)
        model_output_tot = []
        label_output_tot = []


run_model()
outfile.close()
# torch.save(model.state_dict(), r'/home/data_ti4_c/zongwz/data_en_de/parameter.pkl')