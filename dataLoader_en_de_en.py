import pandas as pd

def readTrainDatas():
    cnt = 0
    datasdict = {'query': [], 'doc': [], 'label': [], 'lang':[]}
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/en-de+en_train_split.txt',encoding='utf-8'):
        tmp = line.split('\t')
        if(len(tmp)<4):continue
        cnt+=1
        #if(cnt==50000):break
        datasdict['label'].append(int(int(tmp[1])>0))
        datasdict['lang'].append(tmp[0])
        datasdict['query'].append(tmp[2])
        datasdict['doc'].append(tmp[3])

    print('traindata')
    print(pd.DataFrame(datasdict).shape)
    return datasdict
def readTestDatas():
    cnt = 0
    datasdict = {'query': [], 'doc': [], 'label': [], 'lang':[]}
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/200en-de+en_test0.txt',encoding='utf-8'):
        tmp = line.split('\t')
        if(len(tmp)<4):continue
        cnt+=1
        if(cnt>200000):break
        datasdict['label'].append(int(int(tmp[1])>0))
        datasdict['lang'].append(tmp[0])
        datasdict['query'].append(tmp[2])
        datasdict['doc'].append(tmp[3])
    print('testdata')
    print(pd.DataFrame(datasdict).shape)
    return datasdict
def readDevDatas():
    datasdict = {'query': [], 'doc': [], 'label': [], 'lang':[]}
    cnt = 0
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/en-de+en_dev_split.txt',encoding='utf-8'):
        tmp = line.split('\t')
        if(len(tmp)<4):continue
        cnt+=1
        #if(cnt==50000):break
        datasdict['label'].append(int(int(tmp[1])>0))
        datasdict['lang'].append(tmp[0])
        datasdict['query'].append(tmp[2])
        datasdict['doc'].append(tmp[3])
    print('devdata')
    print(pd.DataFrame(datasdict).shape)
    return datasdict
if __name__ == '__main__':
    readTrainDatas()