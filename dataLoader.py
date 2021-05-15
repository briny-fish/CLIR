import pandas as pd


def readTrainDatas():
    cnt = 0
    datasdict = {'query': [], 'doc': [], 'label': []}
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/en-de_train_split.txt', encoding='utf-8'):
        tmp = line.split('\t')
        if (len(tmp) < 3): continue
        cnt += 1
        if (cnt == 50000): break
        datasdict['label'].append(int(int(tmp[0]) > 0))

        datasdict['query'].append(tmp[1])
        datasdict['doc'].append(tmp[2])
    print('traindata')
    print(pd.DataFrame(datasdict).shape)
    return datasdict


def readTestDatas():
    cnt = 0
    datasdict = {'query': [], 'doc': [], 'label': []}
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/200en-de_test0.txt', encoding='utf-8'):
        tmp = line.split('\t')
        if (len(tmp) < 3): continue
        cnt += 1
        if (cnt >= 50000): break
        datasdict['label'].append(int(int(tmp[0]) > 0))
        datasdict['query'].append(tmp[1])
        datasdict['doc'].append(tmp[2])
    print('testdata')
    print(pd.DataFrame(datasdict).shape)

    return datasdict


def readDevDatas():
    datasdict = {'query': [], 'doc': [], 'label': []}
    cnt = 0
    for line in open(r'/home/data_ti4_c/zongwz/data_en_de/en-de_dev_split.txt', encoding='utf-8'):
        tmp = line.split('\t')
        if (len(tmp) < 3): continue
        cnt += 1
        # if(cnt==50000):break
        datasdict['label'].append(int(int(tmp[0]) > 0))
        datasdict['query'].append(tmp[1])
        datasdict['doc'].append(tmp[2])
    print('devdata')
    print(pd.DataFrame(datasdict).shape)
    return datasdict


def read_enzh(file):
    print(file)
    datasdict = {'query': [], 'doc': [], 'label': []}
    cnt = 0
    for line in open(file, encoding='utf-8'):
        tmp = line.split('\t')
        cnt += 1
        # if(cnt==50000):break
        datasdict['label'].append(int(tmp[0]))
        datasdict['query'].append(tmp[1])
        datasdict['doc'].append(tmp[2][:-1])
    print(pd.DataFrame(datasdict).shape)
    return datasdict


if __name__ == '__main__':
    read_enzh('en-zh.dev')