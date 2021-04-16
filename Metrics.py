import numpy as np
class Metrics(object):
    '''
      包括f1score、acc、recall的计算方法
    '''
    def getDCG(self,scores):
        return np.sum(
            np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
            dtype=np.float32)


    def getNDCG(self,rank_list, pos_items):
        if(len(rank_list)==0):return 0.0
        idcg = getDCG(rank_list)
        dcg = getDCG(pos_items)
        if dcg == 0.0:
            return 0.0
        ndcg = dcg / idcg
        return ndcg

    def __init__(self):
        self._tot_num = 0.0
        self._true_num = 0.0
        self._postive_num = 0.0
        self._true_postive_num = 0.0
        self._acc = 0.0
        self._recall = 0.0
        self._f1 = 0.0
        self._loss = 0.0
        self._Ilist = []
        self._predList = []
        self._nDCG = 0.0

    def add_arg(self, Totnum=0.0, truenum=0.0,Loss=0.0,list1 = [],list2 = []):
        self._tot_num += Totnum
        self._true_num += truenum
        self._loss += Loss
        self._Ilist = list1
        self._predList = list2 
    

    def compute(self):
        self._acc = self._true_num / self._tot_num
        #self._recall = self._true_postive_num / self._postive_num
        #self._f1 = 2 * self._acc * self._recall / (self._acc + self._recall)
        self._loss = self._loss / self._tot_num
        self._nDCG = self.getNDCG(self._Ilist,self._predList)
        return self._acc, self._loss