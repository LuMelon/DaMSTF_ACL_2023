import sys
from BaseModel.BiGCN_Utils.RumorDetectionBasic import RumorDetection
import torch, torch.nn.functional as F, torch.nn as nn
from BaseModel.BiGCN_Utils.PropModel import BiGCN, BiGCNV2
from BaseModel.BiGCN_Utils.Sent2Vec import TFIDFBasedVecV2
from typing import List

class RvNNRumorDetec(RumorDetection):
    def Batch2Vecs(self, batch):
        trees, seqs = batch[1], batch[0]
        inputs = [self.sent2vec(sents) for sents in seqs]
        input_tensors = torch.cat(inputs)
        seq_outs = self.prop_model(trees, input_tensors)
        return seq_outs

class TransformerRumorDetec(RvNNRumorDetec):
    def Batch2Vecs(self, batch):
        trees, seqs = batch[1], batch[0]
        inputs = [self.sent2vec(sents) for sents in seqs]
        seq_outs = self.prop_model(trees, inputs)
        return seq_outs



class BiGCNRumorDetec(RumorDetection):
    def __init__(self, sent2vec:TFIDFBasedVecV2, prop_model:BiGCN, rdm_cls:nn.Module, **kwargs):
        super(BiGCNRumorDetec, self).__init__(sent2vec, prop_model, rdm_cls)
        self.sent2vec = sent2vec
        self.prop_model = prop_model
        self.rdm_cls = rdm_cls
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

    def Batch2Vecs(self, batch):
        TD_graphs, BU_graphs, seqs =batch[1], batch[2], batch[0]
        all_sents = [sent for sents in seqs for sent in sents]
        # inputs = [self.sent2vec(sents) for sents in seqs]
        inputs = self.sent2vec(all_sents)
        seq_outs = self.prop_model(TD_graphs, BU_graphs, inputs)
        return seq_outs

    def AugBatch2Vecs(self, batch):
        TD_graphs, BU_graphs, seqs =batch[1], batch[2], batch[0]
        all_sents = [sent for sents in seqs for sent in sents]
        # inputs = [self.sent2vec(sents) for sents in seqs]
        inputs = self.sent2vec.AugForward(all_sents)
        seq_outs = self.prop_model(TD_graphs, BU_graphs, inputs)
        return seq_outs

    def AugPredict(self, batch):
        seq_outs = self.AugBatch2Vecs(batch)
        preds = self.rdm_cls(seq_outs).softmax(dim=1)
        return preds

class BiGCNRumorDetecV2(BiGCNRumorDetec):
    def __init__(self, sent2vec:TFIDFBasedVecV2, prop_model:BiGCNV2, rdm_cls:nn.Module, **kwargs):
        super(BiGCNRumorDetecV2, self).__init__(sent2vec, prop_model, rdm_cls)
        self.sent2vec = sent2vec
        self.prop_model = prop_model
        self.rdm_cls = rdm_cls
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

    def Batch2Vecs(self, batch):
        tfidf:torch.Tensor = batch[0]
        num_nodes:List = batch[1]
        A_TD:torch.Tensor = batch[2].bool().float()
        A_BU:torch.Tensor = batch[3].bool().float()
        inputs = self.sent2vec(tfidf)
        seq_outs = self.prop_model(num_nodes, A_TD, A_BU, inputs)
        return seq_outs


    def AugBatch2Vecs(self, batch):
        tfidf:torch.Tensor = batch[0]
        num_nodes:List = batch[1]
        A_TD:torch.Tensor = batch[2].bool().float()
        A_BU:torch.Tensor = batch[3].bool().float()

        # inputs = [self.sent2vec(sents) for sents in seqs]
        inputs = self.sent2vec.AugForward(tfidf)
        seq_outs = self.prop_model(num_nodes, A_TD, A_BU, inputs)
        return seq_outs
