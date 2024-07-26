import torch, dgl
import torch.nn as nn
from prefetch_generator import background
from BaseModel.BiGCN_Utils.Sent2Vec import TFIDFBasedVecV2
from BaseModel.BiGCN_Utils.PropModel import BiGCNV2
from Data.BiGCN_Dataloader import FastBiGCNDataset
from BaseModel.BiGCN_Utils.RumorDetectionBasic import BaseEvaluator
from BaseModel.BiGCN_Utils.GraphRumorDect import BiGCNRumorDetecV2


class TwitterBiGCN(BiGCNRumorDetecV2):
    def __init__(self, sent2vec, prop_model, rdm_cls, **kwargs):
        super(TwitterBiGCN, self).__init__(sent2vec, prop_model, rdm_cls)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

def obtain_BiGCN(pretrained_vectorizer, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    lvec = TFIDFBasedVecV2(pretrained_vectorizer, 20,
                         embedding_size=100,
                         w2v_dir="../../glove_en/",
                         emb_update=True,
                         grad_preserve=True).to(device)
    prop = BiGCNV2(100, 256).to(device)
    cls = nn.Linear(1024, 2).to(device)
    BiGCN_model = TwitterBiGCN(lvec, prop, cls)
    return BiGCN_model

class BiGCNEvaluator(BaseEvaluator):
    def __init__(self, dataset:FastBiGCNDataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labelTensor = dataset.labelTensor()

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        tfidf_arr = torch.cat(
            [item[0] for item in items],
            dim=0
        )
        TD_graphs = [item[1] for item in items]
        BU_graphs = [item[2] for item in items]
        labels = [item[3] for item in items]
        topic_labels = [item[4] for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        return tfidf_arr, num_nodes, A_TD, A_BU, \
               torch.tensor(labels), torch.tensor(topic_labels)

    def dataset2dataloader(self):
        idxs = [*range(len(self.dataset))]
        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(self.dataset), self.batch_size):
                batch_idxs = idxs[start:min(start + self.batch_size, len(self.dataset))]
                items = [self.dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)
        return dataloader()