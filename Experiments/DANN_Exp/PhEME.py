import sys, os, pickle, nltk, dgl, random
sys.path.append("../..")
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import torch, torch.nn as nn
import numpy as np
from typing import List
from tqdm import tqdm
from TrainingEnv import CustomDataset
from functools import reduce
from Data.BiGCN_Dataloader import BiGCNTwitterSet, Merge_data
from BaseModel.BiGCN_Utils.Sent2Vec import TFIDFBasedVecV2
from BaseModel.BiGCN_Utils.PropModel import BiGCNV2
from BaseModel.BiGCN import TwitterBiGCN
from DomainAdaptationTrainer.DANN import DANNTrainer
from TrainingEnv import GradientReversal

def Lemma_Factory():
    lemmatizer = WordNetLemmatizer()
    def lemma(word_tokens):
        tags = nltk.pos_tag(word_tokens)
        new_words = []
        for pair in tags:
            if pair[1].startswith('J'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'a'))
            elif pair[1].startswith('V'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'v'))
            elif pair[1].startswith('N'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'n'))
            elif pair[1].startswith('R'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'r'))
            else:
                new_words.append(pair[0])
        return new_words
    return lemma

class FastBiGCNDataset(BiGCNTwitterSet, CustomDataset):
    def __init__(self, batch_size=20):
        super(FastBiGCNDataset, self).__init__(batch_size=batch_size)

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        return torch.tensor(self.data_y, dtype=torch.float32, device=device)

    def collate_fn(self, items):
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
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        return tfidf_arr, num_nodes, A_TD, A_BU, \
               torch.tensor(labels), torch.tensor(topic_labels)

    def initGraph(self):
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}

        for index, d_ID in tqdm(enumerate(self.data_ID)):
            if d_ID in self.g_TD and d_ID in self.g_BU:
                pass
            else:
                g_TD, g_BU = self.construct_graph(index, d_ID)
                self.g_TD[d_ID] = g_TD
                self.g_BU[d_ID] = g_BU

    def initTFIDF(self, tokenizer:TfidfVectorizer):
        self.initGraph()
        for index, d_ID in tqdm(enumerate(self.data_ID)):
            sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
            try:
                self.data[d_ID]['tf-idf'] = tokenizer.transform(sents)
            except:
                print("sents : \n\t", sents)

    def setLabel(self, label:List, idxs:List):
        for k, idx in enumerate(idxs):
            self.data_y[idx] = label[k]

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}
        if not hasattr(self, "lemma_text"):
            self.lemma_text = {}

        if d_ID in self.g_TD and d_ID in self.g_BU:
            g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
        else:
            g_TD, g_BU = self.construct_graph(index, d_ID)
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU
        tf_idf_arr = self.data[d_ID]['tf-idf'].toarray()
        tf_idf = torch.tensor(tf_idf_arr,
                              device=self.device,
                              dtype=torch.float32)[:self.data_len[index]]
        return (tf_idf, dgl.add_self_loop(g_TD), \
               dgl.add_self_loop(g_BU), \
               self.data_y[index], \
               self.data[self.data_ID[index]]['topic_label'])

class DANN_BiGCN(TwitterBiGCN):
    def AdvDLossAndAcc(self, discriminator:nn.Module, batch, grad_reverse=False):
        f = self.Batch2Vecs(batch)
        if grad_reverse:
            f = GradientReversal.apply(f)
        predictions = discriminator(f)
        epsilon = torch.ones_like(predictions) * 1e-8
        preds = (predictions - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-1]
        loss, acc = self.loss_func(preds, labels, label_weight=None, reduction='mean')
        return loss, acc

def obtain_BiGCN(pretrained_vectorizer, device=None):
    DaMSTF_PATH = os.environ['DaMSTF_PATH']
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    lvec = TFIDFBasedVecV2(pretrained_vectorizer, 20,
                         embedding_size=300,
                         w2v_dir=os.path.join(DaMSTF_PATH, "Caches/glove_en/"),
                         emb_update=True,
                         grad_preserve=True).to(device)
    prop = BiGCNV2(300, 256).to(device)
    cls = nn.Linear(1024, 2).to(device)
    BiGCN_model = DANN_BiGCN(lvec, prop, cls)
    return BiGCN_model

def load_events(events_list:List):
    data_list = []
    for event_dir in events_list:
        dataset = FastBiGCNDataset()
        try:
            dataset.load_data_fast(event_dir)
        except: # if no caches
            dataset.load_event_list([event_dir])
            dataset.Caches_Data(event_dir)
        data_list.append(dataset)

    if len(data_list) > 1:
        final_set = reduce(Merge_data, data_list)
        del dataset, data_list
    elif len(data_list) == 1:
        final_set = data_list[0]
    else:
        raise Exception("Something wrong!")
    return final_set

def load_data(source_events:List, target_events:List, lt=0, unlabeled_ratio=0.5):
    train_set = load_events(source_events)
    target_set = load_events(target_events)

    if lt != 0:
        lt_percent = lt*1.0/len(target_set)
        labeled_target, target_set = target_set.split([lt_percent, 1.0])
    else:
        labeled_target = None
    val_set, test_set = target_set.split([0.2, 1.0])

    if unlabeled_ratio <= 0.0:
        return train_set, labeled_target, val_set, test_set
    else:
        ut_size = int(unlabeled_ratio*len(test_set))
        idxs = random.sample(range(len(test_set)), ut_size)
        unlabeled_target:FastBiGCNDataset = test_set.select(idxs) # unlabeled target is a subset of test_set
        unlabeled_target.setLabel(
            [[0, 0] for _ in range(len(unlabeled_target))],
            [*range(len(unlabeled_target))]
        )
        return train_set, labeled_target, val_set, test_set, unlabeled_target


if __name__ == '__main__':
    # log_dir = str(__file__).rstrip(".py")
    DaMSTF_PATH = os.path.abspath(
        os.path.join(
            os.path.curdir,
            '../..'
        )
    )
    os.environ['DaMSTF_PATH'] = DaMSTF_PATH
    data_dir = os.path.join(DaMSTF_PATH, "../pheme-rnr-dataset/")
    events_list = ['charliehebdo',  'ferguson',  'germanwings-crash',  'ottawashooting',  'sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 0
    source_events = [os.path.join(data_dir, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    target_events = [os.path.join(data_dir, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]

    fewShotCnt = 20
    tr, lb_tgt, dev, te = load_data(source_events, target_events, lt=fewShotCnt, unlabeled_ratio=-1)

    log_dir = f"./{test_event_name}/"
    print("%s : (dev event)/(test event)/(train event) = %3d/%3d/%3d" % (
    test_event_name, len(dev), len(te), len(tr)))
    print("\n\n===========%s Train===========\n\n"%te.data[te.data_ID[0]]['event'])
    Tf_Idf_twitter_file = "./saved/TfIdf_twitter.pkl"
    if os.path.exists(Tf_Idf_twitter_file):
        with open(Tf_Idf_twitter_file, "rb") as fr:
            tv = pickle.load(fr)
    else:
        lemma = Lemma_Factory()
        corpus = [" ".join(lemma(txt)) for data in [tr, dev, te] for ID in data.data_ID for txt in
                  data.data[ID]['text']]
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        _ = tv.fit_transform(corpus)
        with open(Tf_Idf_twitter_file, "wb") as fw:
            pickle.dump(tv, fw, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"domian_ID = {domain_ID} : ")
    print(f"\t tr : ", tr.labelTensor().sum(dim=0))
    print(f"\t dev : ", dev.labelTensor().sum(dim=0))
    print(f"\t te : ", te.labelTensor().sum(dim=0))
    tr.initTFIDF(tv)
    dev.initTFIDF(tv)
    te.initTFIDF(tv)
    model1 = obtain_BiGCN(tv)
    trainer = DANNTrainer(random_seed=10086, log_dir=log_dir, suffix=f"{test_event_name}_FS{fewShotCnt}",
                            model_file=f"./DANN_{test_event_name}_FS{fewShotCnt}.pkl",
                                class_num=model1.num_classes, temperature=0.05,
                                    batch_size=28, Lambda=0.1)
    trainer.collate_fn = dev.collate_raw_batch

    # Pretrain Task-specific Model
    if os.path.exists(f"./PreTrainClassifier_T{test_event_name}.pkl"):
        model1.load_state_dict(
            torch.load(f"./PreTrainClassifier_T{test_event_name}.pkl")
        )
    else:
        trainer.training(model1, tr, batch_size=28, max_epoch=20, lr4model=5e-3,
                    dev_evaluator=None, test_evaluator=None,
                        grad_accum_cnt=1, valid_every=100, model_path="./PreTrainClassifier_T{test_event_name}.pkl")
        if os.path.exists(f"./PreTrainClassifier_T{test_event_name}.pkl"):
            model1.load_state_dict(
                torch.load(f"./PreTrainClassifier_T{test_event_name}.pkl")
            )
        else:
            torch.save(model1.state_dict(),
                       f"./PreTrainClassifier_T{test_event_name}.pkl")
    TestLabel = te.labelTensor().clone()
    te.setLabel(np.zeros_like(TestLabel.numpy()).tolist(), np.arange(len(TestLabel)).tolist())
    trainer.valid(model1, dev, dev.labelTensor(), f"{trainer.suffix}_valid", 0)
    trainer.valid(model1, te, TestLabel, f"{trainer.suffix}_test", 0)
    trainer.ModelTrain(model1, tr, lb_tgt, te, dev, te,
                         maxEpoch=40, validEvery=100, test_label=TestLabel)

