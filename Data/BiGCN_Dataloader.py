import sys, nltk, dgl, torch, random, os, pickle
sys.path.append("..")
sys.path.append("../..")
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from typing import List
import numpy as np
from tqdm import tqdm
from functools import reduce
from Data.BiGCN_Data_Utils.twitterloader import BiGCNTwitterSet, CustomDataset

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

    # def __getitem__(self, index):
    #     d_ID = self.data_ID[index]
    #     if not hasattr(self, "g_TD"):
    #         self.g_TD = {}
    #     if not hasattr(self, "g_BU"):
    #         self.g_BU = {}
    #     if not hasattr(self, "lemma_text"):
    #         self.lemma_text = {}
    #
    #     if d_ID in self.g_TD and d_ID in self.g_BU:
    #         g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
    #     else:
    #         g_TD, g_BU = self.construct_graph(index, d_ID)
    #         self.g_TD[d_ID] = g_TD
    #         self.g_BU[d_ID] = g_BU
    #     tf_idf_arr = self.data[d_ID]['tf-idf'].toarray()
    #     tf_idf = torch.tensor(tf_idf_arr,
    #                           device=self.device,
    #                           dtype=torch.float32)[:self.data_len[index]]
    #     return (tf_idf, dgl.add_self_loop(g_TD), \
    #            dgl.add_self_loop(g_BU), \
    #            self.data_y[index], \
    #            self.data[self.data_ID[index]]['topic_label'])

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

        if index in self.lemma_text:
            seq = self.lemma_text[index]
        else:
            seq = [" ".join(self.lemma(self.data[self.data_ID[index]]['text'][j])) for j in range(self.data_len[index])]
            self.lemma_text[index] = seq

        assert len(seq) == g_TD.num_nodes() and len(seq) == g_TD.num_nodes()
        return (seq, dgl.add_self_loop(g_TD), \
               dgl.add_self_loop(g_BU), \
               self.data_y[index], \
               self.data[self.data_ID[index]]['topic_label'])

def Merge_data(data_set1, data_set2):
    new_data = data_set1.__class__()
    new_data.data = dict(data_set1.data, **data_set2.data)
    new_data.data_ID = np.concatenate([np.array(data_set1.data_ID),
                              np.array(data_set2.data_ID)]).tolist()
    new_data.data_len = np.concatenate([np.array(data_set1.data_len),
                            np.array(data_set2.data_len)]).tolist()
    new_data.data_y = np.concatenate([np.array(data_set1.data_y),
                            np.array(data_set2.data_y)]).tolist()
    return new_data

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

def load_data_and_TfidfVectorizer(source_events:List, target_events:List, lt=0, unlabeled_ratio=0.5):
    data_list = load_data(source_events, target_events, lt=0, unlabeled_ratio=-1)
    Tf_Idf_twitter_file = os.path.join(os.environ['DaMSTF_PATH'], "Caches/TfIdf_twitter.pkl")
    if os.path.exists(Tf_Idf_twitter_file):
        with open(Tf_Idf_twitter_file, "rb") as fr:
            tv = pickle.load(fr)
    else:
        lemma = Lemma_Factory()
        corpus = [" ".join(lemma(txt)) for data in [dddata for dddata in data_list if dddata is not None]
                                            for ID in data.data_ID for txt in data.data[ID]['text']]
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        _ = tv.fit_transform(corpus)
        with open(Tf_Idf_twitter_file, "wb") as fw:
            pickle.dump(tv, fw, protocol=pickle.HIGHEST_PROTOCOL)
    data_len = [str(len(dset)) if dset is not None else '0' for dset in data_list]
    print("data length : ", '/'.join(data_len))
    for idx in range(len(data_list)):
        if data_list[idx] is not None:
            data_list[idx].initTFIDF(tv)
    return (tv, ) + tuple(data_list)