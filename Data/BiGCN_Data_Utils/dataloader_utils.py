from torch.utils.data import Dataset
import random, os, pickle, sys, re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import torch, torch.nn as nn, numpy as np
sys.path.append("..")
from transformers.models.bert.tokenization_bert import BertTokenizer
from nltk.stem import WordNetLemmatizer

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

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

# def load_anc_cache_events(events_list:List):
#     for event_dir  in events_list:
#         event_set = FastBiGCNDataset()
#         event_set.load_event_list([event_dir])
#         event_set.Caches_Data(event_dir)

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

def Sample_data(dataset, idx_list):
    new_twitter = dataset.__class__()
    new_twitter.data_ID = [dataset.data_ID[idx] for idx in idx_list]
    new_twitter.data_len = [dataset.data_len[idx] for idx in idx_list]
    new_twitter.data_y = [dataset.data_y[idx] for idx in idx_list]
    new_twitter.data = dataset.data
    return new_twitter

def Sort_data(tr_set, dev_set, te_set):
    bigDic = dict(dict(tr_set.data, **dev_set.data), **te_set.data)
    all_IDs = np.concatenate([np.array(tr_set.data_ID),
                              np.array(dev_set.data_ID),
                              np.array(te_set.data_ID)]).tolist()
    all_l = np.concatenate([np.array(tr_set.data_len),
                            np.array(dev_set.data_len),
                            np.array(te_set.data_len)]).tolist()
    all_y = np.concatenate([np.array(tr_set.data_y),
                            np.array(dev_set.data_y),
                            np.array(te_set.data_y)]).tolist()
    tr_len, dev_len, te_len = len(tr_set), len(dev_set), len(te_set)

    source_created_at = np.concatenate([
        np.array([tr_set.data[ID]['created_at'][0] for ID in tr_set.data_ID]),
        np.array([dev_set.data[ID]['created_at'][0] for ID in dev_set.data_ID]),
        np.array([te_set.data[ID]['created_at'][0] for ID in te_set.data_ID]),
    ])

    idxs = source_created_at.argsort()
    tr_ids, dev_ids, te_ids = ([all_IDs[idx] for idx in l] for l in
                               [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])
    tr_y, dev_y, te_y = ([all_y[idx] for idx in l] for l in
                         [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])
    tr_l, dev_l, te_l = ([all_l[idx] for idx in l] for l in
                         [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])

    tr_set.data = {ID: bigDic[ID] for ID in tr_ids}
    dev_set.data = {ID: bigDic[ID] for ID in dev_ids}
    te_set.data = {ID: bigDic[ID] for ID in te_ids}
    tr_set.data_ID = tr_ids
    tr_set.data_len = tr_l
    tr_set.data_y = tr_y

    dev_set.data_ID = dev_ids
    dev_set.data_len = dev_l
    dev_set.data_y = dev_y

    te_set.data_ID = te_ids
    te_set.data_len = te_l
    te_set.data_y = te_y
    return tr_set, dev_set, te_set

def shuffle_data(tr_set, dev_set, te_set):
    bigDic = dict(dict(tr_set.data, **dev_set.data), **te_set.data)
    all_IDs = np.concatenate([np.array(tr_set.data_ID),
                              np.array(dev_set.data_ID),
                              np.array(te_set.data_ID)]).tolist()
    all_l = np.concatenate([np.array(tr_set.data_len),
                              np.array(dev_set.data_len),
                              np.array(te_set.data_len)]).tolist()
    all_y = np.concatenate([np.array(tr_set.data_y),
                              np.array(dev_set.data_y),
                              np.array(te_set.data_y)]).tolist()
    tr_len, dev_len, te_len = len(tr_set), len(dev_set), len(te_set)
    idxs = random.sample(list(range(tr_len + dev_len + te_len)), tr_len+dev_len+te_len)
    tr_ids, dev_ids, te_ids = ([all_IDs[idx] for idx in l] for l in
                               [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])
    tr_y, dev_y, te_y = ([all_y[idx] for idx in l] for l in
                         [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])
    tr_l, dev_l, te_l = ([all_l[idx] for idx in l] for l in
                         [idxs[:tr_len], idxs[tr_len:tr_len + dev_len], idxs[-dev_len:]])

    tr_set.data = {ID:bigDic[ID] for ID in tr_ids }
    dev_set.data = {ID:bigDic[ID] for ID in dev_ids}
    te_set.data = {ID:bigDic[ID] for ID in te_ids}

    tr_set.data_ID = tr_ids
    tr_set.data_len = tr_l
    tr_set.data_y = tr_y

    dev_set.data_ID = dev_ids
    dev_set.data_len = dev_l
    dev_set.data_y = dev_y

    te_set.data_ID = te_ids
    te_set.data_len = te_l
    te_set.data_y = te_y
    return tr_set, dev_set, te_set

def mask_tokens(inputs, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class Tree(object):
    def __init__(self, root_idx=0, edges=None):
        self.parent = None
        self.root_idx = root_idx
        self.num_children = 0
        self.children = list()
        if edges is not None:
            self.Construct(edges)

    def Construct(self, edges):
        parent_trees = [self]
        while parent_trees:
            child_trees = []
            for edge in edges:
                for tree in parent_trees:
                    if edge[0] == tree.root_idx:
                        child_tree = Tree(root_idx=edge[1])
                        tree.add_child(child_tree)
                        child_trees.append(child_tree)
            parent_trees = child_trees

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', False):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth', False):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def nodes(self):
        if getattr(self, "_nodes", False):
            return self._nodes
        if self.num_children > 0:
            self._nodes = [self] + [node for i in range(self.num_children)
                                     for node in self.children[i].nodes()]
        else:
            self._nodes = [self]
        return self._nodes

    def leaf_node_idxs(self):
        if getattr(self, "_leaf_node_idxs", False):
            return self._leaf_node_idxs
        if self.num_children > 0:
            self._leaf_node_idxs = [idx for i in range(self.num_children)
                                    for idx in self.children[i].leaf_node_idxs()]
        else:
            self._leaf_node_idxs = [self.root_idx]
        return self._leaf_node_idxs

class RumorLoader(Dataset):
    def __init__(self):
        super(RumorLoader, self).__init__()
        self.data = {}
        self.data_ID = []
        self.data_len = []
        self.data_y = []
        self.sample_len = -1

    def split(self, percent=[0.5, 1.0]):
        data_size = len(self.data_ID)
        start_end = [int(item*data_size) for item in percent]
        start_end.insert(0, 0)
        new_idxs = list(random.sample(list(range(data_size)), data_size))
        rst = [self.__class__() for _ in percent]
        for i in range(len(percent)):
            rst[i].data_ID = [self.data_ID[idx] for idx in new_idxs[start_end[i]:start_end[i+1]]]
            rst[i].data_len = [self.data_len[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_y = [self.data_y[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data = {ID:self.data[ID] for ID in rst[i].data_ID}
        return rst

    def select(self, idxs):
        obj = self.__class__()
        obj.data_ID = [self.data_ID[idx] for idx in idxs]
        obj.data_len = [self.data_len[idx] for idx in idxs]
        obj.data_y = [self.data_y[idx] for idx in idxs]
        obj.data = {ID:self.data[ID] for ID in self.data_ID}
        return obj

    def BalancedSplit(self, percent=[0.5, 1.0]):
        """
        the first subset is balanced.
        """
        data_size = len(self.data_ID)
        cnt_list = [int(item * data_size) for item in percent]

        indices = torch.arange(data_size)
        label = torch.tensor(self.data_y).argmax(dim=1)
        pos_idxs = indices[label.__eq__(1)].tolist()
        neg_idxs = indices[label.__eq__(0)].tolist()
        random.shuffle(pos_idxs), random.shuffle(neg_idxs)
        idxs_list = [pos_idxs[:cnt_list[0] // 2] + neg_idxs[:cnt_list[0] - cnt_list[0] // 2]]
        rest_idxs = pos_idxs[cnt_list[0] // 2:] + neg_idxs[cnt_list[0] - cnt_list[0] // 2:]
        random.shuffle(rest_idxs)
        for i in range(1, len(cnt_list)):
            idxs_list.append(rest_idxs[cnt_list[i-1]:cnt_list[i]])
        rst = [self.__class__() for _ in cnt_list]
        for i in range(cnt_list):
            rst[i].data_ID = [self.data_ID[idx] for idx in idxs_list[i]]
            rst[i].data_len = [self.data_len[idx] for idx in idxs_list[i]]
            rst[i].data_y = [self.data_y[idx] for idx in idxs_list[i]]
            rst[i].data = {ID: self.data[ID] for ID in rst[i].data_ID}
        return rst

    def Caches_Data(self, data_prefix="../data/data"):
        data_dic  = "%s_dict.txt" % data_prefix
        y_npy = "%s_y.npy" % data_prefix
        id_npy = "%s_ID.npy" % data_prefix
        len_npy = "%s_len.npy" % data_prefix
        with open(data_dic, "wb") as fw:
            pickle.dump(self.data, fw, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(id_npy, np.array(self.data_ID))
        np.save(y_npy, np.array(self.data_y))
        np.save(len_npy, np.array(self.data_len))

    def load_data_fast(self, data_prefix="../data/train", min_len=-1):
        dic_file = "%s_dict.txt"%data_prefix
        id_npy = "%s_ID.npy"%data_prefix
        len_npy = "%s_len.npy"%data_prefix
        y_npy = "%s_y.npy"%data_prefix
        assert os.path.exists(dic_file) \
                and os.path.exists(id_npy) \
                    and os.path.exists(len_npy) \
                        and os.path.exists(y_npy)
        with open(dic_file, "rb") as handle:
           self.data = pickle.load(handle)
        self.data_ID = np.load(id_npy).tolist()
        self.data_len = np.load(len_npy).tolist()
        self.data_y = np.load(y_npy).tolist()
        if min_len > 0:
            self.filter_short_seq(min_len)
        print("load len: ", len(self.data_ID))

    def filter_short_seq(self, min_len):
        idxs = [idx for idx, l in enumerate(self.data_len) if l > min_len]
        self.data_ID = [self.data_ID[idx] for idx in idxs]
        self.data_len = [self.data_len[idx] for idx in idxs]
        self.data_y = [self.data_y[idx] for idx in idxs]

    def trim_long_seq(self, max_len):
        self.data_len = [min(l, max_len) for l in self.data_len]

    def ResortSample(self, sample_len=-1):
        try:
            assert sample_len >= min(self.data_len)
        except:
            print("Failed to set the sample_len")
        else:
            self.sample_len = sample_len

    def __len__(self):
        return len(self.data_ID)

    def __getitem__(self, index):
        pass

    def collate_raw_batch(self, batch):
        pass

    def InnerBatch(self, batchsize):
        idxs = random.sample(range(len(self.data_ID)), batchsize)
        batch = [self.__getitem__(idx) for idx in idxs]
        return self.collate_raw_batch(batch)