import json, re, pickle, torch, pkuseg, random, dgl
import numpy as np, pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from .dataloader_utils import Tree, RumorLoader

# with open("config.json", "r") as cr:
#     dic = json.load(cr)
# FLAGS = adict(dic)
topics =  ['科技', '政治', '军事', '财经商业', '社会生活', '文体娱乐', '医药健康', '教育考试']
category_dic = {
             '科技': 0,
             '政治': 1,
             '军事': 2,
             '财经商业': 3,
             '社会生活': 4,
             '文体娱乐': 5,
             '医药健康': 6,
             '教育考试': 7
            }
#各类别数目
# 0	65
# 1	280  --> dev
# 2	95
# 3	101
# 4	2899
# 5	805 --> dev
# 6	330 --> dev
# 7	89

class TopicReader(Dataset):
    def __init__(self, data_csv_file):
        df = pd.read_csv(data_csv_file)
        self.sents = [self.lineClear(line) for line in df['content'].values.tolist()]
        self.label = [category_dic[item] for item in df['category'].values.tolist()]

    def lineClear(self, line):
        line = line.lower().strip("\t").strip("\n")
        line = re.sub("@[^ :]*", " @ ", line)
        line = re.sub("#[^# ]*#", " # ", line)
        line = re.sub("http(.?)://[^ ]*", " url ", line)
        return line

    def collate_raw_batch(self, batch):
        sents = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return sents, torch.tensor(labels)

    def __getitem__(self, index):
        return self.sents[index], self.label[index]

    def __len__(self):
        return len(self.sents)

class SubReader(Dataset):
    def __init__(self, data_file):
        df = pd.read_csv(data_file)
        self.label = df['label'].values.tolist()
        self.sents = [self.lineClear(line) for line in df['content'].values]
        self.idxs = list(range(len(self.label)))

    def lineClear(self, line):
        line.lower()
        line = re.sub("@[^ :]*", " @ ", line)
        line = re.sub("#[^ ]*#", " # ", line)
        line = re.sub("http(.?)://[^ ]*", " url ", line)
        return line

    def sample(self, batch_size):
        idxs = random.sample(self.idxs, batch_size)
        return self.sents[idxs], torch.tensor(self.label[idxs])

    def collate_raw_batch(self, batch):
        sents = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return sents, torch.tensor(labels)

    def __getitem__(self, index):
        return self.sents[index], self.label[index]

    def __len__(self):
        return len(self.label)

class WeiboLoader(RumorLoader):
    def __init__(self, max_seq_len=20, min_seq_len=5):
        super(WeiboLoader, self).__init__()
        userdic = ['[', ']', '。', ',', '，', '{', '}', '(', ')', '!', '！', '~', '～']
        self.seg = pkuseg.pkuseg(user_dict=userdic)
        self.batch_size = 20
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

    def lineClear(self, line):
        line.lower()
        line = re.sub("@[^ :]*", " @ ", line)
        line = re.sub("#[^ ]*#", " # ", line)
        line = re.sub("http(.?)://[^ ]*", " url ", line)
        return line

    def transIrregularWord(self, line):
        if not line:
            return ''
        line = self.lineClear(line)
        return self.seg.cut(line)

    def read_json(self, fname, ID, event_data=None):
        if event_data is None:
            with open(fname) as fr:
                event_data = json.load(fr)
        texts = [tweet['original_text'] for tweet in event_data]
        created_at = [tweet['t'] for tweet in event_data]
        IDs = [tweet['id'] for tweet in event_data]
        parents = [tweet['parent'] for tweet in event_data]
        idxs = np.array(created_at).argsort().tolist()
        if len(idxs)>self.max_seq_len:
            idxs = idxs[:self.max_seq_len]
        self.data[ID] = {
            'sentence':[texts[idx] for idx in idxs],
            'text':[self.transIrregularWord(texts[idx]) for idx in idxs],
            'created_at':[created_at[idx] for idx in idxs],
            "tweet_id":[IDs[idx] for idx in idxs],
            "reply_to": [parents[idx] for idx in idxs]
         }
        self.data_len.append(min(len(texts), self.max_seq_len))

    def load_data(self, weibo_dir="/home/hadoop/Rumdect/Weibo",
                  weibo_file="../data/dev_ids.csv",
                  weibo_df=None, cached_prefix=None):
        if weibo_df is None:
            vals = pd.read_csv(weibo_file).values.transpose(1, 0)
        else:
            vals = weibo_df.values.transpose(1, 0)
        topic_labels = vals[2].tolist()
        self.data_y = [[0, 1] if y_label == 1 else [1, 0]
                            for y_label in vals[1].tolist()]
        self.data_ID = [str(ID) for ID in vals[0].tolist()]
        for idx, ID in enumerate(tqdm(self.data_ID)):
            fname = "%s/%s.json"%(weibo_dir, ID)
            self.read_json(fname, ID)
            self.data[ID]['topic_label'] = topic_labels[idx]
        if cached_prefix is not None:
            self.Caches_Data(cached_prefix)

    def load_events(self, vals, weibo_caches=None, cached_prefix=None): #"../data/weibo.pkl"
        self.data_ID = [str(ID) for ID in vals[0].tolist()]
        self.data_y = vals[1].tolist()
        if weibo_caches is None:
            weibo_dir = "../data/Rumdect/Weibo"
            for ID in tqdm(self.data_ID):
                fname = "%s/%s.json" % (weibo_dir, ID)
                self.read_json(fname, ID)
        else:
            with open(weibo_caches, 'rb') as fr:
                caches = pickle.load(fr)
            for ID in tqdm(self.data_ID):
                self.read_json('', str(ID), event_data=caches[str(ID)])

        if cached_prefix is not None:
            self.Caches_Data(cached_prefix)

class WeiboSet(WeiboLoader):
    def __init__(self, max_seq_len=20,  min_seq_len=5):
        super(WeiboSet, self).__init__(max_seq_len=max_seq_len, min_seq_len=min_seq_len)

    @property
    def label(self):
        if isinstance(self.data_y, list):
            self.data_y = np.array(self.data_y)
        return self.data_y

    def setLabel(self, label, idxs):
        if isinstance(self.data_y, list):
            self.data_y = np.array(self.data_y)
        self.data_y[idxs] = label

    def labelTensor(self, device=None):
        return torch.tensor(self.data_y, device=device)

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        lens = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        topic_labels = [item[3] for item in batch]
        return seqs, torch.tensor(lens), torch.tensor(labels), torch.tensor(topic_labels)

    def __getitem__(self, index):
        if self.sample_len != -1:
            tmp_seq = [" ".join(self.data[self.data_ID[index]]['text'][j]) for j in range(self.data_len[index])]
            new_len = min(self.sample_len, len(tmp_seq))
            seq = tmp_seq[0:1] + [tmp_seq[idx] for idx in
                                  np.sort(random.sample(list(range(1, len(tmp_seq))), new_len - 1))]
            return seq, len(seq), self.data_y[index], self.data[self.data_ID[index]]['topic_label']
        else:
            seq = [" ".join(self.data[self.data_ID[index]]['text'][j]) for j in range(self.data_len[index])]
            return seq, self.data_len[index], self.data_y[index], self.data[self.data_ID[index]]['topic_label']

    def __len__(self):
        return len(self.data_ID)

class CAMI_WeiboSet(WeiboSet):
    def __init__(self, max_seq_len=10000, min_seq_len=1, fixed_seq_len=20):
        super(CAMI_WeiboSet, self).__init__(max_seq_len=max_seq_len,  min_seq_len=min_seq_len)
        self.fixed_seq_len = fixed_seq_len

    def __getitem__(self, index):
        gap = self.data_len[index]//self.fixed_seq_len
        gap = 1 if gap == 0 else gap
        seq = []
        for j in range(0, self.data_len[index], gap):
            s = ""
            for i in range(j, j+gap, 1):
                s = s + " ".join(self.data[self.data_ID[index]]['text'][j])
            seq.append(s)
        return seq, len(seq), self.data_y[index], self.data[self.data_ID[index]]['topic_label']


class GraphWeiboSet(WeiboSet):
    def __init__(self, max_seq_len=20,  min_seq_len=5):
        super(GraphWeiboSet, self).__init__(max_seq_len=max_seq_len,  min_seq_len=min_seq_len)

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        graphs = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        topic_labels = [item[3] for item in batch]
        return seqs, graphs, torch.tensor(labels).argmax(dim=1), torch.tensor(topic_labels)

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        tIds_dic = {ID: idx for idx, ID in enumerate(self.data[d_ID]["tweet_id"])}
        src = np.arange(0, len(self.data[d_ID]["tweet_id"]), 1)
        dst = np.array([tIds_dic[ID] for ID in self.data[d_ID]["reply_to"][1:]])
        g = dgl.graph((src[1:], dst), num_nodes=len(src))
        g = dgl.to_bidirected(g, readonly=False)
        seq = [" ".join(self.data[self.data_ID[index]]['text'][j]) for j in range(self.data_len[index])]
        return seq, g, self.data_y[index], self.data[self.data_ID[index]]['topic_label']

class TreeWeiboSet(WeiboSet):
    def __init__(self, max_seq_len=20,  min_seq_len=5):
        super(TreeWeiboSet, self).__init__(max_seq_len=max_seq_len,  min_seq_len=min_seq_len)

    def init_trees(self):
        self.data_trees = []
        for index, d_ID in enumerate(tqdm(self.data_ID)):
            tIds_dic = {}
            dup_cnt = 0
            dup_idxs = []
            for idx, ID in enumerate(self.data[d_ID]["tweet_id"]):
                if ID in tIds_dic:
                    self.data_len[index] -= 1
                    dup_cnt += 1
                    dup_idxs.append(idx)
                else:
                    tIds_dic[ID] = idx - dup_cnt
            for i, idx in enumerate(dup_idxs):
                self.data[d_ID]["tweet_id"].pop(idx - i)
                self.data[d_ID]["reply_to"].pop(idx - i)
                self.data[d_ID]["text"].pop(idx - i)
                self.data[d_ID]["sentence"].pop(idx - i)
                self.data[d_ID]["created_at"].pop(idx - i)
            edges = [(tIds_dic[dst_ID], src_idx+1) if dst_ID in tIds_dic else (0, src_idx+1)
                     for src_idx, dst_ID in enumerate(self.data[d_ID]["reply_to"][1:self.data_len[index]])]
            tree = Tree(root_idx=0)
            tree.Construct(edges)
            assert tree.size() == self.data_len[index]
            self.data_trees.append(tree)

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        trees = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        topic_labels = [item[3] for item in batch]
        return seqs, trees, torch.tensor(labels), torch.tensor(topic_labels)

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not getattr(self, "data_trees", False):
            self.init_trees()
        tree = self.data_trees[index]
        seq = [" ".join(self.data[d_ID]['text'][j]) for j in range(self.data_len[index])]
        assert len(seq) == (tree.size())
        return seq, tree, self.data_y[index], self.data[self.data_ID[index]]['topic_label']

class BiGCNWeiboSet(WeiboSet):
    def __init__(self, max_seq_len=20,  min_seq_len=5):
        super(BiGCNWeiboSet, self).__init__(max_seq_len=max_seq_len,  min_seq_len=min_seq_len)

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        TD_graphs = [item[1] for item in batch]
        BU_graphs = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        topic_labels = [item[4] for item in batch]
        return seqs, TD_graphs, BU_graphs, torch.tensor(labels).argmax(dim=1), torch.tensor(topic_labels)

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
            tIds_dic = {}
            dup_cnt = 0
            dup_idxs = []
            for idx, ID in enumerate(self.data[d_ID]["tweet_id"]):
                if ID in tIds_dic:
                    self.data_len[index] -= 1
                    dup_cnt += 1
                    dup_idxs.append(idx)
                else:
                    tIds_dic[ID] = idx - dup_cnt
            for idx in dup_idxs:
                    self.data[d_ID]["tweet_id"].pop(idx)
                    self.data[d_ID]["reply_to"].pop(idx)
                    self.data[d_ID]["text"].pop(idx)
                    self.data[d_ID]["sentence"].pop(idx)
                    self.data[d_ID]["created_at"].pop(idx)
                    self.data[d_ID]["reply_idx"].pop(idx)
            edges = [(src_idx, tIds_dic[dst_ID])
                    for src_idx, dst_ID in enumerate(self.data[d_ID]["reply_to"][:self.data_len[index]]) if dst_ID in tIds_dic]
            src = np.array([item[0] for item in edges])
            dst = np.array([item[1] for item in edges])
            g_TD = dgl.graph((src, dst), num_nodes=self.data_len[index])
            g_BU = dgl.graph((dst, src), num_nodes=self.data_len[index])
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU

        if index in self.lemma_text:
            seq = self.lemma_text[index]
        else:
            seq = [" ".join(self.data[self.data_ID[index]]['text'][j]) for j in range(self.data_len[index])]
            self.lemma_text[index] = seq

        assert len(seq) == g_TD.num_nodes() and len(seq) == g_TD.num_nodes()
        return (seq, dgl.add_self_loop(g_TD), \
               dgl.add_self_loop(g_BU), \
               self.data_y[index], \
               self.data[self.data_ID[index]]['topic_label'])
