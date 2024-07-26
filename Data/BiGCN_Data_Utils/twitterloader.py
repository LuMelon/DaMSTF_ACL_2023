import os, time, datetime, numpy as np
import random, math, re, pdb, pickle
import torch, json, sys
import pandas as pd
from tqdm import tqdm
import pkuseg
from typing import List
from .dataloader_utils import Tree
from torch.utils.data import Dataset
from .dataloader_utils import RumorLoader
import dgl, nltk
from nltk import WordNetLemmatizer
from prefetch_generator import background

event_dics = {
    'charliehebdo': 0,
    'ferguson': 1,
    'germanwings-crash': 2,
    'ottawashooting': 3,
    'sydneysiege': 4
}

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp

def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p] for (t, p) in zip(time[idxs], posts[idxs])]
    del time, posts
    return rst

@background(max_prefetch=5)
def twitter_data_process(files):
    for file_path in tqdm(files):
        ret = {}
        ss = file_path.split("/")
        data = json.load(open(file_path, mode="r", encoding="utf-8"))
        # 'Wed Jan 07 11:14:08 +0000 2015'
        if data['lang'] == 'en':
            ret[ss[-3]] = {
                            'topic_label': event_dics[ss[-5]],
                            'label': ss[-4],
                            'event': ss[-5],
                            'sentence': [data['text'].lower()],
                            'created_at': [str2timestamp(data['created_at'])],
                            'tweet_id': [data['id']],
                            "reply_to": [data['in_reply_to_status_id']]
                           }
            yield ret


class CustomDataset(Dataset):
    items, _label, _confidence, _entrophy, _domain = [], None, None, None, None
    read_indexs:np.array = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def collate_fn(self, items):
        raise NotImplementedError("'collate_fn' is not impleted")

    def __len__(self):
        return len(self.read_indexs)

    @property
    def label(self):
        return self._label[self.read_indexs]

    def setLabel(self, label, idxs):
        indexs = self.read_indexs[idxs]
        self._label[indexs] = label

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._label[0])
        return torch.tensor(self._label[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    @property
    def confidence(self):
        return self._confidence[self.read_indexs]

    def setConfidence(self, confidence, idxs):
        indexs = self.read_indexs[idxs]
        self._confidence[indexs] = confidence

    @property
    def entrophy(self):
        return self._entrophy[self.read_indexs]

    def setEntrophy(self, entrophy, idxs):
        indexs = self.read_indexs[idxs]
        self._entrophy[indexs] = entrophy

    @property
    def domain(self):
        return self._domain[self.read_indexs]

    def tensor_dataset(self, device=None):
        """
        :param device:
        :return: TensorDataset, collate_fn
        """
        raise NotImplementedError("'tensor_dataset' is not impleted")

    def domain_tensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._domain[0])
        return torch.tensor(self._domain[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    def initLabel(self, label):
        assert len(self.read_indexs) == len(label)
        self._label = label

    def initConfidence(self, confidence):
        assert len(self.read_indexs) == len(confidence)
        self._confidence = confidence

    def initEntrophy(self, entrophy):
        assert len(self.read_indexs) == len(entrophy)
        self._entrophy = entrophy

    def initDomain(self, d_arr):
        assert len(self.read_indexs) == len(d_arr)
        self._domain = d_arr

    def read_item(self, index):
        raise NotImplementedError("'read_item' has not been implemented!")

    def Derive(self, idxs: List):
        raise NotImplementedError("'Derive' has not been implemented!")

    def Split(self, percent=[0.5, 1.0]):
        raise NotImplementedError("'Split' has not been implemented!")

    def Merge(self, another_set):
        raise NotImplementedError("'Merge' has not been implemented!")

    def __getitem__(self, idx):
        if self.read_indexs is not None:
            index = self.read_indexs[idx]
            return self.read_item(index)
        else:
            return self.read_item(idx)

class SentiReader(Dataset):
    def __init__(self, data_csv_file):
        df = pd.read_csv(data_csv_file).dropna()
        self.sents = [self.lineClear(line) for line in df['content'].values.tolist()]
        self.label = [item for item in df['label'].values.tolist()]
        self.idxs = list(range(len(self.label)))

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

    def sample_batch(self, batch_size):
        batch_idxs = random.sample(self.idxs, batch_size)
        batch = [self.__getitem__(idx) for idx in batch_idxs]
        return self.collate_raw_batch(batch)

    def __getitem__(self, index):
        return self.sents[index], self.label[index]

    def __len__(self):
        return len(self.sents)

class TopicReader(Dataset):
    def __init__(self, data_csv_file):
        df = pd.read_csv(data_csv_file)
        self.sents = [self.lineClear(line) for line in df['content'].values.tolist()]
        self.label = [event_dics[item] for item in df['event'].values.tolist()]

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

class TwitterLoader(RumorLoader):
    def __init__(self):
        super(TwitterLoader, self).__init__()
        self.files = []
        userdic = [':)', '[', ']', '。', ',', '，', '{', '}', '(', ')', '!', '！',
                   '~', '～', '"', ':', '+', '-', '$', '.', '?', '<', '>', '|', '=', '...']
        self.seg = pkuseg.pkuseg(user_dict=userdic)

    def transIrregularWord(self, line, seg=None):
        if not line:
            return ''
        line.lower()
        line = re.sub("@[^ \n\t]*", " @ ", line)
        line = re.sub("#[^ \n\t]*", " # ", line)
        line = re.sub("http(.?)://[^ ]*", " url ", line)
        if seg is not None:
            return seg.cut(line)
        return line.split()

    def list_files(self, path):
        self.scan_dir(path)

    def scan_dir(self, dir_name):
        for item in os.walk(dir_name):
            if len(item[2]) == 0:
                # no file in this dir
                pass
            else:
                for fname in item[2]:
                    tmp_path = os.path.join(item[0], fname)
                    if tmp_path[-5:] == ".json": # is a json file
                        self.files.append(tmp_path)
                    else:
                        print("Warning: non json format file exists in %s : %s" % (dir_name, tmp_path))

    def fetch(self, twitter_dict):
        for key in twitter_dict.keys():  # use temporary data to organize the final whole data
            if key in self.data:
                if twitter_dict[key]['tweet_id'][0] in self.data[key]['tweet_id']:
                    pass  # sometimes, there are dumlicated posts
                else:
                    self.data[key]['tweet_id'].append(twitter_dict[key]['tweet_id'][0])
                    self.data[key]['sentence'].append(twitter_dict[key]['sentence'][0])
                    self.data[key]['created_at'].append(twitter_dict[key]['created_at'][0])
                    self.data[key]['reply_to'].append(twitter_dict[key]['reply_to'][0])
            else:
                self.data[key] = twitter_dict[key]

    def read_json_files(self, files):
        for twitter_dict in twitter_data_process(files):
            self.fetch(twitter_dict)

    def preprocess_files(self):
        print("preprocessing files: ")
        self.read_json_files(self.files)

    def sort_by_timeline(self, key, temp_idxs):
        self.data[key]['sentence'] = [self.data[key]['sentence'][idx] for idx in temp_idxs]
        self.data[key]['created_at'] = [self.data[key]['created_at'][idx] for idx in temp_idxs]
        self.data[key]['tweet_id'] = [self.data[key]['tweet_id'][idx] for idx in temp_idxs]
        self.data[key]['reply_to'] = [self.data[key]['reply_to'][idx] for idx in temp_idxs]

    def gather_posts(self, key, temp_idxs, post_fn):
        id2idx = {t_id: idx for idx, t_id in enumerate(self.data[key]['tweet_id'])}
        id2idx[None] = -1
        self.data[key]['text'] = []
        ttext = ""
        for i in range(len(temp_idxs)):
            if i % post_fn == 0:  # merge the fixed number of texts in a time interval
                if len(ttext) > 0:  # if there are data already in ttext, output it as a new instance
                    words = self.transIrregularWord(ttext, self.seg)
                    self.data[key]['text'].append(words)
                    ttext = ''
                else:
                    ttext = self.data[key]['sentence'][i]
            else:
                ttext += " " + self.data[key]['sentence'][i]
        # keep the last one
        if len(ttext) > 0:
            words = self.transIrregularWord(ttext)
            self.data[key]['text'].append(words)

    def dataclear(self, post_fn=1):
        print("data clear:")
        for key, value in tqdm(self.data.items()):
            temp_idxs = np.array(self.data[key]['created_at']).argsort().tolist()
            self.sort_by_timeline(key, temp_idxs)
            self.gather_posts(key, temp_idxs, post_fn)

        for key in self.data.keys():
            self.data_ID.append(key)
        self.data_ID = random.sample(self.data_ID, len(self.data_ID))  # shuffle the data id

        for i in range(len(self.data_ID)):  # pre processing the extra informations
            self.data_len.append(len(self.data[self.data_ID[i]]['text']))
            if self.data[self.data_ID[i]]['label'] == "rumours":
                self.data_y.append([0.0, 1.0])
            else:
                self.data_y.append([1.0, 0.0])

    def load_data(self, data_path = "../pheme-rnr-dataset/"):
        self.scan_dir(data_path)
        self.read_json_files(self.files)
        self.dataclear()

    def load_event_list(self, event_list):
        for event_path in event_list:
            self.scan_dir(event_path)

        self.read_json_files(self.files)
        self.dataclear()

class Covid19Loader(RumorLoader):
    def __init__(self):
        super(Covid19Loader, self).__init__()

    def load_data(self, data_dir):
        with open(f"{data_dir}/Twitter_label_all.txt") as fr:
            lines = [line.strip('\n') for line in fr]
        items = [line.split('\t') for line in lines]
        self.data_ID = [item[0] for item in items]
        self.data_y = [[0, 1] if item[1] == '1' else [1, 0] for item in items]

        self.data = {
            ID: {
                'text': [],
                'created_at': [],
                'edges': []
            } for ID in self.data_ID
        }

        error_IDs = []
        with open(f"{data_dir}/Twitter_data_all.txt") as fr:
            for line in tqdm(fr):
                s = line.strip('\n').split('\t')
                if not s[0] in self.data:
                    if not s[0] in error_IDs:
                        error_IDs.append(s[0])
                        print(f" WARNING : ID '{s[0]}' does not exists in the label file")
                        print(f"The count of the error ID is {len(error_IDs)}")
                    continue
                self.data[s[0]]['text'].append(s[4].split(' '))
                self.data[s[0]]['created_at'].append(float(s[3]))
                if s[1] != 'None':
                    self.data[s[0]]['edges'].append([int(s[1]) - 1, int(s[2]) - 1])

        self.data_len = []
        for ID in self.data_ID:
            self.data_len.append(len(self.data[ID]['text']))

    def construct_graph(self, index, d_ID):
        edges = self.data[d_ID]['edges']
        src = np.array([item[0] for item in edges])
        dst = np.array([item[1] for item in edges])
        g_TD = dgl.graph((dst, src), num_nodes=self.data_len[index])
        g_BU = dgl.graph((src, dst), num_nodes=self.data_len[index])
        return g_TD, g_BU

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
            seq = [" ".join(self.lemma(self.data[self.data_ID[index]]['text'][j])) for j in
                   range(self.data_len[index])]
            self.lemma_text[index] = seq

        assert len(seq) == g_TD.num_nodes() and len(seq) == g_TD.num_nodes()
        return (seq, dgl.add_self_loop(g_TD), \
                dgl.add_self_loop(g_BU), \
                self.data_y[index], \
                self.data[self.data_ID[index]]['topic_label'])

class TwitterSet(TwitterLoader, CustomDataset):
    def __init__(self, batch_size=20):
        super(TwitterSet, self).__init__()
        self.batch_size = batch_size
        self.sample_len = -1
        self.lemmatizer = WordNetLemmatizer()

    def lemma(self, word_tokens):
        try:
            tags = nltk.pos_tag(word_tokens)
        except:
            print("lemma tokens : ", word_tokens)
            raise
        new_words = []
        for pair in tags:
            if pair[1].startswith('J'):
                new_words.append(self.lemmatizer.lemmatize(pair[0], 'a'))
            elif pair[1].startswith('V'):
                new_words.append(self.lemmatizer.lemmatize(pair[0], 'v'))
            elif pair[1].startswith('N'):
                new_words.append(self.lemmatizer.lemmatize(pair[0], 'n'))
            elif pair[1].startswith('R'):
                new_words.append(self.lemmatizer.lemmatize(pair[0], 'r'))
            else:
                new_words.append(pair[0])
        return new_words

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        lens = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        topic_labels = [item[3] for item in batch]
        return seqs, torch.tensor(lens), torch.tensor(labels).argmax(dim=1), torch.tensor(topic_labels)

    def __len__(self):
        return len(self.data_ID)

    def read_item(self, index):
        if self.sample_len != -1:
            tmp_seq = [" ".join(self.lemma(self.data[self.data_ID[index]]['text'][j])) for j in range(self.data_len[index])]
            new_len = min(self.sample_len, len(tmp_seq))
            seq = tmp_seq[0:1] + [tmp_seq[idx] for idx in np.sort(random.sample(list(range(1, len(tmp_seq))), new_len-1))]
            return seq, len(seq), self.data_y[index], self.data[self.data_ID[index]]['topic_label']
        else:
            seq = [" ".join(self.lemma(self.data[self.data_ID[index]]['text'][j])) for j in range(self.data_len[index])]
            return seq, self.data_len[index], self.data_y[index], self.data[self.data_ID[index]]['topic_label']

class BiGCNTwitterSet(TwitterSet):
    def __init__(self, batch_size=20):
        super(BiGCNTwitterSet, self).__init__(batch_size=batch_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def collate_raw_batch(self, batch):
        seqs = [item[0] for item in batch]
        TD_graphs = [item[1] for item in batch]
        BU_graphs = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        topic_labels = [item[4] for item in batch]
        return seqs, TD_graphs, BU_graphs, torch.tensor(labels).argmax(dim=1), torch.tensor(topic_labels)

    def construct_graph(self, index, d_ID):
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
            self.data[d_ID]["tweet_id"].pop(idx-i)
            self.data[d_ID]["reply_to"].pop(idx-i)
            self.data[d_ID]["text"].pop(idx-i)
            self.data[d_ID]["sentence"].pop(idx-i)
            self.data[d_ID]["created_at"].pop(idx-i)

        edges = [(src_idx, tIds_dic[dst_ID] if dst_ID in tIds_dic else 0)
                 for src_idx, dst_ID in enumerate(self.data[d_ID]["reply_to"][:self.data_len[index]])]
        src = np.array([item[0] for item in edges])
        dst = np.array([item[1] for item in edges])
        g_TD = dgl.graph((dst, src), num_nodes=self.data_len[index])
        g_BU = dgl.graph((src, dst), num_nodes=self.data_len[index])
        return g_TD, g_BU

    def extract_sentence_pairs(self, index, d_ID):
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
        edges = [(tIds_dic[parent_ID], child)
                 for child, parent_ID in enumerate(self.data[d_ID]["reply_to"])
                    if parent_ID in tIds_dic]
        sentence_pairs = [
            (" ".join(self.data[d_ID]['text'][src]), " ".join(self.data[d_ID]['text'][dst]))
            for src, dst in edges
        ]
        return sentence_pairs

    def sentence_pairs_to_files(self, file_path):
        with open(file_path, 'a' if os.path.exists(file_path) else 'w') as fw:
            try:
                for idx, d_ID in enumerate(tqdm(self.data_ID)):
                    sentence_pairs = self.extract_sentence_pairs(self, idx, d_ID)
                    dicts = [json.dumps({'sent':item[0], 'reply':item[1]}) for item in sentence_pairs]
                    for line in dicts:
                        fw.write(line+"\n")
            except:
                fw.close()
                raise
            else:
                fw.close()

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