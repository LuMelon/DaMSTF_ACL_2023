from typing import AnyStr, List, Tuple
from transformers import PreTrainedTokenizer
import pandas as pd, numpy as np
import json, random, re, torch
from torch.utils.data import Dataset
from TrainingEnv import CustomDataset
import sqlite3, pickle, os

Senti_domain_map = {
  'books':0,
  'dvd':1,
  'kitchen':2,
  'electronics':3
}

def DataSplit(dataset, length=[]):
    idxs = random.sample(list(range(len(dataset))), len(dataset))
    news_sets = [dataset.__class__() for _ in length]
    start_idx = 0
    for i, l in enumerate(length):
        news_sets[i].dataset = dataset.dataset.iloc[idxs[start_idx:start_idx+l]]
        news_sets[i].tokenizer = dataset.tokenizer
        start_idx += l
    return news_sets

def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: List = None):
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    max_len = tokenizer.max_len if hasattr(tokenizer, 'max_len') else tokenizer.model_max_length
    if text_pair is None:
        items = [tokenizer.encode_plus(sent, text_pair=None, add_special_tokens=True, max_length=max_len,
                                       return_length=False, return_attention_mask=True,
                                       return_token_type_ids=True)
                 for sent in text]
    else:
        assert len(text) == len(text_pair)
        items = [tokenizer.encode_plus(s1, text_pair=s2, add_special_tokens=True, max_length=max_len,
                                        return_length=False, return_attention_mask=True,
                                            return_token_type_ids=True)
                                        for s1, s2 in zip(text, text_pair)]
    return [item['input_ids'] for item in items], \
              [item['attention_mask'] for item in items], \
                 [item['token_type_ids'] for item in items]

def collate_Senti_batch_with_device(device):
    def collate_batch_transformer(input_data: Tuple):
        input_ids = [i[0][0] for i in input_data]
        masks = [i[1][0] for i in input_data]
        seg_ids = [i[2][0] for i in input_data]
        labels = [i[3] for i in input_data]
        domains = [i[4] for i in input_data]

        max_length = max([len(i) for i in input_ids])

        input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
        masks = [(m + [0] * (max_length - len(m))) for m in masks]
        seg_ids = [(s + [0] * (max_length - len(s))) for s in seg_ids]

        assert (all(len(i) == max_length for i in input_ids))
        assert (all(len(m) == max_length for m in masks))
        assert (all(len(s) == max_length for s in seg_ids))
        return torch.tensor(input_ids, device=device), torch.tensor(masks, device=device), \
                    torch.tensor(seg_ids, device=device), torch.tensor(labels, device=device), \
                        torch.tensor(domains, device=device)
    return collate_batch_transformer

def collate_batch_transformer(input_data: Tuple):
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]
    domains = [i[3] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), \
           torch.tensor(labels), torch.tensor(domains)

def collate_batch_transformer_with_index(input_data: Tuple):
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)

def read_senti(txt_path: AnyStr):
    """ Convert all of the ratings in amazon product XML file to dicts
    :param xml_file: The XML file to convert to a dict
    :return: All of the rows in the xml file as dicts
    """
    reviews = []
    domain_name = txt_path.rsplit('/', 2)[1]
    with open(txt_path, encoding='utf8', errors='ignore') as f:
        for line in f:
            s = line.strip("\n").split(' ', 1)
            reviews.append({'sent' : s[1],
                            'label' : int(s[0]),
                            'domain' : Senti_domain_map[domain_name]})
    return reviews

def transIrregularWord(line):
    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ \n\t]*", "", line)
    line = re.sub("#[^ \n\t]*", "", line)
    line = re.sub("http(.?)://[^ ]*", "", line)
    return line

class SentiDataset(CustomDataset):
    """
    Implements a dataset for the multidomain sentiment analysis dataset
    """
    def __init__(
            self,
            database_name: AnyStr = None,
            seen_domains: List = None,
            tokenizer: PreTrainedTokenizer = None,
            max_data_size: int = -1,
            data_type:List = None,
            C_dimension = 0,
            load_data=False
    ):
        """
        :param database_name:
        :param seen_domains:
        :param tokenizer:
        :param max_data_size::
        :param data_type:
        :param C_dimension: the dimension of the label. If 'C_dimension==0', _label is a Long type Tensor;
                            otherwise, _label is a one-hot vector or
        :param load_data:
        """
        super(SentiDataset, self).__init__()
        self.database_name = database_name
        self.sents, self.read_indexs = None, None
        self._label, self._confidence, self._entrophy, self._domain = None, None, None, None
        self.tokenizer = tokenizer
        self.max_data_size = max_data_size
        self.data_type = data_type
        self.C_dimension = C_dimension
        if seen_domains is not None:
            self.seen_domains = seen_domains
            self.seen_domain_ids = [Senti_domain_map[d_name] for d_name in self.seen_domains]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.collate_raw_batch = collate_Senti_batch_with_device(device)
        if load_data:
            self.load_data()

    def collate_fn(self, items):
        return self.collate_raw_batch(items)

    def Cache_data(self, filename):
        with open(filename, 'wb') as fw:
            pickle.dump([self.sents, self.read_indexs, self._label,
                         self._confidence, self._entrophy, self._domain], fw, protocol=pickle.HIGHEST_PROTOCOL)

    def Load_Cache(self, cache_name):
        with open(cache_name, 'rb') as fr:
            self.sents, self.read_indexs, self._label, \
                self._confidence, self._entrophy, self._domain = pickle.load(fr)

    def load_data(self):
        item_list = self.obtainDataset()
        self.sents = [item[0] for item in item_list]
        if self.C_dimension > 0:
            self._label = np.zeros([len(item_list), self.C_dimension])
            for idx, item in enumerate(item_list):
                if item[1] >= 0:
                    self._label[idx][item[1]] = 1.0
        else:
            self._label = np.array([item[1] for item in item_list], dtype=np.long)
        self._domain = np.array([Senti_domain_map[item[2]] for item in item_list])
        self._confidence = torch.ones(len(self._label))
        self._entrophy = torch.zeros(len(self._label))
        self.read_indexs = np.arange(len(self._label))

    def obtainDataset(self):
        """
        For different type of dataset (e.g., unlabeled data, labeled data), we have to implement different function
        to obtain the dataset.
        Here, the database is organized as follows:
                       ID INT PRIMARY KEY     NOT NULL,
                       sentence       TEXT    NOT NULL,
                       label          INT     NOT NULL, // -1 indicates unlabeled data
                       domain         TEXT    NOT NULL,
                       data_type      TEXT    NOT NULL // train, valid, test, unlabeled
        """
        # raise NotImplementedError("'obtainDataset' is not impleted")
        assert (self.database_name is not None) and \
                    (self.seen_domains is not None) and \
                        (self.data_type is not None)
        con = sqlite3.connect(self.database_name)
        cur = con.cursor()
        item_list = []
        for d_name in self.seen_domains:
            for d_type in self.data_type:
                cur.execute(
                    f"SELECT sentence, label, domain FROM Amazon WHERE domain =='{d_name}' AND data_type='{d_type}'"
                )
                item_list.extend(cur.fetchall())
        if self.max_data_size > 0:
            return random.sample(item_list, min(len(item_list), self.max_data_size))
        return random.sample(item_list, len(item_list)) # shuffle the instances

    def domainSelect(self, domain_id):
        if not domain_id in self.seen_domain_ids:
            print(f"Select domain failed, due to the domain_id {domain_id} is missed in data domains")
        read_indexs = np.arange(len(self._domain))
        self.valid_domain_id = domain_id
        self.read_indexs = read_indexs[self._domain.__eq__(domain_id)]

    def Derive(self, idxs:List):
        new_set = self.__class__(None, None, self.tokenizer, -1)
        real_idxs = [self.read_indexs[idx] for idx in idxs]
        new_set.read_indexs = np.arange(len(real_idxs))
        new_set.initLabel(self._label[real_idxs].copy())
        new_set.initConfidence(self.confidence[real_idxs].clone())
        new_set.initEntrophy(self.entrophy[real_idxs].clone())
        new_set.sents = [self.sents[kk] for kk in real_idxs]
        new_set.initDomain(self._domain[real_idxs])

        self.sents = [self.sents[kk] for kk in range(len(self._label)) if kk not in real_idxs]
        self._label, self._domain, self._confidence, self._entrophy = np.delete(self._label, real_idxs, axis=0), \
                                                                          np.delete(self._domain, real_idxs, axis=0), \
                                                                            np.delete(self._confidence, real_idxs, axis=0), \
                                                                                np.delete(self._entrophy, real_idxs, axis=0)
        self.read_indexs = np.arange(len(self._label))
        return new_set

    def Split(self, percent=[0.5, 1.0]):
        rst_list = [self.__class__(None, None, self.tokenizer, -1) for _ in percent]
        shuffle_idxs = random.sample(range(len(self)), len(self))
        start = 0
        for r_idx, p in enumerate(percent):
            end = int(len(shuffle_idxs)*p)
            real_idxs = [self.read_indexs[idx] for idx in shuffle_idxs[start:end]]
            rst_list[r_idx].read_indexs = np.arange(len(real_idxs))
            rst_list[r_idx].initLabel(self._label[real_idxs].copy())
            rst_list[r_idx].initConfidence(self.confidence[real_idxs].clone())
            rst_list[r_idx].initEntrophy(self.entrophy[real_idxs].clone())
            rst_list[r_idx].sents = [self.sents[kk] for kk in real_idxs]
            rst_list[r_idx].initDomain(self._domain[real_idxs])
            start = end
        return tuple(rst_list)

    def Clone(self):
        new_set = self.__class__(None, None, self.tokenizer, -1)
        new_set.read_indexs = np.arange(len(self.read_indexs))
        new_set.initLabel(self._label[self.read_indexs].copy())
        new_set.initConfidence(self.confidence[self.read_indexs].clone())
        new_set.initEntrophy(self.entrophy[self.read_indexs].clone())
        new_set.sents = [self.sents[kk] for kk in self.read_indexs]
        new_set.initDomain(self._domain[self.read_indexs])
        return new_set

    def Merge(self, another_set:CustomDataset):
        extend_idxs = np.arange(len(self._label), len(self._label) + len(another_set), 1)
        another_sents = [another_set.sents[kk] for kk in another_set.read_indexs]
        another_domain = another_set.domain[another_set.read_indexs]
        self.sents.extend(another_sents)
        self._domain, self._label, self._confidence, self._entrophy = np.concatenate([self._domain, another_domain]), \
                                                                      np.concatenate([self._label, another_set.label]), \
                                                                      np.concatenate([self._confidence, another_set.confidence]), \
                                                                      np.concatenate([self._entrophy, another_set.entrophy])
        self.read_indexs = np.concatenate(
            [self.read_indexs, extend_idxs]
        )

    def read_item(self, idx):
        input_ids, mask, seg_ids = text_to_batch_transformer([self.sents[idx]],
                                                              self.tokenizer,
                                                              text_pair=None)
        return input_ids, mask, seg_ids, self._label[idx], \
                        self._domain[idx]


def obtain_Mul2One_set(new_domain_name, tokenizer_M, database_name=None, few_shot_cnt=100):
    if database_name is None:
        dir_name = os.path.abspath(__file__)
        database_name = os.path.join(
            dir_name,
            "Amazon_Utils/Sentiment.db"
        )
    domain_set = set(['books', 'dvd', 'kitchen', 'electronics'])
    domain_set.remove(new_domain_name)
    source_domain = SentiDataset(
        database_name = database_name,
        seen_domains = list(domain_set),
        tokenizer = tokenizer_M,
        max_data_size = -1,
        data_type = ['train', 'valid', 'test'],
        C_dimension=2,
        load_data = True
    )
    test_target = SentiDataset(
        database_name=database_name,
        seen_domains=[new_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['train'], #  70% of the labeled data in the target domain are taged with 'train'
        C_dimension=2,
        load_data=True
    )
    unlabeled_target = SentiDataset(
        database_name=database_name,
        seen_domains=[new_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['unlabeled'],
        C_dimension=2,
        load_data=True
    )
    val_set = SentiDataset(
        database_name=database_name,
        seen_domains=[new_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['test'], #  20% of the labeled data in the target domain are taged with 'test'
        C_dimension=2,
        load_data=True
    )
    if few_shot_cnt > 0:
        labeled_target = SentiDataset(
            database_name=database_name,
            seen_domains=[new_domain_name],
            tokenizer=tokenizer_M,
            max_data_size=few_shot_cnt,
            data_type=['valid'], #  20% of the labeled data in the target domain are taged with 'valid'
            C_dimension=2,
            load_data=True
        )
        return source_domain, val_set, test_target, labeled_target, unlabeled_target
    else:
        return source_domain, val_set, test_target, None, unlabeled_target


def obtain_data_set(source_domain_names:List, target_domain_names:List, tokenizer_M,
                        database_name=None, few_shot_cnt=100):
    if database_name is None:
        dir_name = os.path.abspath(__file__)
        database_name = os.path.join(
            dir_name,
            "Amazon_Utils/Sentiment.db"
        )
    source_domain = SentiDataset(
        database_name = database_name,
        seen_domains = source_domain_names,
        tokenizer = tokenizer_M,
        max_data_size = -1,
        data_type = ['train', 'valid', 'test'],
        C_dimension=2,
        load_data = True
    )
    test_target = SentiDataset(
        database_name=database_name,
        seen_domains=target_domain_names,
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['train'], #  70% of the labeled data in the target domain are taged with 'train'
        C_dimension=2,
        load_data=True
    )
    unlabeled_target = SentiDataset(
        database_name=database_name,
        seen_domains=target_domain_names,
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['unlabeled'],
        C_dimension=2,
        load_data=True
    )
    val_set = SentiDataset(
        database_name=database_name,
        seen_domains=target_domain_names,
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['test'], #  20% of the labeled data in the target domain are taged with 'test'
        C_dimension=2,
        load_data=True
    )
    if few_shot_cnt > 0:
        labeled_target = SentiDataset(
            database_name=database_name,
            seen_domains=target_domain_names,
            tokenizer=tokenizer_M,
            max_data_size=few_shot_cnt,
            data_type=['valid'], #  20% of the labeled data in the target domain are taged with 'valid'
            C_dimension=2,
            load_data=True
        )
        return source_domain, val_set, test_target, labeled_target, unlabeled_target
    else:
        return source_domain, val_set, test_target, None, unlabeled_target
