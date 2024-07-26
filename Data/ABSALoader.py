import sys, torch, random, sqlite3
sys.path.append("..")
from TrainingEnv import CustomDataset
from torch.utils.data import Dataset
from typing import AnyStr, List, Tuple
import numpy as np
import pickle, os
from transformers import PreTrainedTokenizer


Senti_domain_map = {
  'laptop':0,
  'restaurant':1,
  'twitter':2,
}

def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: List = None):
    """Turn a piece of text into a batch for transformer model
    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    max_len = tokenizer.max_len if hasattr(tokenizer, 'max_len') else tokenizer.model_max_length
    if text_pair is None:
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_len) for t in text]
        masks = [[1] * len(i) for i in input_ids]
        return input_ids, masks
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

class ABSA_Dataset(CustomDataset):
    """
    Implements a dataset for the multidomain sentiment analysis dataset
    """
    def __init__(
            self,
            database_name: AnyStr = None,
            seen_domains: List = None,
            table_name_list: List = None,
            tokenizer: PreTrainedTokenizer = None,
            max_data_size: int = -1,
            C_dimension=0,
            load_data=False
    ):
        """
        :param database_name: The name of the database that stores the dataset
        :param domains: The set of domains to load data for
        :param: tokenizer: The tokenizer to use
        :param: domain_ids: A list of ids to override the default domain IDs
        """
        super(ABSA_Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.database_name = database_name
        print("database ", self.database_name )
        self.max_data_size = max_data_size
        self.seen_domains = seen_domains
        self.aspects, self.sents, self.read_indexs = None, None, None
        self._label, self._confidence, self._entrophy, self._domain = None, None, None, None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.table_name_list = table_name_list
        self.C_dimension = C_dimension
        self.collate_raw_batch = collate_Senti_batch_with_device(device)
        if load_data:
            self.load_data()

    def Cache_data(self, filename):
        with open(filename, 'wb') as fw:
            pickle.dump([self.aspects, self.sents, self.read_indexs, self._label,
                         self._confidence, self._entrophy, self._domain], fw, protocol=pickle.HIGHEST_PROTOCOL)

    def Load_Cache(self, cache_name):
        with open(cache_name, 'rb') as fr:
            self.aspects, self.sents, self.read_indexs, self._label, \
                self._confidence, self._entrophy, self._domain = pickle.load(fr)

    def collate_fn(self, items):
        return self.collate_raw_batch(items)

    def load_data(self):
        item_list = self.obtainDataset(self.database_name, self.seen_domains,
                                       data_size=self.max_data_size)
        self.aspects = [item[0].replace('[pad]', "'") for item in item_list]
        self.sents = [item[1].replace('[pad]', "'") for item in item_list]
        if self.C_dimension > 0:
            self._label = np.zeros([len(item_list), self.C_dimension])
            for idx, item in enumerate(item_list):
                if item[2] >= 0:
                    self._label[idx][item[2]] = 1.0
        else:
            self._label = np.array([item[2] for item in item_list], dtype=np.int64)
        self._domain = np.array([Senti_domain_map[item[3]] for item in item_list], dtype=np.int64)
        self._confidence = torch.ones(len(self._label))
        self._entrophy = torch.zeros(len(self._label))
        self.read_indexs = np.arange(len(self._label))

    def obtainDataset(self, database=None, seen_domain_list=None, data_size=-1):
        assert database is not  None or self.database_name is not None
        assert seen_domain_list is not None or self.seen_domains is not None
        if database is None:
            database = self.database_name
        if seen_domain_list is None:
            domain_list = self.seen_domains
        else:
            domain_list = seen_domain_list
        con = sqlite3.connect(database)
        cur = con.cursor()
        item_list = []
        for d_name in domain_list:
            for table_name in self.table_name_list:
                cur.execute(
                    f"SELECT aspect, sentence, label, domain FROM {table_name} WHERE domain =='{d_name}' AND label!=3"
                )
                item_list.extend(cur.fetchall())
        if data_size > 0 and data_size < len(item_list):
            return random.sample(item_list, data_size)
        return random.sample(item_list, len(item_list)) # shuffle the instances

    def Derive(self, idxs:List):
        new_set = self.__class__(None, None, None, self.tokenizer, -1)
        real_idxs = [self.read_indexs[idx] for idx in idxs]
        new_set.read_indexs = np.arange(len(real_idxs))
        new_set.initLabel(self._label[real_idxs].copy())
        new_set.initConfidence(self.confidence[real_idxs].clone())
        new_set.initEntrophy(self.entrophy[real_idxs].clone())

        new_set.aspects = [self.aspects[kk] for kk in real_idxs]
        new_set.sents = [self.sents[kk] for kk in real_idxs]
        new_set.initDomain(self._domain[real_idxs])

        self.aspects, self.sents = [self.aspects[kk] for kk in range(len(self._label)) if kk not in real_idxs], \
                                        [self.sents[kk] for kk in range(len(self._label)) if kk not in real_idxs]
        self._label, self._domain, self._confidence, self._entrophy = np.delete(self._label, real_idxs, axis=0), \
                                                                          np.delete(self._domain, real_idxs, axis=0), \
                                                                            np.delete(self._confidence, real_idxs, axis=0), \
                                                                                np.delete(self._entrophy, real_idxs, axis=0)
        self.read_indexs = np.arange(len(self._label))
        return new_set

    def Clone(self):
        new_set = self.__class__(None, None, None, self.tokenizer, -1)
        new_set.read_indexs = np.arange(len(self.read_indexs))
        new_set.initLabel(self._label[self.read_indexs].copy())
        new_set.initConfidence(self.confidence[self.read_indexs].clone())
        new_set.initEntrophy(self.entrophy[self.read_indexs].clone())

        new_set.aspects = [self.aspects[kk] for kk in self.read_indexs]
        new_set.sents = [self.sents[kk] for kk in self.read_indexs]
        new_set.initDomain(self._domain[self.read_indexs])
        return new_set

    def Merge(self, another_set:CustomDataset):
        another_aspects = [another_set.aspects[kk] for kk in another_set.read_indexs]
        another_sents = [another_set.sents[kk] for kk in another_set.read_indexs]
        another_domain = another_set.domain[another_set.read_indexs]
        extend_idxs = np.arange(len(self._label), len(self._label) + len(another_set), 1)
        self.read_indexs = np.concatenate(
            [self.read_indexs, extend_idxs]
        )
        self.aspects.extend(another_aspects)
        self.sents.extend(another_sents)
        self._domain, self._label, self._confidence, self._entrophy = np.concatenate([self._domain, another_domain]), \
                                                                      np.concatenate([self._label, another_set.label]), \
                                                                      torch.cat([self._confidence, another_set.confidence]), \
                                                                      torch.cat([self._entrophy, another_set.entrophy])

    def __getitem__(self, item) -> Tuple:
        idx = self.read_indexs[item]
        input_ids, mask, seg_ids = text_to_batch_transformer([self.aspects[idx]],
                                                              self.tokenizer,
                                                              text_pair=[self.sents[idx]])
        return input_ids, mask, seg_ids, self._label[idx], \
                        self.domain[idx], item

def obtain_Mul2One_set(new_domain_name, tokenizer_M, database_name=None, few_shot_cnt=100):
    if database_name is None:
        dir_name = os.path.dirname(__file__)
        database_name = os.path.join(dir_name, 'ABSA_Utils/DA_ASBA.db')
    domain_set = set(Senti_domain_map)
    domain_set.remove(new_domain_name)
    source_domain = ABSA_Dataset(
        database_name = database_name,
        seen_domains = list(domain_set),
        table_name_list=['Train'],
        tokenizer = tokenizer_M,
        max_data_size = -1,
        C_dimension=3,
        load_data = True
    )
    test_target = ABSA_Dataset(
        database_name=database_name,
        seen_domains=[new_domain_name],
        table_name_list=['Test'],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        C_dimension=3,
        load_data=True
    )
    unlabeled_target = ABSA_Dataset(
        database_name=database_name,
        seen_domains=[new_domain_name],
        table_name_list=['Train'],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        C_dimension=3,
        load_data=True
    )
    val_idxs = random.sample(range(len(unlabeled_target)),
                             int(0.2*len(unlabeled_target)))
    val_set = unlabeled_target.Derive(idxs=val_idxs)
    if few_shot_cnt > 0:
        labeled_target = ABSA_Dataset(
            database_name=database_name,
            seen_domains=[new_domain_name],
            table_name_list=['FewShot'],
            tokenizer=tokenizer_M,
            max_data_size=few_shot_cnt,
            C_dimension=3,
            load_data=True
        )
        return source_domain, val_set, test_target, labeled_target, unlabeled_target
    else:
        return source_domain, val_set, test_target, None, unlabeled_target

def obtain_data_set(source_domain_names:List, target_domain_names:List, tokenizer_M,
                        database_name=None, few_shot_cnt=100):
    if database_name is None:
        dir_name = os.path.dirname(__file__)
        database_name = os.path.join(dir_name, 'ABSA_Utils/DA_ASBA.db')
    source_domain = ABSA_Dataset(
        database_name = database_name,
        seen_domains = source_domain_names,
        table_name_list=['Train'],
        tokenizer = tokenizer_M,
        max_data_size = -1,
        C_dimension=3,
        load_data = True
    )
    test_target = ABSA_Dataset(
        database_name=database_name,
        seen_domains=target_domain_names,
        table_name_list=['Test'],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        C_dimension=3,
        load_data=True
    )
    unlabeled_target = ABSA_Dataset(
        database_name=database_name,
        seen_domains=target_domain_names,
        table_name_list=['Train'],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        C_dimension=3,
        load_data=True
    )
    val_idxs = random.sample(range(len(unlabeled_target)),
                             int(0.2*len(unlabeled_target)))
    val_set = unlabeled_target.Derive(idxs=val_idxs)
    if few_shot_cnt > 0:
        labeled_target = ABSA_Dataset(
            database_name=database_name,
            seen_domains=target_domain_names,
            table_name_list=['FewShot'],
            tokenizer=tokenizer_M,
            max_data_size=few_shot_cnt,
            C_dimension=3,
            load_data=True
        )
        return source_domain, val_set, test_target, labeled_target, unlabeled_target
    else:
        return source_domain, val_set, test_target, None, unlabeled_target