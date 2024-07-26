import sys, os, dgl, random, torch
sys.path.append("..")
sys.path.append("../..")
from BaseModel.BiGCN import obtain_BiGCN
from BaseModel.BiGCN import BiGCNEvaluator
from Data.BiGCN_Dataloader import load_data_and_TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from BaseModel.BiGCN_Utils.RumorDetectionBasic import RumorBaseTrainer
from prefetch_generator import background
from Data.BiGCN_Dataloader import FastBiGCNDataset
from torch.utils.data import Dataset


class BiGCNTrainer(RumorBaseTrainer):
    def __init__(self, log_dir, tokenizer: TfidfVectorizer, **kwargs):
        super(BiGCNTrainer, self).__init__()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

        self.running_dir = log_dir
        if not os.path.exists(self.running_dir):
            os.system(f"mkdir {self.running_dir}")
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    def trainset2trainloader(self, dataset: Dataset, shuffle=False, batch_size=32):
        return self.dataset2dataloader(dataset, shuffle, batch_size)

    def dataset2dataloader(self, dataset: FastBiGCNDataset, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(dataset), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                items = [dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)

        return dataloader()


if __name__ == '__main__':
    # log_dir = str(__file__).rstrip(".py")
    data_dir = "../../MLLu/pheme-rnr-dataset/"
    events_list = ['charliehebdo',  'ferguson',  'germanwings-crash',  'ottawashooting',  'sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 0
    source_events = [os.path.join(data_dir, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    target_events = [os.path.join(data_dir, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]
    tfidf_vectorizer, tr, _, dev, te = load_data_and_TfidfVectorizer(
        source_events, target_events, 0, unlabeled_ratio=-1
    )

    log_dir = f"./{test_event_name}/"
    print("%s : (dev event)/(test event)/(train event) = %3d/%3d/%3d" % (
    test_event_name, len(dev), len(te), len(tr)))
    print("\n\n===========%s Train===========\n\n"%te.data[te.data_ID[0]]['event'])

    model = obtain_BiGCN(tfidf_vectorizer)
    dev_eval = BiGCNEvaluator(dev, batch_size=20)
    te_eval = BiGCNEvaluator(te, batch_size=20)
    trainer = BiGCNTrainer(log_dir, tfidf_vectorizer, model_rename=True)
    trainer.fit(model, tr, dev_eval, te_eval, batch_size=32, grad_accum_cnt=1, learning_rate=5e-4, max_epochs=20,
                    model_file=os.path.join(log_dir, f'BiGCN_{test_event_name}.pkl'))