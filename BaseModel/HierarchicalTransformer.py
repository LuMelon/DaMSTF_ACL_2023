import sys, os, dgl, random
sys.path.append("..")
from prefetch_generator import background
from BaseModel.BiGCN_Utils.RumorDetectionBasic import BaseEvaluator
from BaseModel.BiGCN_Utils.GraphRumorDect import BiGCNRumorDetecV2
from Data.BiGCN_Dataloader import BiGCNTwitterSet, load_data_and_TfidfVectorizer
from BaseModel.modeling_bert import *
from transformers.models.bert import BertConfig, BertTokenizer
import torch, torch.nn as nn
from typing import List
from torch.utils.data import Dataset
from BaseModel.BiGCN_Utils.RumorDetectionBasic import RumorBaseTrainer

# assert  torch.cuda.is_available() and torch.cuda.device_count() == 4

class SentBert(nn.Module):
    def __init__(self, bertPath):
        super(SentBert, self).__init__()
        self.bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        self.model = nn.DataParallel(
            BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda:0')),
            device_ids=[0, 1, 2, 3]
        )

    def text_to_batch_transformer(self, text: List):
        """Turn a piece of text into a batch for transformer model

        :param text: The text to tokenize and encode
        :param tokenizer: The tokenizer to use
        :param: text_pair: An optional second string (for multiple sentence sequences)
        :return: A list of IDs and a mask
        """
        max_len = self.tokenizer.max_len if hasattr(self.tokenizer, 'max_len') else self.tokenizer.model_max_length
        items = [self.tokenizer.encode_plus(sent, text_pair=None, add_special_tokens=True, max_length=max_len,
                                       return_length=False, return_attention_mask=True,
                                       return_token_type_ids=True)
                 for sent in text]
        input_ids =  [item['input_ids'] for item in items]
        masks = [item['attention_mask'] for item in items]
        seg_ids = [item['token_type_ids'] for item in items]
        max_length = max([len(i) for i in input_ids])

        input_ids = torch.tensor([(i + [0] * (max_length - len(i))) for i in input_ids], device=torch.device('cuda:0'))
        masks = torch.tensor([(m + [0] * (max_length - len(m))) for m in masks], device=torch.device('cuda:0'))
        seg_ids = torch.tensor([(s + [0] * (max_length - len(s))) for s in seg_ids], device=torch.device('cuda:0'))
        return input_ids, masks, seg_ids

    def forward(self, sents):
        input_ids, masks, seg_ids = self.text_to_batch_transformer(sents)
        encoder_dict = self.model.forward(
            input_ids=input_ids,
            attention_mask=masks,
            token_type_ids=seg_ids
        )
        return encoder_dict.pooler_output#.unsqueeze(0)

    def save_model(self, model_file):
        torch.save(self.model.module.state_dict(), model_file)

    def load_model(self, pretrained_file):
        self.model.module.load_state_dict(torch.load(pretrained_file))


class GraphEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])

    def set_aug_type(self, aug_type):
        for i in range(self.config.num_hidden_layers):
            self.layer[i].aug_type = aug_type

    def forward(
        self,
        hidden_states,
        root_idxs,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        assert hidden_states.dim() == 3
        lens = root_idxs + [hidden_states.data.size(1)]
        eye_mtx = torch.eye(len(root_idxs), device=hidden_states.data.device)
        trans_mtx = torch.stack([eye_mtx[jj-1] for jj in range(1, len(root_idxs)+1, 1) \
                                    for _ in range(lens[jj-1], lens[jj])])

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            new_hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            hidden_states = torch.matmul(trans_mtx, hidden_states.squeeze(0)[root_idxs]).unsqueeze(0) \
                                + new_hidden_states


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class TwitterTransformer(BiGCNRumorDetecV2):
    def __init__(self, sent2vec, prop_model, rdm_cls, **kwargs):
        super(TwitterTransformer, self).__init__(sent2vec, prop_model, rdm_cls)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

    def Batch2Vecs(self, batch):
        sentences = [text for s_list in batch[0] for text in s_list]
        assert len(sentences) == sum(batch[1])
        A_TD: torch.Tensor = batch[2].bool().float()
        A_BU: torch.Tensor = batch[3].bool().float()
        adj_mtx = (A_TD + A_BU).__ne__(0.0).float()
        attn_mask = (-10000 * (1.0 - adj_mtx)).unsqueeze(0).unsqueeze(0)
        num_nodes: List = batch[1]
        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        inputs = self.sent2vec(sentences)
        rst = self.prop_model(inputs.unsqueeze(0), root_idxs, attention_mask=attn_mask)
        hiddens = rst[0].squeeze(0)
        return hiddens[root_idxs]

def obtain_Transformer(bertPath, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    lvec = SentBert(bertPath)
    prop = GraphEncoder(lvec.bert_config, 2).to(device)
    cls = nn.Linear(768, 2).to(device)
    BiGCN_model = TwitterTransformer(lvec, prop, cls)
    return BiGCN_model

class TransformerEvaluator(BaseEvaluator):
    def __init__(self, dataset:BiGCNTwitterSet, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labelTensor = dataset.labelTensor()

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        sent_list = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        return sent_list, num_nodes, A_TD, A_BU, \
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


class BiGCNTrainer(RumorBaseTrainer):
    def __init__(self, log_dir, tokenizer, **kwargs):
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
        sents = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        return sents, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)

    def trainset2trainloader(self, dataset: Dataset, shuffle=False, batch_size=32):
        return self.dataset2dataloader(dataset, shuffle, batch_size)

    def dataset2dataloader(self, dataset: BiGCNTwitterSet, shuffle=False, batch_size=32):
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
    data_dir = "/mnt/VMSTORE/pheme-rnr-dataset"
    # data_dir = r"../MetaGenerator/pheme-rnr-dataset/"
    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
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
    print("\n\n===========%s Train===========\n\n" % te.data[te.data_ID[0]]['event'])

    # bertPath = r"/data/run01/scz3924/hezj/MetaGenerator/models/bert-base-uncased"
    bertPath = "/mnt/VMSTORE/bert_en"
    model = obtain_Transformer(bertPath)
    dev_eval = TransformerEvaluator(dev, batch_size=20)
    te_eval = TransformerEvaluator(te, batch_size=20)
    trainer = BiGCNTrainer(log_dir, tfidf_vectorizer, model_rename=True)
    trainer.fit(model, tr, dev_eval, te_eval, batch_size=32, grad_accum_cnt=1, learning_rate=2e-5, max_epochs=20, 
                model_file=os.path.join(log_dir, f'BiGCN_{test_event_name}.pkl'))