import torch.nn.functional as F
import torch.nn as nn
from TrainingEnv import VirtualModel, AdversarialModel, CustomDataset
from BaseModel.modeling_bert import BertForMaskedLM, BertForSequenceClassification
from transformers.models.bert import BertConfig, BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class PromptBERT(BertForMaskedLM, VirtualModel):
    def __init__(self, bert_config:BertConfig, **kwargs):
        super(PromptBERT, self).__init__(bert_config)
        self.prompt_num = kwargs.get('prompt_num', 5)
        self.prompt_embedding = nn.Embedding(self.prompt_num, bert_config.hidden_size).to(self.device)
        self.mask_sep_tensor = torch.tensor(
            [kwargs.get('mask_id'), kwargs.get('sep_id')],
            device=self.device
        )
        self.label_token_id = torch.tensor(
            [kwargs.get('mask_id'), kwargs.get('sep_id')],
            device=self.device
        )

    def grouped_parameters(self, learning_rate):
        def lr_coefficient(par_name):
        # layer-wise fine-tuning
            if "layer." in par_name:
                layer_num = int(par_name.split("layer.")[1].split(".", 1)[0])
                return pow(0.8, 12 - layer_num)
            elif "embedding" in par_name:
                return pow(0.8, 13)
            elif 'prompt_embedding' in par_name:
                return 1e2
            else:
                return 1.0
        if learning_rate is None:
            learning_rate = self.learning_rate
        model_paras = [{'params': p, 'lr': learning_rate * lr_coefficient(n)}
                                        for n, p in self.named_parameters()]
        return model_paras

    def Batch2Vecs(self, batch):
        if batch[0].device != self.device:  # data is on a different device
            input_ids, masks, seg_ids = batch[0].to(self.device), \
                                        batch[1].to(self.device), \
                                        batch[2].to(self.device)
        else:
            input_ids, masks, seg_ids = batch[0], batch[1], batch[2]

        batch_size, _ = input_ids.shape

        word_embs = self.bert.embeddings.word_embeddings(input_ids[:, :-1]) # [batch_size, seq_len-1, dim]
        prompt_emb = self.prompt_embedding.weight.repeat((batch_size, 1, 1)) # [batch_size, prompt_len, dim]
        mlm_embeding = self.bert.embeddings.word_embeddings(
            self.mask_sep_tensor
        ).repeat((batch_size, 1, 1)) # [batch_size, 2, dim]
        input_emb = torch.cat([word_embs, prompt_emb, mlm_embeding], dim=1)

        prompt_mask = torch.ones([batch_size, self.prompt_num+2], device=self.device)
        masks = torch.cat([masks[:, :-1], prompt_mask], dim=1)

        prompt_seg_ids = torch.ones([batch_size, self.prompt_num+2],
                                    dtype=torch.long,
                                    device=self.device)*int(seg_ids.max()+1)
        seg_ids = torch.cat([seg_ids[:, :-1], prompt_seg_ids], dim=1)

        encoder_dict = self.bert.forward(
            input_ids=None,
            attention_mask=masks,
            token_type_ids=seg_ids,
            inputs_embeds=input_emb
        )
        hidden_state =  encoder_dict.last_hidden_state
        return hidden_state[:, -1, :]

    def mlm_predict(self, batch, temperature=1.0):
        vecs = self.Batch2Vecs(batch)
        logits = self.cls(vecs)
        pred = F.softmax(logits/temperature, dim=1)
        return pred

    def predict(self, batch, temperature=1.0):
        pred = self.mlm_predict(batch, temperature)
        return pred[:, self.label_token_id]

    def lossAndAcc(self, batch, temperature=1.0, label_weight:torch.Tensor=None, reduction='mean'):
        pred = self.mlm_predict(batch, temperature)
        label:torch.Tensor = batch[-2]
        if label.dim() == 2:
            label = label.argmax(dim=1)
        mlm_label = self.label_token_id[label.to(self.device)]
        loss = F.nll_loss(pred.log(), mlm_label)

        hard_p = pred.data[:, self.label_token_id].argmax(dim=1)
        acc = (label - hard_p).__eq__(0).float().sum() / len(hard_p)
        return loss, acc

class PromptBERTV2(PromptBERT):
    def init_prototypical_vector(self, data_set:CustomDataset, batch_size=20):
        labelTensor = data_set.labelTensor()

        if labelTensor.dim() == 1:
            class_num = labelTensor.max()
            indices = torch.stack([
                torch.arange(len(labelTensor), dtype=torch.long, device=labelTensor.device),
                labelTensor
            ])
            labelTensor = torch.sparse_coo_tensor(
                indices=indices,
                values=torch.ones_like(labelTensor),
                dtype=torch.float32,
                device=labelTensor.device
            ).to_dense()
        else:
            class_num = labelTensor.argmax(dim=1).max()+1

        proto_vecs = None
        with torch.no_grad():
            for batch in tqdm(DataLoader(
                data_set, batch_size=batch_size, shuffle=False, collate_fn=data_set.collate_fn
            )):
                mlm_logits = self.mlm_predict(batch)
                if proto_vecs is None:
                    proto_vecs = torch.ones([class_num, mlm_logits.size(1)],
                                            dtype=torch.float32, device=mlm_logits.device)
                label:torch.Tensor = batch[-2]
                if label.dim() == 2:
                    label = label.argmax(dim=1)
                for c in range(class_num):
                    proto_vecs[c] += mlm_logits.data[label.__eq__(c)].sum(dim=0)

        proto_vecs = proto_vecs/(labelTensor.sum(dim=0).unsqueeze(1))
        self.proto_vecs = proto_vecs/(proto_vecs.norm(dim=1).unsqueeze(1))

    def predict(self, batch, temperature=1.0):
        if not hasattr(self, 'proto_vecs'):
            raise Exception('need to initialize the "proto_vecs" first!')
        logits = self.mlm_predict(batch, temperature)
        normed_logits = logits/(logits.norm(dim=1).unsqueeze(1))
        cosine = torch.matmul(normed_logits, self.proto_vecs.transpose(1, 0))
        return F.softmax(cosine/temperature, dim=1)

    def lossAndAcc(self, batch, temperature=1.0, label_weight:torch.Tensor=None, reduction='mean'):
        preds = self.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (
                    preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc


class Senti_BERT(BertForSequenceClassification, AdversarialModel):
    def __init__(self, bert_config):
        super(Senti_BERT, self).__init__(bert_config)

    def grouped_parameters(self, learning_rate):
        def lr_coefficient(par_name):
        # layer-wise fine-tuning
            if "layer." in par_name:
                layer_num = int(par_name.split("layer.")[1].split(".", 1)[0])
                return pow(0.8, 12 - layer_num)
            elif "embedding" in par_name:
                return pow(0.8, 13)
            else:
                return 1.0
        if learning_rate is None:
            learning_rate = self.learning_rate
        optimizerGroupedParameters = [{'params': p, 'lr': learning_rate * lr_coefficient(n)}
                                        for n, p in self.named_parameters()]
        return optimizerGroupedParameters

    def Batch2Vecs(self, batch):
        if batch[0].device != self.device:  # data is on a different device
            input_ids, masks, seg_ids = batch[0].to(self.device), \
                                        batch[1].to(self.device), \
                                        batch[2].to(self.device)
        else:
            input_ids, masks, seg_ids = batch[0], batch[1], batch[2]
        encoder_dict = self.bert.forward(
            input_ids=input_ids,
            attention_mask=masks,
            token_type_ids=seg_ids
        )
        return encoder_dict.pooler_output

    def predict(self, batch, temperature=1.0):
        if batch[0].device != self.device:  # data is on a different device
            input_ids, masks, seg_ids = batch[0].to(self.device), \
                                        batch[1].to(self.device), \
                                        batch[2].to(self.device)
        else:
            input_ids, masks, seg_ids = batch[0], batch[1], batch[2]
        encoder_dict = self.bert.forward(
            input_ids=input_ids,
            attention_mask=masks,
            token_type_ids=seg_ids
        )
        pooled_output = self.dropout(encoder_dict.pooler_output)
        logits = self.classifier(pooled_output)
        return F.softmax(logits / temperature, dim=1)

    def lossAndAcc(self, batch, temperature=1.0, label_weight=None, reduction='mean'):
        preds = self.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (
                    preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2]
        labels = labels.to(preds.device).argmax(dim=1) if labels.dim() == 2 else labels.to(preds.device)
        loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
        acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
        acc = acc_t.data.item()
        return loss, acc

    def set_aug_type(self, aug_type):
        self.bert.embeddings.aug_type = aug_type

class DomainDiscriminator(nn.Module, VirtualModel):
    def __init__(self, hidden_size, model_device, learningRate, domain_num):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size * 2, domain_num)).to(model_device)

        self.device = model_device
        self.learning_rate = learningRate

    def obtain_optim(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        return torch.optim.Adam(
            [{
                'params': self.discriminator.parameters(),
                'lr': learning_rate
            }]
        )

    def forward(self, vecs):
        return self.discriminator(vecs)

def obtain_model(bertPath, model_device, lr_model):
    bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
    tokenizer_M = BertTokenizer.from_pretrained(bertPath)
    bert_config.num_labels = 2
    bert_config.hidden_act = "relu"
    tokenizer_M.model_max_length = 256
    # Create the model
    bert = Senti_BERT.from_pretrained(bertPath, config=bert_config).to(model_device)
    bert.learning_rate = lr_model
    return bert, tokenizer_M

def obtain_adver_model(bertPath, model_device, lr_G, lr_D):
    bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
    tokenizer_M = BertTokenizer.from_pretrained(bertPath)
    bert_config.num_labels = 2
    bert_config.hidden_act = "relu" # gelu is not necessary in model finetuning
    tokenizer_M.model_max_length = 256
    # Create the model
    bert = Senti_BERT.from_pretrained(bertPath, config=bert_config).to(model_device)
    bert.learning_rate = lr_G
    discriminator = DomainDiscriminator(hidden_size=bert_config.hidden_size,
                                        model_device=model_device,
                                        learningRate=lr_D,
                                        domain_num=4)
    return bert, tokenizer_M, discriminator