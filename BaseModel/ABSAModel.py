import sys, torch, random
sys.path.append("..")
from TrainingEnv import VirtualModel
import torch.nn.functional as F
import torch.nn as nn
from BaseModel.modeling_bert import BertForSequenceClassification

class DomainDiscriminator(VirtualModel):
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

class ABSA_BERT(BertForSequenceClassification, VirtualModel):
    def __init__(self, bert_config):
        super(ABSA_BERT, self).__init__(bert_config)

    def obtain_optim(self, learning_rate=None):
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
        return torch.optim.Adam(optimizerGroupedParameters)

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


