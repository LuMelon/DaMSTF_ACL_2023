import sys, os, torch, pickle
sys.path.append("../")
from Data.AmazonLoader import Senti_domain_map, SentiDataset, PreTrainedTokenizer
from transformers import  BertConfig, BertTokenizer
from BaseModel.AmazonModel import Senti_BERT
import numpy as np
import torch.nn as nn
from TrainingEnv import GradientReversal

class DANN_Amazon_Model(Senti_BERT):
    def AdvDLossAndAcc(self, discriminator: nn.Module, batch, grad_reverse=False):
        f = self.Batch2Vecs(batch)
        if grad_reverse:
            f = GradientReversal.apply(f)
        predictions = discriminator(self.dropout(f)).softmax(dim=1)
        epsilon = torch.ones_like(predictions) * 1e-8
        preds = (predictions - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-1]
        loss, acc = self.loss_func(preds, labels, label_weight=None, reduction='mean')
        return loss, acc

def obtain_model(bertPath, model_device, lr_model):
    bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
    tokenizer_M = BertTokenizer.from_pretrained(bertPath)
    bert_config.num_labels = 2
    bert_config.hidden_act = "relu" # gelu is not necessary in model finetuning
    tokenizer_M.model_max_length = 256
    # Create the model
    bert = DANN_Amazon_Model.from_pretrained(bertPath, config=bert_config).to(model_device)
    bert.learning_rate = lr_model
    return bert, tokenizer_M

def obtain_domain_set(target_domain_name, tokenizer_M, database_name, few_shot_cnt=100):
    all_domains = list(Senti_domain_map.keys())
    source_domain_names = [d_name for d_name in all_domains if d_name != target_domain_name]
    print("source domains : ", source_domain_names)
    source_domain = SentiDataset(
        database_name = database_name,
        seen_domains = source_domain_names,
        tokenizer = tokenizer_M,
        max_data_size = -1,
        data_type = ['train', 'valid', 'test'],
        C_dimension=2,
        load_data = True
    )
    test_set = SentiDataset(
        database_name=database_name,
        seen_domains=[target_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['train'], #  70% of the labeled data in the target domain are taged with 'train'
        C_dimension=2,
        load_data=True
    )

    unlabeled_target = SentiDataset(
        database_name=database_name,
        seen_domains=[target_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=2000,
        data_type=['unlabeled'],
        C_dimension=2,
        load_data=True
    )
    valid_set = SentiDataset(
        database_name=database_name,
        seen_domains=[target_domain_name],
        tokenizer=tokenizer_M,
        max_data_size=-1,
        data_type=['test'], #  20% of the labeled data in the target domain are taged with 'test'
        C_dimension=2,
        load_data=True
    )
    if few_shot_cnt > 0:
        labeled_target = SentiDataset(
            database_name=database_name,
            seen_domains=[target_domain_name],
            tokenizer=tokenizer_M,
            max_data_size=few_shot_cnt,
            data_type=['valid'], #  20% of the labeled data in the target domain are taged with 'valid'
            C_dimension=2,
            load_data=True
        )
        return source_domain, valid_set, test_set, labeled_target, unlabeled_target
    else:
        return source_domain, valid_set, test_set, None, unlabeled_target


def RunTask(tID, fewShotCnt, logDir, bert_path, database_name="./Sentiment.db"):
    SentiDomainList = list(Senti_domain_map.keys())
    target_domain_name = SentiDomainList[tID]
    model, tokenizer_M = obtain_model(
                bert_path,
                torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
                lr_model=5e-5
    )
    tokenizer_M.max_len = 256
    source_domain, valid_set, test_set, labeled_target, unlabeled_target = obtain_domain_set(
        target_domain_name,
        tokenizer_M,
        database_name,
        few_shot_cnt=fewShotCnt
    )
    TestLabel = test_set.labelTensor()
    print("TestLabel : ", TestLabel.tolist())
    unlabeled_target.setLabel(np.zeros_like(unlabeled_target.label), list(range(len(unlabeled_target))))
    print("Zero ULabel : ", unlabeled_target.label.tolist())

    marker = f"Multi2{target_domain_name[0].upper()}"
    trainer = MMETrainer(seed=10086, log_dir=logDir, suffix=f"{target_domain_name}_FS{fewShotCnt}",
                            model_file=f"./DANN_{target_domain_name}_FS{fewShotCnt}.pkl",
                                class_num=2, temperature=0.05, batch_size=28, Lambda=0.1)
    print("marker : ", marker)
    # Pretrain Task-specific Model
    if os.path.exists(f"./PreTrainClassifier_{marker}.pkl"):
        model.load_state_dict(
            torch.load(f"./PreTrainClassifier_{marker}.pkl")
        )
    else:
        trainer.training(model, source_domain, batch_size=28, max_epoch=20, lr4model=5e-3,
                    dev_evaluator=None, test_evaluator=None,
                        grad_accum_cnt=1, valid_every=100, model_path="./PreTrainClassifier_T{newDomainName}.pkl")
        if os.path.exists(f"./PreTrainClassifier_T{target_domain_name}.pkl"):
            model.load_state_dict(
                torch.load(f"./PreTrainClassifier_T{target_domain_name}.pkl")
            )
        else:
            torch.save(model.state_dict(),
                       f"./PreTrainClassifier_T{target_domain_name}.pkl")
    trainer.ModelTrain(model, source_domain, labeled_target, unlabeled_target, valid_set, test_set,
                         maxEpoch=40, validEvery=100)


if __name__ == "__main__":
    logDir = str(__file__).rstrip(".py")
    # logDir = "OnlineTest"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not os.path.exists(logDir):
        os.system("mkdir %s"%logDir)
    else:
        os.system("rm -rf %s" % logDir)
        os.system("mkdir %s" % logDir)
    if not os.path.exists("./Caches/"):
        os.system("mkdir Caches")

    for target_ID in range(4):
        fewShotCnt = 100
        RunTask(target_ID, fewShotCnt, logDir, bert_path='../../bert_en/',
                    database_name = "../Sentiment.db")