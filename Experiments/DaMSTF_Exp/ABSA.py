import sys, os
sys.path.append("../")
from Data.ABSALoader import Senti_domain_map, ABSA_Dataset
from transformers import  BertConfig, BertTokenizer
from BaseModel.ABSAModel import ABSA_BERT, DomainDiscriminator
from DomainAdaptationTrainer.MME import MMETrainer
import random, numpy as np
import torch, torch.nn as nn
from TrainingEnv import GradientReversal

class MME_ABSA_Model(ABSA_BERT):
    def AdvPredict(self, batch):
        f = self.Batch2Vecs(batch)
        f = GradientReversal.apply(f)
        predictions = self.classifier(self.dropout(f)).softmax(dim=1)
        return predictions

def obtain_model(bertPath, model_device, lr_model):
    bert_config = BertConfig.from_pretrained(bertPath, num_labels=3)
    tokenizer_M = BertTokenizer.from_pretrained(bertPath)
    bert_config.num_labels = 3
    bert_config.hidden_act = "relu"
    tokenizer_M.model_max_length = 256
    # Create the model
    bert = MME_ABSA_Model.from_pretrained(bertPath, config=bert_config).to(model_device)
    bert.learning_rate = lr_model
    return bert, tokenizer_M

def obtain_domain_set(new_domain_name, tokenizer_M, database_name, few_shot_cnt=100):
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


def RunTask(domainID, fewShotCnt, logDir, bert_path, database_name="./DA_ASBA.db"):
    SentiDomainList = list(Senti_domain_map.keys())
    newDomainName = SentiDomainList[domainID]
    model1, tokenizer_M = obtain_model(
        bert_path,
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        lr_model=5e-5
    )
    source_domain, val_set, test_target, labeled_target, unlabeled_target = obtain_domain_set(
        newDomainName,
        tokenizer_M,
        database_name,
        few_shot_cnt=fewShotCnt
    )
    TestLabel = test_target.labelTensor()
    print("TestLabel : ", TestLabel.tolist())
    unlabeled_target.setLabel(np.zeros_like(unlabeled_target.label),
                              list(range(len(unlabeled_target))))
    print("Zero ULabel : ", unlabeled_target.label.tolist())
    trainer = MMETrainer(seed=10086, log_dir=logDir, suffix=f"{newDomainName}_FS{fewShotCnt}",
                         model_file=f"./MME_{newDomainName}_FS{fewShotCnt}.pkl",
                         class_num=3, temperature=0.05, learning_rate=2e-5, batch_size=32,
                         Lambda=0.1)
    trainer.collate_fn = val_set.collate_raw_batch

    # Pretrain Task-specific Model
    if os.path.exists(f"./PreTrainClassifier_T{newDomainName}.pkl"):
        model1.load_state_dict(
            torch.load(f"./PreTrainClassifier_T{newDomainName}.pkl")
        )
    else:
        trainer.training(model1, source_domain, batch_size=28, max_epoch=20, lr4model=5e-3,
                    dev_evaluator=None, test_evaluator=None,
                        grad_accum_cnt=1, valid_every=100, model_path="./PreTrainClassifier_T{newDomainName}.pkl")
        if os.path.exists(f"./PreTrainClassifier_T{newDomainName}.pkl"):
            model1.load_state_dict(
                torch.load(f"./PreTrainClassifier_T{newDomainName}.pkl")
            )
        else:
            torch.save(model1.state_dict(),
                       f"./PreTrainClassifier_T{newDomainName}.pkl")

    trainer.valid(model1, val_set, val_set.labelTensor(), f"{trainer.suffix}_valid", 0)
    trainer.valid(model1, test_target, TestLabel, f"{trainer.suffix}_test", 0)
    trainer.ModelTrain(model1, source_domain, labeled_target, unlabeled_target, val_set, test_target,
                         maxEpoch=20, validEvery=20)

if __name__ == "__main__":
    logDir = str(__file__).rstrip(".py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # logDir = "OnlineTest"
    if not os.path.exists(logDir):
        os.system("mkdir %s"%logDir)
    else:
        os.system("rm -rf %s" % logDir)
        os.system("mkdir %s" % logDir)
    if not os.path.exists("./Caches/"):
        os.system("mkdir Caches")
    domainID = 0
    fewShotCnt = 100
    RunTask(domainID, fewShotCnt, logDir, bert_path='../../bert_en/', database_name="../DA_ASBA.db")