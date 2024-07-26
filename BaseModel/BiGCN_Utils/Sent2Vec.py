import sys, os, pickle, random, pdb, re
sys.path.append("..")
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers.models.bert import BertTokenizer, BertModel, BertConfig
from torch import Tensor, device
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from gensim.models.doc2vec import Doc2Vec
from transformers.models.bert.modeling_bert import BertLayer
from ..modeling_bert import BertEmbeddings
from functools import reduce
from nltk.stem import WordNetLemmatizer

def mask_tokens(inputs, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class SentenceModel(nn.Module):
    def tokens2vecs(self, sents, attn_mask=None):
        raise NotImplementedError("'tokens2vecs' is not impleted")

    def tokens2vecs_aug(self, sents, attn_mask=None):
        raise NotImplementedError("'tokens2vecs_aug' is not impleted")

    def AugForward(self, sents):
        raise NotImplementedError("'AugForward' is not impleted")

    def forward(self, sents):
        raise NotImplementedError("'forward' is not impleted")

    def save_model(self, model_file):
        raise NotImplementedError("'save_model' is not impleted")

    def load_model(self, model_file):
        raise NotImplementedError("'load_model' is not impleted")

class W2VBasedModel(SentenceModel):
    def __init__(self, w2v_dir, seg=None):
        super(W2VBasedModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with open(os.path.join(w2v_dir, "vocab.txt"), "r") as fr:
            self.vocab = [line.strip('\n') for line in fr]
        self.word2index = {w:idx for idx, w in enumerate(self.vocab)}
        self.embedding_size = 300
        self.emb = nn.Embedding(len(self.vocab), embedding_dim=self.embedding_size).to(self.device)
        state_dict = torch.load(os.path.join(w2v_dir, "embedding.pkl"))
        self.emb.load_state_dict(state_dict['embedding'])
        self.seg = seg

    def encode_sentence(self, sent):
        if reduce(lambda x, y: x and y, map(lambda x:type(x)==int, sent)): # if sent is consistuted with the word indexs
            return sent
        if self.seg is None:
            tokens = sent.split()
        else:
            tokens = self.seg.cut(sent)
        idxs = [self.word2index[token] if token in self.word2index else self.word2index['[UNK]'] for token in tokens]
        return idxs

    def Pad_Sequence(self, ipt_ids, mlm_labels=None):
        max_sent_len = max([len(ids) for ids in ipt_ids])
        ipt_tensors = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64, device=self.device)
        attn_masks = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64, device=self.device)
        if mlm_labels is not None:
            labels = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64, device=self.device) * -1
        for i in range(len(ipt_ids)):
            ipt_tensors[i, :len(ipt_ids[i])] = ipt_ids[i]
            if mlm_labels is not None:
                labels[i, :len(ipt_ids[i])] = mlm_labels[i]
            attn_masks[i, len(ipt_ids[i]):] = 0
        if mlm_labels is None:
            return ipt_tensors, attn_masks
        else:
            return ipt_tensors, attn_masks, labels

    def sents2mlm_ids(self, sents, mlm_probs):
        print("Error: sents2mlm_ids is not implemented for W2VBasedModel in this version")
        sys.exit(0)

    def sents2ids(self, sents):
        text_inputs = [torch.tensor(self.encode_sentence(sent)) for sent in sents]
        input_ids, att_masks = self.Pad_Sequence(text_inputs)
        return input_ids, att_masks

class TokenizerBasedModel(SentenceModel):
    def __init__(self):
        super(TokenizerBasedModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def Pad_Sequence(self, ipt_ids, mlm_labels=None):
        try:
            max_sent_len = max([len(ids) for ids in ipt_ids])
            if max_sent_len > 512:
                ipt_ids = [t if len(t)<512 else t[:512] for t in ipt_ids]
                max_sent_len = 512
            ipt_tensors = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64) * 102
            attn_masks = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64)
            if mlm_labels is not None:
                labels = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64) * -1
            for i in range(len(ipt_ids)):
                ipt_tensors[i, :len(ipt_ids[i])] = ipt_ids[i]
                if mlm_labels is not None:
                    labels[i, :len(ipt_ids[i])] = mlm_labels[i]
                attn_masks[i, len(ipt_ids[i]):] = 0
        except:
            pdb.set_trace()
            raise
        if mlm_labels is None:
            return ipt_tensors.to(self.device), attn_masks.to(self.device)
        else:
            return ipt_tensors.to(self.device), attn_masks.to(self.device), labels.to(self.device)

    def sents2mlm_ids(self, sents, mlm_probs):
        try:
            text_inputs = [mask_tokens(torch.tensor(self.tokenizer.encode(sent, add_special_tokens=True)), self.tokenizer, mlm_probs) for sent in sents]
            ids = [items[0] for items in text_inputs]
            labels = [items[1] for items in text_inputs]
            input_ids, att_masks, masked_lm_labels = self.Pad_Sequence(ids, labels)
        except:
            pdb.set_trace()
            raise
        return input_ids, att_masks, masked_lm_labels

    def sents2ids(self, sents):
        text_inputs = [torch.tensor(self.tokenizer.encode(sent, add_special_tokens=True)) for sent in sents]
        input_ids, att_masks = self.Pad_Sequence(text_inputs)
        return input_ids, att_masks

class CN_TokenizerBasedModel(TokenizerBasedModel):
    def __init__(self):
        super(CN_TokenizerBasedModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        self.lemmatizer = WordNetLemmatizer()

    def contains_CN_Char(self, string):
        return self.zh_pattern.search(string)

    def lemma(self, word):
        word_n = self.lemmatizer.lemmatize(word, 'n')
        word_a = self.lemmatizer.lemmatize(word_n, 'a')
        word_v= self.lemmatizer.lemmatize(word_a, 'v')
        word_r = self.lemmatizer.lemmatize(word_v, 'r')
        return word_r

    def sent_to_tokens(self, sent):
        words = sent.split(" ")
        tokens = [self.tokenizer.cls_token]
        for word in words:
            if self.contains_CN_Char(word):
                tokens.extend(list(word))
            else:
                tokens.append(self.lemma(word))
            if len(tokens) + 1 > self.bert_config.max_position_embeddings:
                break
        return tokens[:self.bert_config.max_position_embeddings-1] + [self.tokenizer.sep_token]

    def sents2mlm_ids(self, sents, mlm_probs):
        if reduce(lambda x, y: x and y, map(lambda x:type(x)==int, sents[0])): # if sent is consistuted with the word indexs
            text_inputs = [mask_tokens(torch.tensor(tokens), self.tokenizer, mlm_probs) for tokens in sents]
        else:
            sent_tokens = [self.sent_to_tokens(sent) for sent in sents]
            text_inputs = [mask_tokens(torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)), self.tokenizer, mlm_probs) for tokens in sent_tokens]
        ids = [items[0] for items in text_inputs]
        labels = [items[1] for items in text_inputs]
        input_ids, att_masks, masked_lm_labels = self.Pad_Sequence(ids, labels)
        return input_ids, att_masks, masked_lm_labels

    def sents2ids(self, sents):
        if reduce(lambda x, y: x and y, map(lambda x:type(x)==int, sents[0])): # if sent is consistuted with the word indexs
            text_inputs = [torch.tensor(tokens) for tokens in sents]
        else:
            sent_tokens = [self.sent_to_tokens(sent) for sent in sents]
            text_inputs = [torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)) for tokens in sent_tokens]
        input_ids, att_masks = self.Pad_Sequence(text_inputs)
        return input_ids, att_masks

class Para2Vec(SentenceModel):
    def __init__(self, model_file):
        super(Para2Vec, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_file = model_file
        if os.path.exists(model_file):
            self.model = Doc2Vec.load(model_file)
        else:
            print("model file %s not exists!" % model_file)
            sys.exit(0)

    def forward(self, sents):
        sent_tensors = torch.tensor(
                                    np.stack([self.model.infer_vector(sent) for sent in sents]),
                                    device=self.device
        )
        return sent_tensors

    def load_model(self, model_file):
        if os.path.exists(model_file):
            self.model = Doc2Vec.load(model_file)
        elif os.path.exists(model_file):
            self.model = Doc2Vec.load(self.model_file)
        else:
            print("model file %s not exists!" % model_file)
            sys.exit(0)

class TFIDFBasedVec(SentenceModel):
    def __init__(self, pretrained_vectorizer, top_K, embedding_size, w2v_dir,
                 emb_update=True, grad_preserve=False, aug_type=None):
        """
        :param pretrained_vectorizer:
        :param top_K:
        :param embedding_size:
        :param w2v_dir:
        :param emb_update:
        :param grad_preserve:
        :param aug_type: g_blur, gaussian, adver, rMask, rReplace, mix
        """
        super(TFIDFBasedVec, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = embedding_size
        self.vectorizer = pretrained_vectorizer
        self.top_K = top_K
        with open(os.path.join(w2v_dir, "vocab.txt"), "r") as fr:
            self.vocab = [line.strip('\n') for line in fr]
        self.word2index = {w:idx for idx, w in enumerate(self.vocab)}
        self.embedding_size = 300
        state_dict:torch.Tensor = torch.load(os.path.join(w2v_dir, "embedding.pkl"))['embedding']['weight']
        emb = torch.zeros(
            [len(self.vectorizer.vocabulary_), embedding_size],
            dtype=torch.float32,
            device=state_dict.device
        )
        oov_count = 0
        for word in self.vectorizer.vocabulary_.keys():
            if word in self.word2index:
                emb[self.vectorizer.vocabulary_[word]] += state_dict[ self.word2index[word] ]
            else:
                emb[self.vectorizer.vocabulary_[word]] += (state_dict[self.word2index['unknow']] + state_dict[self.word2index['word']])
                oov_count += 1

        self.embedding = nn.Embedding.from_pretrained(
            emb
        ).to(self.device, non_blocking=True)
        print("self.embedding.weight.requires_grad", self.embedding.weight.requires_grad)
        self.emb_update = emb_update
        assert emb_update == grad_preserve
        if emb_update:
            print("requires_grad = True")
            for par in self.embedding.parameters():
                par.requires_grad = True
        else:
            print("requires_grad = False")
            for par in self.embedding.parameters():
                par.requires_grad = False
        self.aug_type = aug_type
        print("OOV Count:", oov_count)
        print("OOV Ratio:", oov_count*1.0/len(self.vectorizer.vocabulary_))
        del state_dict, emb

    def forward(self, sents):
        tfidf_arr = torch.tensor(self.vectorizer.transform(sents).toarray(),
                                 dtype=torch.float32, device=self.device)
        sort_vals, sort_idxs = tfidf_arr.sort(dim=1)
        token_ids = sort_idxs[:, -self.top_K:]
        weights = sort_vals[:, -self.top_K:].unsqueeze(-1)
        X = self.embedding(token_ids)
        if self.aug_type == "gaussian":
            X = self.gaussian_aug(X)
        elif self.aug_type == "g_blur":
            X = self.gaussian_blur(X)
        elif self.aug_type == "adver":
            X = self.adversarial_aug(X, token_ids)
        elif self.aug_type == "rMask":
            X = self.randomMask(X)
        elif self.aug_type == "rReplace":
            X = self.randomReplace(X)
        elif self.aug_type is None:
            pass
        else:
            print(f"!!!! aug_type '{self.aug_type}' is not impleted, please re-check it!")
        sent_vec = (weights * X).mean(dim=1)
        return sent_vec

    def gaussian_aug(self, tensor):
        print("gaussian_aug")
        if random.random() < 0.7:
            return tensor + 5e-3 * torch.randn(tensor.shape, device=tensor.device)
        else:
            return tensor

    def adversarial_aug(self, tensor, token_idxs):
        print("adversarial_aug")
        if self.embedding.weight.grad is None:
            print("no gradient")
            return tensor
        noise = self.embedding.weight.grad[token_idxs].clone()
        return tensor + 5e-3*noise

    def randomMask(self, tensor):
        print("randomMask")
        sent_num, sent_len = tensor.size(0), tensor.size(1)
        mask = torch.tensor([[0 if random.random() < 0.2 else 1 for _ in range(sent_len)]
                             for _ in range(sent_num)], device=self.device).unsqueeze(-1)
        tensor = tensor * mask
        return tensor

    def cut_off(self, tensor:torch.Tensor):
        print("Aug: cut_off")
        if not hasattr(self, 'cut_off_prob'):
            self.cut_off_prob = 0.1
        batch_size, seq_len, feature_dim = tensor.shape
        col_mask = torch.ones(
            [batch_size, feature_dim],
            dtype=torch.float32,
            device=tensor.device
        ).bernoulli(1.0-self.cut_off_prob).unsqueeze(1)
        return tensor*col_mask

    def randomReplace(self, tensor:torch.Tensor):
        print("randomReplace")
        sent_num, sent_len = tensor.size(0), tensor.size(1)
        replace_mask = torch.tensor([[1 if random.random() < 0.2 else 0 for _ in range(sent_len)]
                             for _ in range(sent_num)], device=self.device).unsqueeze(-1)
        tensor = tensor + replace_mask * torch.rand_like(tensor, device=tensor.device)
        return tensor

    def gaussian_blur(self, tensor):
        """
        :param tensor: [batch_size, topK, dim]
        :return:
        """
        print("gaussian_blur")
        if not hasattr(self, "gBlur_kernel"):
            cent = (1, 1) #kernel shape: (3, 3)
            dist = torch.tensor(
                [ [ np.sqrt((i-cent[0])*(i-cent[0]) + (j-cent[1])*(j-cent[1]))
                        for j in range(3)]
                        for i in range(3)],
                device=tensor.device,
                dtype=torch.float32
            )
            self.gBlur_kernel = (1.0 / 2*np.pi) * torch.pow(np.e, -1*dist)
            self.gBlur_kernel = self.gBlur_kernel/self.gBlur_kernel.sum()
        return F.conv2d(
                        tensor.unsqueeze(1), # [batchsize, 1, topK, dim]
                        self.gBlur_kernel.unsqueeze(0).unsqueeze(0), # [1, 1, kW, kH]
                        stride=(1, 1),
                        padding=1,
                        dilation=1
                    ).squeeze(1) # [batchsize, topK, dim]

    def AugForward(self, sents):
        return self.forward(sents)

    def set_aug_type(self, type=None):
        self.aug_type = type

    def save_model(self, model_file):
        if not self.emb_update:
            pass
        else:
            torch.save(
                {
                "embedding": self.embedding.state_dict()
                 },
            model_file
            )

    def load_model(self, pretrained_file):
        if self.emb_update:
            ch = torch.load(pretrained_file)
            self.embedding.load_state_dict(ch['embedding'])

class TFIDFBasedVecV2(TFIDFBasedVec):
    def __init__(self, pretrained_vectorizer, top_K, embedding_size, w2v_dir,
                 emb_update=True, grad_preserve=False, aug_type=None):
        """
        :param pretrained_vectorizer:
        :param top_K:
        :param embedding_size:
        :param w2v_dir:
        :param emb_update:
        :param grad_preserve:
        :param aug_type: g_blur, gaussian, adver, rMask, rReplace, mix
        """
        super(TFIDFBasedVecV2, self).__init__(pretrained_vectorizer, top_K, embedding_size, w2v_dir,
                                              emb_update, grad_preserve, aug_type)

    def forward(self, tfidf_arr:torch.Tensor):
        sort_vals, sort_idxs = tfidf_arr.sort(dim=1)
        token_ids = sort_idxs[:, -self.top_K:]
        weights = sort_vals[:, -self.top_K:].unsqueeze(-1)
        X = self.embedding(token_ids)
        if self.aug_type == "gaussian":
            X = self.gaussian_aug(X)
        elif self.aug_type == "g_blur":
            X = self.gaussian_blur(X)
        elif self.aug_type == "adver":
            X = self.adversarial_aug(X, token_ids)
        elif self.aug_type == "rMask":
            X = self.randomMask(X)
        elif self.aug_type == "rReplace":
            X = self.randomReplace(X)
        elif self.aug_type == "cutoff":
            X = self.cut_off(X)
        elif self.aug_type is None:
            pass
        else:
            raise NotImplementedError(f"aug_type '{self.aug_type}' is not impleted, please re-check it!")
        sent_vec = (weights * X).mean(dim=1)
        return sent_vec

class TFIDFBasedVec_CN(SentenceModel):
    def __init__(self, pretrained_vectorizer, top_K, embedding_size, w2v_file,
                 emb_update=True, grad_preserve=False, aug_type=None):
        super(TFIDFBasedVec_CN, self).__init__()
        assert emb_update == grad_preserve
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = embedding_size
        self.vectorizer = pretrained_vectorizer
        self.top_K = top_K
        with open(w2v_file, "rb") as fr:
            w2v = pickle.load(fr)
        emb = np.zeros([len(self.vectorizer.vocabulary_), embedding_size], dtype=np.float32)
        oov_count = 0
        for word in self.vectorizer.vocabulary_.keys():
            if word in w2v:
                emb[self.vectorizer.vocabulary_[word]] += w2v[word]
            else:
                emb[self.vectorizer.vocabulary_[word]] += (w2v['未知'] + w2v['词'])
                oov_count += 1
        print("OOV Count:", oov_count)
        print("OOV Ratio:", oov_count * 1.0 / len(self.vectorizer.vocabulary_))
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(emb, dtype=torch.float32)
        ).to(self.device)
        self.emb_update = emb_update
        if emb_update:
            print("requires_grad = False")
            for par in self.embedding.parameters():
                par.requires_grad = True
        else:
            print("requires_grad = False")
            for par in self.embedding.parameters():
                par.requires_grad = False
        del w2v, emb
        self.aug_type = aug_type
        self.noise_rate = 5e-4
        self.max_dist, self.min_dist = 10, 1e-3

    def gaussian_aug(self, tensor):
        print("gaussian_aug")
        if random.random() < 0.7:
            return tensor + 5e-3 * torch.randn(tensor.shape, device=tensor.device)
        else:
            return tensor

    def adversarial_aug(self, tensor, token_idxs):
        # print("adversarial_aug")
        if self.embedding.weight.grad is None:
            print("no gradient")
            return tensor
        noise = self.embedding.weight.grad[token_idxs].clone()
        noise_norm = noise.norm(2)
        # print("noise_norm : ", noise_norm)
        if noise_norm > self.max_dist: # PGD step: Projecting
            noise = self.max_dist*(noise / noise_norm)
        if noise_norm < self.min_dist: # PGD step: Projecting
            noise = self.max_dist*(noise / (noise_norm + 1e-10))
        return tensor + self.noise_rate*noise

    def randomMask(self, tensor):
        print("randomMask")
        sent_num, sent_len = tensor.size(0), tensor.size(1)
        mask = torch.tensor([[0 if random.random() < 0.2 else 1 for _ in range(sent_len)]
                             for _ in range(sent_num)], device=self.device).unsqueeze(-1)
        tensor = tensor * mask
        return tensor

    def randomReplace(self, tensor:torch.Tensor):
        print("randomReplace")
        sent_num, sent_len = tensor.size(0), tensor.size(1)
        replace_mask = torch.tensor([[1 if random.random() < 0.2 else 0 for _ in range(sent_len)]
                             for _ in range(sent_num)], device=self.device).unsqueeze(-1)
        tensor = tensor + replace_mask * torch.rand_like(tensor, device=tensor.device)
        return tensor

    def gaussian_blur(self, tensor):
        """
        :param tensor: [batch_size, topK, dim]
        :return:
        """
        print("gaussian_blur")
        if not hasattr(self, "gBlur_kernel"):
            cent = (1, 1) #kernel shape: (3, 3)
            dist = torch.tensor(
                [ [ np.sqrt((i-cent[0])*(i-cent[0]) + (j-cent[1])*(j-cent[1]))
                        for j in range(3)]
                        for i in range(3)],
                device=tensor.device,
                dtype=torch.float32
            )
            self.gBlur_kernel = (1.0 / 2*np.pi) * torch.pow(np.e, -1*dist)
            self.gBlur_kernel = self.gBlur_kernel/self.gBlur_kernel.sum()
        return F.conv2d(
                        tensor.unsqueeze(1), # [batchsize, 1, topK, dim]
                        self.gBlur_kernel.unsqueeze(0).unsqueeze(0), # [1, 1, kW, kH]
                        stride=(1, 1),
                        padding=1,
                        dilation=1
                    ).squeeze(1) # [batchsize, topK, dim]

    def set_aug_type(self, type=None):
        self.aug_type = type

    def AugForward(self, sents):
        return self.forward(sents)

    def forward(self, sents):
        tfidf_arr = self.vectorizer.transform(sents).toarray()
        token_ids = torch.tensor(tfidf_arr.argsort(axis=1)[:, -self.top_K:]).to(self.device)
        weights = torch.tensor(np.sort(tfidf_arr, axis=1)[:, -self.top_K:], dtype=torch.float32).unsqueeze(-1).to(self.device)
        X = self.embedding(token_ids)
        if self.aug_type == "gaussian":
            X = self.gaussian_aug(X)
        elif self.aug_type == "g_blur":
            X = self.gaussian_blur(X)
        elif self.aug_type == "adver":
            X = self.adversarial_aug(X, token_ids)
        elif self.aug_type == "rMask":
            X = self.randomMask(X)
        elif self.aug_type == "rReplace":
            X = self.randomReplace(X)
        elif self.aug_type is None:
            pass
        else:
            print(f"!!!! aug_type '{self.aug_type}' is not impleted, please re-check it!")
        sent_vec = (weights * X).mean(dim=1)
        return sent_vec

    def save_model(self, model_file):
        if not self.emb_update:
            pass
        else:
            torch.save(
                {
                "embedding": self.embedding.state_dict()
                 },
            model_file
            )

    def load_model(self, model_file):
        if self.emb_update:
            ch = torch.load(model_file)
            self.embedding.load_state_dict(ch['embedding'])

class TokenizerBasedModel(SentenceModel):
    def __init__(self):
        super(TokenizerBasedModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def Pad_Sequence(self, ipt_ids, mlm_labels=None):
        try:
            max_sent_len = max([len(ids) for ids in ipt_ids])
            if max_sent_len > 512:
                ipt_ids = [t if len(t)<512 else t[:512] for t in ipt_ids]
                max_sent_len = 512
            ipt_tensors = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64) * 102
            attn_masks = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64)
            if mlm_labels is not None:
                labels = torch.ones([len(ipt_ids), max_sent_len], dtype=torch.int64) * -1
            for i in range(len(ipt_ids)):
                ipt_tensors[i, :len(ipt_ids[i])] = ipt_ids[i]
                if mlm_labels is not None:
                    labels[i, :len(ipt_ids[i])] = mlm_labels[i]
                attn_masks[i, len(ipt_ids[i]):] = 0
        except:
            pdb.set_trace()
            raise
        if mlm_labels is None:
            return ipt_tensors.to(self.device), attn_masks.to(self.device)
        else:
            return ipt_tensors.to(self.device), attn_masks.to(self.device), labels.to(self.device)

    def sents2mlm_ids(self, sents, mlm_probs):
        try:
            text_inputs = [
                mask_tokens(
                    torch.tensor(self.tokenizer.encode(sent, add_special_tokens=True)),
                    self.tokenizer, mlm_probs
                )
                for sent in sents
            ]
            ids = [items[0] for items in text_inputs]
            labels = [items[1] for items in text_inputs]
            input_ids, att_masks, masked_lm_labels = self.Pad_Sequence(ids, labels)
        except:
            pdb.set_trace()
            raise
        return input_ids, att_masks, masked_lm_labels

    def sents2ids(self, sents):
        text_inputs = [torch.tensor(self.tokenizer.encode(sent, add_special_tokens=True)) for sent in sents]
        input_ids, att_masks = self.Pad_Sequence(text_inputs)
        return input_ids, att_masks


class LSTMVec(TokenizerBasedModel):
    def __init__(self, bert_embedding, bert_dir, embedding_size, sent_hidden_size, num_layers=1, emb_update=False):
        super(LSTMVec, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.embedding_size = embedding_size
        self.sent_hidden_size = sent_hidden_size
        self.emb = nn.Embedding(len(self.tokenizer), self.embedding_size).to(device=self.device)
        assert self.sent_hidden_size % 2 ==0
        self.lstm = nn.LSTM(self.embedding_size, int(self.sent_hidden_size/2), num_layers=num_layers, bias=False, batch_first=True, bidirectional=True).to(device=self.device)
        if os.path.exists(bert_embedding):
            ch = torch.load(bert_embedding)
            self.emb.load_state_dict(ch['embeddings'])
        else:
            print("embedding_file %s not exists!" % bert_embedding)
            sys.exit(0)
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        embeddings = self.emb(ipt_ids)
        sent_len = attn_masks.sum(dim=1).tolist()
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, sent_len, batch_first=True, enforce_sorted=False)
        outs, (hiddens, cells) = self.lstm(packed)
        outs, len = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True)
        val_mask = attn_masks.unsqueeze(-1)
        return outs*val_mask, (len, hiddens, cells)

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        outs, _ = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = outs[:, 0, :] + outs[:, 1:, :].max(dim=1)[0]
        return sent_vecs

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.lstm.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                "embedding": self.emb.state_dict(),
                "sent2vec": self.lstm.state_dict()
                 },
            model_file
            )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])
        self.lstm.load_state_dict(ch['sent2vec'])

class BertEmb_LSTMVec(TokenizerBasedModel):
    def __init__(self, bert_dir, sent_hidden_size, num_layers=1, bert_embedding=None, emb_update=False):
        super(BertEmb_LSTMVec, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        config = BertConfig.from_pretrained(bert_dir)
        self.embedding_size = config.hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.emb = BertEmbeddings(config).to(self.device)
        assert self.sent_hidden_size % 2 ==0
        self.lstm = nn.LSTM(self.embedding_size, int(self.sent_hidden_size/2), num_layers=num_layers, bias=False, batch_first=True, bidirectional=True).to(device=self.device)
        if bert_embedding is not None and os.path.exists(bert_embedding):
            ch = torch.load(bert_embedding)
            self.emb.load_state_dict(ch['embeddings'])
        else:
            print("Warning:embedding_file %s not exists!" % bert_embedding)
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        embeddings = self.emb(ipt_ids)
        sent_len = attn_masks.sum(dim=1).tolist()
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, sent_len, batch_first=True, enforce_sorted=False)
        outs, (hiddens, cells) = self.lstm(packed)
        outs, len = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True)
        val_mask = attn_masks.unsqueeze(-1)
        return outs*val_mask, (len, hiddens, cells)

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        outs, _ = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = outs[:, 0, :] + outs[:, 1:, :].max(dim=1)[0]
        return sent_vecs

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.lstm.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                "embedding": self.emb.state_dict(),
                "sent2vec": self.lstm.state_dict()
             },
            model_file
            )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])(ch['embedding'])
        self.lstm.load_state_dict(ch['sent2vec'])

class BertEmb_RDMVec(TokenizerBasedModel):
    def __init__(self, bert_embedding, bert_dir, embedding_size, sent_hidden_size, emb_update=False):
        super(BertEmb_RDMVec, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.embedding_size = embedding_size
        self.sent_hidden_size = sent_hidden_size
        self.emb = nn.Embedding(len(self.tokenizer), self.embedding_size).to(device=self.device)
        self.linear = nn.Linear(self.embedding_size, self.sent_hidden_size).to(device=self.device)
        if os.path.exists(bert_embedding):
            ch = torch.load(bert_embedding)
            self.emb.load_state_dict(ch['embeddings'])
        else:
            print("embedding_file %s not exists!" % bert_embedding)
            sys.exit(0)
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        embeddings = self.emb(ipt_ids)
        outs = self.linear(embeddings)
        val_mask = attn_masks.unsqueeze(-1)
        return outs*val_mask

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        outs = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = outs.max(dim=1)[0]
        return sent_vecs

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.linear.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                "embedding": self.emb.state_dict(),
                "sent2vec": self.linear.state_dict()
             },
            model_file
            )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])(ch['embedding'])
        self.lstm.load_state_dict(ch['sent2vec'])

class BertVec(TokenizerBasedModel):
    def __init__(self, bert_dir, bert_parallel=False, para_update=True):
        super(BertVec, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.bert = BertModel.from_pretrained(bert_dir).to(self.device) # <----- zhai
        self.bert_config = self.bert.config
        self.sent_hidden_size = self.bert.config.hidden_size
        if bert_parallel:
            self.bert = nn.DataParallel(self.bert, device_ids=list(range(torch.cuda.device_count())))
        self.para_update = para_update

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        hiddens, outs = self.bert(ipt_ids, attention_mask=attn_masks)
        return hiddens, outs

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        if self.para_update:
            hiddens, outs = self.tokens2vecs(ipt_ids, attn_masks)
        else:
            with torch.no_grad():
                hiddens, outs = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = (hiddens * attn_masks.unsqueeze(-1)).mean(dim=1)
        return sent_vecs

    def save_model(self, model_file):
        torch.save(
            {
                "bert": self.bert.state_dict()
            },
            model_file
        )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        self.bert.load_state_dict(ch['bert'])

class BertVec_CN(CN_TokenizerBasedModel):
    def __init__(self, bert_dir, bert_parallel=False, para_update=True):
        super(BertVec_CN, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.bert = BertModel.from_pretrained(bert_dir).to(self.device)
        self.bert_config = self.bert.config
        self.sent_hidden_size = self.bert.config.hidden_size
        if bert_parallel:
            self.bert = nn.DataParallel(self.bert, device_ids=list(range(torch.cuda.device_count())))
        self.para_update = para_update

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        hiddens, outs = self.bert(ipt_ids, attention_mask=attn_masks)
        return hiddens, outs

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        if self.para_update:
            hiddens, outs = self.tokens2vecs(ipt_ids, attn_masks)
        else:
            with torch.no_grad():
                hiddens, outs = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = (hiddens * attn_masks.unsqueeze(-1)).mean(dim=1)
        return sent_vecs

    def save_model(self, model_file):
        torch.save(
            {
                "bert": self.bert.state_dict()
            },
            model_file
        )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        self.bert.load_state_dict(ch['bert'])

class W2V_Transformer(W2VBasedModel):
    def __init__(self, w2v_dir, config_file, emb_update=False):
        super(W2V_Transformer, self).__init__(w2v_dir)
        config = BertConfig.from_pretrained(config_file)
        self.transformer = BertLayer(config).to(self.device)
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        if self.emb_update:
            embeddings = self.emb(ipt_ids)
        else:
            with torch.no_grad():
                embeddings = self.emb(ipt_ids)
        outputs = self.transformer(embeddings, attention_mask=attn_masks)
        return outputs[0]

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        extended_attention_mask = self.get_extended_attention_mask(attn_masks, ipt_ids.shape, ipt_ids.device)
        hiddens = self.tokens2vecs(ipt_ids, extended_attention_mask)
        sent_vecs = hiddens.max(dim=1)[0]
        return sent_vecs

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.transformer.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.transformer.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                    "embedding": self.emb.state_dict(),
                    "transformer": self.transformer.state_dict()
                },
                model_file
            )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])(ch['embedding'])
        self.transformer.load_state_dict(ch['transformer'])

class W2VLSTMVec(W2VBasedModel):
    def __init__(self, w2v_dir, sent_hidden_size, num_layers=1, seg=None, emb_update=False):
        super(W2VBasedModel, self).__init__(w2v_dir, seg)
        self.sent_hidden_size = sent_hidden_size
        assert self.sent_hidden_size % 2 ==0
        self.lstm = nn.LSTM(self.embedding_size, int(self.sent_hidden_size/2), num_layers=num_layers, bias=False, batch_first=True, bidirectional=True).to(device=self.device)
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        embeddings = self.emb(ipt_ids)
        sent_len = attn_masks.sum(dim=1).tolist()
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, sent_len, batch_first=True, enforce_sorted=False)
        outs, (hiddens, cells) = self.lstm(packed)
        outs, len = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True)
        val_mask = attn_masks.unsqueeze(-1)
        return outs*val_mask, (len, hiddens, cells)

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        outs, _ = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = outs.max(dim=1)[0]
        return sent_vecs

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.lstm.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                "embedding": self.emb.state_dict(),
                "sent2vec": self.lstm.state_dict()
             },
            model_file
            )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])(ch['embedding'])
        self.lstm.load_state_dict(ch['sent2vec'])

class W2VLSTMVec_CN(W2VLSTMVec):
    def __init__(self, w2v_dir, sent_hidden_size, num_layers=1, seg=None, emb_update=False):
        super(W2VLSTMVec_CN, self).__init__(w2v_dir, sent_hidden_size, num_layers=num_layers, seg=seg, emb_update=emb_update)

class W2VRDMVec(W2VBasedModel):
    def __init__(self, w2v_dir, sent_hidden_size, seg=None, emb_update=False):
        super(W2VRDMVec, self).__init__(w2v_dir, seg)
        self.sent_hidden_size = sent_hidden_size
        assert self.sent_hidden_size % 2 ==0
        self.linear = nn.Linear(self.embedding_size, self.sent_hidden_size).to(device=self.device)
        self.emb_update = emb_update
        self.emb_update = emb_update
        if not emb_update:
            for par in self.emb.parameters():
                par.requires_grad = False

    def tokens2vecs(self, ipt_ids, attn_masks=None):
        embeddings = self.emb(ipt_ids)
        outs = self.linear(embeddings)
        val_mask = attn_masks.unsqueeze(-1)
        return outs*val_mask

    def forward(self, sents):
        ipt_ids, attn_masks = self.sents2ids(sents)
        outs = self.tokens2vecs(ipt_ids, attn_masks)
        sent_vecs = outs.max(dim=1)[0]
        return sent_vecs

    def save_model(self, model_file):
        if not self.emb_update:
            torch.save(
                {
                    "sent2vec": self.linear.state_dict()
                },
                model_file
            )
        else:
            torch.save(
                {
                "embedding": self.emb.state_dict(),
                "sent2vec": self.linear.state_dict()
             },
            model_file
        )

    def load_model(self, pretrained_file):
        ch = torch.load(pretrained_file)
        if self.emb_update:
            self.emb.load_state_dict(ch['embedding'])(ch['embedding'])
        self.linear.load_state_dict(ch['sent2vec'])