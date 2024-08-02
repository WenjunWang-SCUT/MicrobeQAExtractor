import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import sqrt, inf
from typing import List, Optional, Union, Tuple, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .original_model import BertForQuestionAnswering


@dataclass
class QAModelOutputWithClassify(QuestionAnsweringModelOutput):
    pred_impossible: Optional[Tuple[torch.FloatTensor]] = None # 输入GoldTruth也是Float

def padding(X: Tensor, length: int, value: float=-inf):
    padding_shape = X.shape[:-2] + (length - X.shape[-2], X.shape[-1])
    return torch.concat([X, torch.zeros(padding_shape, device=X.device)], dim=0) # 这里1024被我改成了X_len=384


def locateSEP(ids: List[int])->Tuple[int, int]:
    firs_idx = ids.index(102)
    second_idx = ids.index(102, firs_idx + 1)
    return firs_idx, second_idx


def locateSEPList(input_ids: List[List[int]])->List[Tuple[int, int]]:
    idxes = []
    for ids in input_ids:
        idxes.append(locateSEP(ids))
    return idxes

        
class AttentionQ2C(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.Wq = torch.nn.Linear(embedding_size, embedding_size)
        self.Wk = torch.nn.Linear(embedding_size, embedding_size)
        self.Wv = torch.nn.Linear(embedding_size, embedding_size)
        self.sqrt_dk = sqrt(embedding_size)

    def forward(self, full_ebd: Tensor, SEQ_idxes: List[Tuple[int, int]])->Tensor:
        Hs = []
        for i in range(len(full_ebd)):
            first_idx, second_idx = SEQ_idxes[i]
            question_part = full_ebd[i, 1: first_idx, :]
            context_part = full_ebd[i, 1 + first_idx: second_idx, :]
            Q = self.Wq(full_ebd[i])    # (token_len, embedding_size)
            K = self.Wk(question_part)  # (question_len, embedding_size)
            V = self.Wv(question_part)  # (question_len, embedding_size)
            attention = torch.softmax(Q @ torch.transpose(K, -1, -2) / self.sqrt_dk, -1) # (token_len, embedding_size) @ (embedding_size, question_len) -> (token_len, question_len)
            H = attention @ V           # (token_len, question_len) * (question_len, embedding_size) -> (token_len, embedding_size)
            Hs.append(H)
        return torch.stack(Hs)          #  -> (batch_size, token_len, embedding_size)


class AttentionC2Q(AttentionQ2C):

    def __init__(self, embedding_size):
        super().__init__(embedding_size)

    def forward(self, full_ebd: Tensor, SEQ_idxes: List[Tuple[int, int]]) -> Tensor:
        Hs = []
        for i in range(len(full_ebd)):
            first_idx, second_idx = SEQ_idxes[i]
            question_part = full_ebd[i, 1: first_idx, :]
            context_part = full_ebd[i, 1 + first_idx: second_idx, :]
            Q = self.Wq(full_ebd[i])    # (token_len, embedding_size)
            K = self.Wk(context_part)   # (context_len, embedding_size)
            V = self.Wv(context_part)   # (context_len, embedding_size)
            attention = torch.softmax(Q @ torch.transpose(K, -1, -2) / self.sqrt_dk, -1) # (token_len, embedding_size) @ (embedding_size, context_len) -> (token_len, context_len)
            H = attention @ V           # (token_len, context_len) * (context_len, embedding_size) -> (token_len, embedding_size)
            Hs.append(H)
        return torch.stack(Hs)          #  -> (batch_size, token_len, embedding_size)


class BiClassifyHeaderOne(nn.Module):

    def __init__(self, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(self.seq_len * 768, 1) # 全连接层full connect TODO: 前一个参数不能是固定值，应该是计算出来的才对
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.fc(x)
        out = self.sigmoid(x1)

        return out
    

class BiClassifyHeaderTwo(nn.Module):

    def __init__(self, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(self.seq_len * 768, 2) # 全连接层full connect TODO: 前一个参数不能是固定值，应该是计算出来的才对
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.fc(x)
        out = self.softmax(x1)

        return out
    

class BiClassifyCNNHeader(nn.Module): # 参考文献：Convolutional Neural Networks for Sentence Classification

    def __init__(self, in_channel) -> None:
        super().__init__()
        # self.conv2d = nn.Conv2d(in_channel, in_channel, 5, 1)
        self.conv1d = nn.Conv1d(in_channel, in_channel, 5, 1)
        self.maxpool = nn.MaxPool1d(764) # TODO: 最好能设计成动态变化的
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(384, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1d(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = torch.squeeze(out, 2)
        out = self.fc(out)

        return out
    
    
class BioModel(nn.Module):

    def __init__(self, model_name, config, args):
        super().__init__()
        self.args = args
        if args.load_remote_model: # load_remote: download model from huggingface
            self.biobert = AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                from_tf=False,
                config=config,
                cache_dir=None,
            )
        else:
            self.biobert = BertForQuestionAnswering(config)
            model_state = torch.load(model_name, map_location='cpu')
            model_state['bert.embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1))
            self.biobert.load_state_dict(model_state)

    def postProcess(
        self, 
        batch: Tuple,
        outputs, 
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return outputs

    def forward(
        self,
        batch: Tuple,
        is_training: bool=False,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:

        if is_training:
            if self.args.use_distloss:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "output_hidden_states": True,
                    "position_nums": batch[8], # 注意，这是distlosss模型中新加入的参数
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "output_hidden_states": True,
                }
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "output_hidden_states": True,
            }

        # if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
        #     del inputs["token_type_ids"]

        # if args.model_type in ["xlnet", "xlm"]:
        #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        #     if args.with_neg:
        #         inputs.update({"is_impossible": batch[7]})
        #     if hasattr(model, "config") and hasattr(model.config, "lang2id"):
        #         inputs.update(
        #             {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
        #         )
        outputs =  self.biobert(**inputs)
        return self.postProcess(batch, outputs)


############## 二分类头 ##############
class BioModelClassify(BioModel):
    __metaclass__ = ABCMeta

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, args)
        self.seq_len = seq_len
        self.bc_header = None
        ## lock model's parameters
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        # ==> 
        self.biobert.requires_grad_(False)
        self.lossFunc = None

    @abstractmethod
    def runClassify(
            self, 
            batch: Tuple, 
            outputs
        ):
        # TODO: 拿到outputs.hidden_states的最后一层，然后经过二分类头，输出output2，计算loss，加入到原先的loss，然后返回result
        pass

    def postProcess(
            self, 
            batch: Tuple, 
            outputs
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        out, bc_loss = self.runClassify(batch, outputs)

        return QAModelOutputWithClassify(
            loss=bc_loss,
            start_logits=outputs.start_logits,
            end_logits=outputs.end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pred_impossible=out,
        )

    def saveBCHeader(self, dir: str="", name: str="biclassify_header.pt"):
        torch.save(self.bc_header.state_dict(), os.path.join(dir, name))
        ## load
        # model = BiClassifyHeader(seq_len)
        # model.load_state_dict(torch.load(PATH))

    def loadBCHeader(self, root: str="./output/biclassify_header.pt"):
        if self.bc_header:
            self.bc_header.load_state_dict(torch.load(root))


class BioModelClassifyOne(BioModelClassify):

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, args)
        self.bc_header = BiClassifyHeaderOne(seq_len)
        self.lossFunc = nn.MSELoss()

    def runClassify(self, batch: Tuple, outputs):
        embedding = outputs.hidden_states[-1] # 12*384*768
        embedding = embedding.view(embedding.size(0), -1) # 12*294912
        out = self.bc_header(embedding)
        is_impossible = batch[7]
        loss = self.lossFunc(out, is_impossible.unsqueeze(1))
        return out, loss
    

class BioModelClassifyTwo(BioModelClassify):

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, seq_len, args)
        self.bc_header = BiClassifyHeaderTwo(seq_len)
        self.lossFunc = nn.CrossEntropyLoss()

    def runClassify(self, batch: Tuple, outputs):
        embedding = outputs.hidden_states[-1] # 12*384*768
        embedding = embedding.view(embedding.size(0), -1) # 12*294912
        out = self.bc_header(embedding)
        is_impossible = batch[7].unsqueeze(1)
        truth = torch.cat((is_impossible, 1 - is_impossible), 1)
        loss = self.lossFunc(out, truth)
        return out[:, 0].unsqueeze(1), loss


class BioModelClassifyCNN(BioModelClassify):

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, seq_len, args)
        self.bc_header = BiClassifyCNNHeader(seq_len)
        self.lossFunc = nn.CrossEntropyLoss()

    def runClassify(self, batch: Tuple, outputs):
        embedding = outputs.hidden_states[-1] # 12*384*768
        # embedding = embedding.view(embedding.size(0), -1) # 12*294912
        out = self.bc_header(embedding)
        is_impossible = batch[7].unsqueeze(1)
        truth = torch.cat((is_impossible, 1 - is_impossible), 1)
        loss = self.lossFunc(out, truth)
        return out[:, 0].unsqueeze(1), loss


############## 注意力机制 ##############
class BioModelExtend(BioModel):
    __metaclass__ = ABCMeta

    def __init__(self, model_name, config, args):
        super().__init__(model_name, config, args)
        self.config = config
        for p in self.biobert.parameters():
            p.requires_grad = False

    @abstractmethod
    def getLogits( # custom
        self, 
        outputs: Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput], 
        input_ids: Optional[torch.Tensor] = None,
    )->Tensor:
        pass

    def postProcess(
        self, 
        batch: Tuple,
        outputs, 
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        input_ids = batch[0]
        start_positions = batch[3]
        end_positions = batch[4]

        return_dict = self.config.use_return_dict
        logits = self.getLogits(outputs, input_ids)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BioModelQkv(BioModelExtend):

    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        self.attention_q2c = AttentionQ2C(config.hidden_size)
        self.linear1 = torch.nn.Linear(config.hidden_size, 2)
        self.sotfmax = torch.nn.Softmax(dim=-2)

    def getLogits(
        self, 
        outputs: Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput], 
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tensor:
        encoder_layers = outputs.hidden_states
        embedding = encoder_layers[-1] # 12*384*768
        idxes_list = locateSEPList(input_ids.cpu().numpy().tolist())
        q2c = self.attention_q2c(embedding, idxes_list)
        logits = self.linear1(q2c)
        logits = self.sotfmax(logits)
        return logits


class BioModelQkvBiDirection(BioModelQkv):
    
    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        self.linear1 = torch.nn.Linear(config.hidden_size * 2, 2)
        self.attention_c2q = AttentionC2Q(config.hidden_size)

    def getLogits(
        self, 
        outputs: Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput], 
        input_ids: Optional[torch.Tensor] = None
    ) -> Tensor:
        encoder_layers = outputs.hidden_states
        embedding = encoder_layers[-1] # 12*384*768
        idxes_list = locateSEPList(input_ids.cpu().numpy().tolist())
        q2c = self.attention_q2c(embedding, idxes_list)
        c2q = self.attention_c2q(embedding, idxes_list)
        attention = torch.concat((q2c, c2q), -1)
        logits = self.linear1(attention)
        logits = self.sotfmax(logits)
        return logits       


class BioModelQkvBiDirectionResidual(BioModelQkvBiDirection):

    def __init__(self, model_name, config, pad_zero: bool=False):
        super().__init__(model_name, config)
        self.linear1 = torch.nn.Linear(config.hidden_size * 2, 2)
        self.pad_zero = pad_zero

    def getLogits(
        self, 
        outputs: Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput], 
        input_ids: Optional[torch.Tensor] = None
    ) -> Tensor:
        encoder_layers = outputs.hidden_states
        embedding = encoder_layers[-1] # 12*384*768
        idxes_list = locateSEPList(input_ids.cpu().numpy().tolist())
        q2c = self.attention_q2c(embedding, idxes_list)
        c2q = self.attention_c2q(embedding, idxes_list)
        attention = torch.concat((q2c, c2q), -1)

        # skip connection
        if self.pad_zero:
            residual = torch.concat((embedding, torch.zeros_like(embedding)), -1) + attention # 用0填充
        else:
            residual = torch.concat((embedding, embedding), -1) + attention

        logits = self.linear1(residual)
        logits = self.sotfmax(logits)
        return logits   
    
