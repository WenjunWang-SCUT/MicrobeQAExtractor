import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn

from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

# BioModel encapsulates the question-answering model (e.g., DeBERTaV3, BioBERT) from the Transformers library
class BioModel(nn.Module):
    def __init__(self, model_name, config, args):
        super().__init__()
        self.args = args
        
        # Load the existing question-answering model from the pretrained model name
        self.qa_bert = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            from_tf=False,
            config=config,
            cache_dir=None,
        )

    def process(
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

        # Prepare the input data based on whether the model is in training mode
        if is_training:
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

        # Forward pass through the question-answering model
        outputs =  self.qa_bert(**inputs)
        return self.process(batch, outputs)


@dataclass
class QAModelOutputWithClassify(QuestionAnsweringModelOutput):
    pred_impossible: Optional[Tuple[torch.FloatTensor]] = None

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
    

class BiClassifyCNNHeader(nn.Module):

    def __init__(self, in_channel) -> None:
        super().__init__()
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
    

############## classify ##############
class BioModelClassify(BioModel):
    __metaclass__ = ABCMeta

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, args)
        self.seq_len = seq_len
        self.bc_header = None
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

    def process(
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

    def loadBCHeader(self, root: str="./output/biclassify_header.pt"):
        if self.bc_header:
            self.bc_header.load_state_dict(torch.load(root))


class BioModelClassifyOne(BioModelClassify):

    def __init__(self, model_name, config, args, seq_len=384):
        super().__init__(model_name, config, args)
        self.bc_header = BiClassifyHeaderOne(seq_len)
        self.lossFunc = nn.MSELoss()

    def runClassify(self, batch: Tuple, outputs):
        embedding = outputs.hidden_states[-1]
        embedding = embedding.view(embedding.size(0), -1)
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
        embedding = outputs.hidden_states[-1]
        embedding = embedding.view(embedding.size(0), -1)
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
        embedding = outputs.hidden_states[-1]
        out = self.bc_header(embedding)
        is_impossible = batch[7].unsqueeze(1)
        truth = torch.cat((is_impossible, 1 - is_impossible), 1)
        loss = self.lossFunc(out, truth)
        return out[:, 0].unsqueeze(1), loss
