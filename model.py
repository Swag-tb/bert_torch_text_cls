#actor:NJUST_Tang Bin
#@file: model
#@time: 2021/12/28 17:17
#-*-coding:UTF-8-*-
from torch import nn
import torch
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel,BertSelfAttention
import logging
from torch.nn import CrossEntropyLoss
import numpy as np
logger=logging.getLogger(__name__)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:,1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertTorchClassfication(BertPreTrainedModel):
    '''
    通用文本分类模型
    '''
    base_model_prefix = "bert"

    def __init__(self,config):
        super(BertTorchClassfication, self).__init__(config)
        self.bert=BertModel(config)
        self.pooler=BertPooler(config)
        self.cls=nn.Linear(config.hidden_size,2)
        self.softmax=nn.Softmax()
        self.bert_attention=BertSelfAttention(config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,position_ids=None,
                index1=None,index2=None):
        '''
        :param input_ids: 使用tokenizer将输入文本转化为token_ids
        :param attention_mask: 0和1组成的mask向量，让模型知道应该学习哪一部分
        :param token_type_ids: segment向量
        :param labels: 标签
        :param position_ids:位置信息向量
        :param index1: 需要做attention的文本1位置信息
        :param index2: 需要做attention的文本2位置信息
        :return:
        '''
        outputs=self.bert.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids)
        pooled_output=outputs[1]#[CLS]的长度为768的向量
        '''
        #如果需要做attention的话，使用该部分,并且将线性回归的参数config.hidden_size修改为相应的大小*n
        all_data=outputs[0]#句子全部输出
        attention_1=torch.index_select(all_data,dim=1,index=index1)
        attention_2=torch.index_select(all_data,dim=1,index=index2)
        extra_attention_output=self.bert_attention.forward(hidden_states=attention_1,
                                                           encoder_hidden_states=attention_2,
                                                           attention_mask=attention_mask)[0]
        extra_attention=torch.mean(extra_attention_output.float(),dim=1)#将得到的attention做平均
        output=torch.cat((pooled_output,extra_attention),dim=1)#将tensor拼接
        '''
        prediction_scores=self.cls(pooled_output)
        prediction_scores=self.softmax(prediction_scores)
        prediction_label=torch.argmax(prediction_scores,dim=1)
        outputs=(prediction_scores,prediction_label,)

        if labels is not None:
            #如果样本不均衡，这里可以将小样本的权重设置大一点
            #loss_fct=CrossEntropyLoss(weight=torch.from_numpy(np.array([1,8])).float().to(device))
            loss_fct=CrossEntropyLoss()#交叉熵损失函数
            loss=loss_fct(prediction_scores.view(-1,2),labels.view(-1))#将softmax概率分布修改维度，并使用交叉熵计算softmax概率分布和真实标签之间的差异
            outputs=(loss,)+outputs
        return outputs



        
