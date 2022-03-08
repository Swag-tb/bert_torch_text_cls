#actor:NJUST_Tang Bin
#@file: predict
#@time: 2022/1/10 11:25
#-*-coding:UTF-8-*-
import torch
import os
import json
import random
import numpy as np
import argparse
import logging
from transformers import BertTokenizer,BertConfig
from model import BertTorchClassfication
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

def set_args():
    parser=argparse.ArgumentParser()#创建一个解析器
    parser.add_argument('--device',default='2',type=str,help='设置训练或测试时使用的显卡')
    parser.add_argument('--model_path', default='D:\硕士\培训\预训练语言模型\中文/uer-bert-chinese-large', type=str, help='保存的模型路径')
    parser.add_argument('--vocab_path', default='D:\硕士\培训\预训练语言模型\中文/uer-bert-chinese-large/vocab.txt', type=str, help='预训练模型字典数据')
    parser.add_argument('--max_len', default=512, type=int, help='输入模型的文本的最大长度')
    parser.add_argument('--seed', default=2022, type=int, help='随机种子')
    return parser.parse_args()#调用parse_args方法解析参数

def convert_feature(tokenizer,sentence):
    '''
    数据处理函数
    :param sample: 输入的每个文本
    :return:
    '''
    sentence_tokens=[i for i in sentence]
    tokens=['[CLS]']+sentence_tokens+['[SEP]']
    input_ids=tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids=[0]*len(input_ids)
    position_ids=[s for s in range(len(tokens))]
    attention_mask=[1]*len(tokens)
    assert len(input_ids)==len(token_type_ids)
    assert len(input_ids)==len(attention_mask)
    assert len(input_ids)<=512
    return input_ids,token_type_ids,position_ids,attention_mask

def predict_one_sample(sentence,model,tokenizer,device):
    input_ids, token_type_ids, position_ids, attention_mask=convert_feature(tokenizer,sentence)
    with torch.no_grad():
        input_tensors=torch.tensor([input_ids]).to(device)
        token_type_tensors = torch.tensor([token_type_ids]).to(device)
        position_tensors = torch.tensor([position_ids]).to(device)
        attention_mask_tensors = torch.tensor([attention_mask]).to(device)
        scores,prediction=model.forward(input_ids=input_tensors,
                                        token_type_ids=token_type_tensors)
        scores=scores.cpu().numpy().tolist()
        prediction=prediction.cpu().numpy().tolist()[0]
        return prediction,scores

def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    model = BertTorchClassfication.from_pretrained(args.model_path)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    model.to(device)
    model.eval()
    # 单条测试
    # sentence='11111111asdasdq'
    # result,score=predict_one_sample(sentence,model,tokenizer,device)
    # print(result)
    # exit()
    #多条测试
    labels=[]
    preds=[]
    with open('few_shot_data/test.txt','r',encoding='utf-8') as input_data:
        for line in input_data.readlines():
            sentence=line.strip('\n').split('\t')[1]
            if len(sentence)>=500:
                continue
            label=line.strip('\n').split('\t')[0]
            result,score=predict_one_sample(sentence,model,tokenizer,device)
            labels.append(str(label))
            preds.append(str(result))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    f=2*precision*recall/(precision+recall)
    p=np.array(preds)
    r=np.array(labels)
    acc=np.mean((p==r))
    print('p:{},R:{},F:{},ACC:{}'.format(precision,recall,f,acc))

if __name__=="__main__":
    main()

