#actor:NJUST_Tang Bin
#@file: train
#@time: 2021/12/29 16:09
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
from data_set import TextClassfication,collate_func
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm,trange

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger=logging.getLogger(__name__)

def train(model,device,tokenizer,args):
    '''
    调整模型
    :param model:
    :param device:
    :param tokenizer:
    :param args:
    :return:
    '''
    tb_write=SummaryWriter()
    if args.gradient_accumulation_steps<1:
        raise ValueError('梯度积累参数无效，必须大于等于1')
    train_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
    train_data=TextClassfication(tokenizer,args.max_len,args.data_dir,"train_text_classification",path_file=args.train_file_path)
    train_sampler=RandomSampler(train_data)
    train_data_loader=DataLoader(train_data,
                                 sampler=train_sampler,
                                 batch_size=train_batch_size,
                                 collate_fn=collate_func)
    total_steps=int(len(train_data_loader)*args.num_train_epochs/args.gradient_accumulation_steps)

    dev_data=TextClassfication(tokenizer,args.max_len,args.data_dir,"dev_text_classfication",path_file=args.dev_file_path)
    test_data = TextClassfication(tokenizer, args.max_len, args.data_dir, "test_text_classfication",path_file=args.test_file_path)
    logging.info("总训练步数为：{}".format(total_steps))
    model.to(device)
    #获取模型所有参数，选择不想权重衰减的参数
    param_optimizer=list(model.named_parameters())
    no_decay=["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_grouped_parameters=[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight':0.0}
    ]
    #设置优化器
    optimizer=AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    # unfreeze_layers = ['cls.']
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             break
    # # 验证一下
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())
    # 过滤掉requires_grad = False的参数
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    schedular=get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=int(args.warmup_proportion*total_steps),
                                              num_training_steps=total_steps)

    #清空cuda缓存
    torch.cuda.empty_cache()
    model.train()
    tr_loss,logging_loss=0.0,0.0
    global_step=0
    for iepoch in trange(0,int(args.num_train_epochs),desc="Epoch",disable=False):
        iter_bar=tqdm(train_data_loader,desc="Iter (loss=X.XXX)",disable=False)
        for step,batch in enumerate(iter_bar):
            input_ids=batch["input_ids"].to(device)
            token_type_ids=batch["token_type_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            position_ids=batch["position_ids"].to(device)
            labels=batch["labels"].to(device)
            outputs=model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  labels=labels)
            loss=outputs[0]
            tr_loss+=loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)"%loss.item())
            #判断是否进行梯度积累，如果进行，则将损失值除以累积步数，每隔多少步更新一次参数
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)
            #如果步数整除累计步数，进行参数优化
            if (step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                schedular.step()
                optimizer.zero_grad()
                global_step+=1
                if args.logging_steps>0 and global_step%args.logging_steps==0:
                    tb_write.add_scalar("lr",schedular.get_lr()[0],global_step)
                    tb_write.add_scalar("train_loss",(tr_loss-logging_loss)/(args.logging_steps*args.gradient_accumulation_steps),global_step)
                    logging_loss=tr_loss
                #如果步数整除save_model_steps，保存训练好的模型
                if args.save_model_steps>0 and global_step%args.save_model_steps==0:
                    eval_acc,json_data=evaluate(model,device,dev_data,args)
                    model.train()
                    logger.info("dev_acc:{}".format(eval_acc))
                    tb_write.add_scalar("dev_acc",eval_acc,global_step)
                    output_dir=os.path.join(args.output_dir,"checkpoint-{}".format(global_step))
                    model_to_save=model.module if hasattr(model,"module") else model
                    model_to_save.save_pretrained(output_dir)
                    json_output_dir=os.path.join(output_dir,"json_data.json")
                    fin=open(json_output_dir,'w',encoding='utf-8')
                    fin.write(json.dumps(json_data,ensure_ascii=False,indent=4))
                    fin.close()
                    test_acc,test_json_data=evaluate(model,device,test_data,args)
                    model.train()
                    logger.info("test_acc:{}".format(test_acc))
                    tb_write.add_scalar("test_acc",test_acc,global_step)
                    json_output_dir=os.path.join(output_dir,"test_json_data.json")
                    fin=open(json_output_dir,"w",encoding='utf-8')
                    fin.write(json.dumps(test_json_data,ensure_ascii=False,indent=4))
                    fin.close()
        torch.cuda.empty_cache()
    eval_acc,json_data=evaluate(model,device,dev_data,args)
    logger.info("dev_acc:{}".format(eval_acc))
    tb_write.add_scalar("dev_acc",eval_acc,global_step)
    output_dir=os.path.join(args.output_dir,"checkpoint-{}".format(global_step))
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    json_output_dir = os.path.join(output_dir, "json_data.json")
    fin = open(json_output_dir, 'w', encoding='utf-8')
    fin.write(json.dumps(json_data, ensure_ascii=False, indent=4))
    fin.close()
    test_acc, test_json_data = evaluate(model, device, test_data, args)
    model.train()
    logger.info("test_acc:{}".format(test_acc))
    tb_write.add_scalar("test_acc", test_acc, global_step)
    json_output_dir = os.path.join(output_dir, "test_json_data.json")
    fin = open(json_output_dir, "w", encoding='utf-8')
    fin.write(json.dumps(test_json_data, ensure_ascii=False, indent=4))
    fin.close()

def evaluate(model,device,dev_data,args):
    '''
    对验证集数据进行模型测试
    :param model:
    :param device:
    :param dev_data:
    :param args:
    :return:
    '''
    test_sampler=SequentialSampler(dev_data)
    test_data_loader=DataLoader(dev_data,sampler=test_sampler,batch_size=args.test_batch_size,collate_fn=collate_func)
    iter_bar=tqdm(test_data_loader,desc="iter",disable=False)
    y_true=[]
    y_predict=[]
    y_scores=[]
    samples=[]
    for step,batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            labels=batch["labels"]
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            scores,prediction=model.forward(input_ids=input_ids,
                                  token_type_ids=token_type_ids)
            y_true.extend(labels.numpy().tolist())
            y_predict.extend(prediction.cpu().numpy().tolist())
            y_scores.extend(scores.cpu().numpy().tolist())
            samples.extend(batch["samples"])
    json_data={"data":[],"acc":None}
    for label,pre,score,sample in zip(y_true,y_predict,y_scores,samples):
        sample["label"]=label
        sample["pre"]=pre
        sample["scores"]=score
        json_data["data"].append(sample)
    y_true=np.array(y_true)
    y_predict=np.array(y_predict)
    eval_acc=np.mean((y_true==y_predict))
    json_data["acc"]=str(eval_acc)
    return eval_acc,json_data

def set_args():
    parser=argparse.ArgumentParser()#创建一个解析器
    parser.add_argument('--device',default='-1',type=str,help='设置训练或测试时使用的显卡')
    parser.add_argument('--train_file_path',default='data/train.txt',type=str,help='训练数据')
    parser.add_argument('--dev_file_path', default='data/dev.txt', type=str, help='验证数据')
    parser.add_argument('--test_file_path', default='data/test.txt', type=str, help='测试数据')
    parser.add_argument('--vocab_path', default='pre_train_model/sci-uncased/vocab.txt', type=str, help='预训练模型字典数据')
    parser.add_argument('--pretrained_model_path', default='pre_train_model/bert_wwm_ext_chinese/', type=str, help='预训练模型路径')
    parser.add_argument('--data_dir', default='cached/', type=str, help='缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=128, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=128, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup概率，即训练总补偿的百分之多少，进行warmup')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--save_model_steps', default=300, type=int, help='保存训练模型步数')
    parser.add_argument('--logging_steps', default=300, type=int, help='保存训练日志的步数')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='model_output/', type=str, help='模型输出路径')
    parser.add_argument('--seed', default=2022, type=int, help='随机种子')
    parser.add_argument('--max_len', default=512, type=int, help='输入模型的文本的最大长度')
    return parser.parse_args()#调用parse_args方法解析参数

def main():
    args=set_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device=torch.device("cuda" if torch.cuda.is_available() and int(args.device)>=0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    #加载模型的config，重新定义token_type个数
    # model_path='pre_train_model/sci-uncased/pytorch_model.bin'
    # config_path='pre_train_model/sci-uncased/config.json'
    # state_dict=torch.load(model_path,map_location='cpu')
    # a=state_dict['bert.embeddings.token_type_embedding.weight'].tolist()
    # b = state_dict['bert.embeddings.token_type_embedding.weight'].tolist()
    # c=a+b
    # state_dict['bert.embeddings.token_type_embedding.weight']=torch.tensor(c)
    # model_config=BertConfig.from_json_file(config_path)
    # model=BertTorchClassfication.from_pretrained(None,config=config_path,state_dict=state_dict)
    model=BertTorchClassfication.from_pretrained(args.pretrained_model_path)
    #实例化tokenizer
    tokenizer=BertTokenizer.from_pretrained(args.vocab_path,do_lower_case=True)
    for i in range(1,100):
        tokenizer.add_tokens("[uncased{}]".format(i),special_tokens=True)
    #创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #开始训练
    train(model,device,tokenizer,args)

if __name__=="__main__":
    main()
