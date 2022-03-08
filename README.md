# bert_torch_text_cls
使用中文bert预训练模型对文本进行分类
1、按照few-shot文件夹下的数据格式修改自己的数据。
2、data_helper.py为数据整理文件
3、model.py为模型结构文件，在此处搭建通用文本分类模型
4、data_set.py为数据输入文件，将data_helper.py制作出的数据整理成模型输入要求的格式。
5、train.py模型训练文件。
6、predict.py模型测试文件。
