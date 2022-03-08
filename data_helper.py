#actor:NJUST_Tang Bin
#@file: data_helper
#@time: 2022/3/4 9:42
#-*-coding:UTF-8-*-
import os
import json
#在保存的模型中根据验证集和测试集平均效果挑选出最好的模型
def find_best_model(save_models_path):
    path_list=os.listdir(save_models_path)
    models={}
    for p in path_list:
        dev_path=os.path.join(save_models_path,p,"json_data.json")
        with open(dev_path,"r",encoding="utf-8") as fh:
            json_data=json.load(fh)
            dev_acc=json_data["acc"]
        test_path=os.path.join(save_models_path,p,"test_json_data.json")
        with open(test_path,"r",encoding="utf-8") as fh:
            json_data=json.load(fh)
            test_acc=json_data["acc"]
        avg_acc=(float(dev_acc)+float(test_acc))/2
        models[p]=avg_acc
    best_model=sorted(models.items(),key=lambda x:x[1],reverse=True)[0]
    return best_model
