import numpy as np
from utils import grid_search, random_search, change_name
from sklearn.model_selection import train_test_split
import re
import os


def folder_run(resume_path):
    param = {
        'select_data': '0',
        'ncentroids': '0',
    }

    for dir in os.listdir(resume_path):
        file = os.path.join(resume_path, dir, 'training.log')
        with open(file) as f:
            txt = f.read()

        # 检查是否是一个有效的log
        reg = f"INFO:root:\[eval\] loss: (.*?)," # find train_mode
        reg_result = re.findall(reg, txt)
        reg = f"INFO:root:train_mode: (.*?)\n" # find train_mode
        train_mode = re.findall(reg, txt)
        if len(reg_result) == 0 or train_mode != ["ph_simclr"]: continue

        for key in param.keys():
            reg = f"INFO:root:{key}: (.*?)\n" # find train_mode
            reg_result = re.findall(reg, txt)
            parameter[key.replace('_', '-')] = reg_result
            print('*' * 50)
            print(key,reg_result)
            print('*' * 50 + '\n')

        grid_search('main.py', parameter)


def folder_run_with_conditions(resume_path):
    conditions = {
        "train_mode": "ph_simclr",
        "backbone": "resnet18_1d", # rwkdcae, resnet18_1d
        "select_data": "0.5",
        "ncentroids": "20",
        "select_positive": "cluster_postive", # cluster_postive, combine_inst_dis, only_inst_dis
        "data_mode": "reallife",  # "0HP 1HP 2HP 3HP", "artificial", "reallife"
    }

    for dir in os.listdir(resume_path):
        file = os.path.join(resume_path, dir, 'training.log')
        with open(file) as f:
            txt = f.read()
        
        condition_init = True
        for key in conditions.keys():
            reg = f"INFO:root:{key}: (.*?)\n" # find train_mode
            reg_result = re.findall(reg, txt)
            if len(reg_result) == 0:
                condition_init = False
                break
            else:
                condition_init *= reg_result[0] == conditions[key]

        if condition_init and os.path.exists(os.path.join(resume_path, dir, 'checkpoint_best.pt')):
            print(dir)
            parameter["train-mode"] = ['finetune']
            parameter["resume-mode"] = [conditions["train_mode"]]
            parameter["data-mode"] = [conditions["data_mode"]]
            parameter["backbone"] = [conditions["backbone"]]
            parameter["ncentroids"] = [conditions["ncentroids"]]
            parameter["select-data"] = [conditions["select_data"]]
            if conditions["data_mode"] == "0HP 1HP 2HP 3HP":
                parameter["data-mode"] = ["'0HP 1HP 2HP 3HP'"]
            
            if "dataaug" in parameter.keys(): del parameter["dataaug"]
            if conditions["train_mode"] == "ph_simclr":
                parameter["resume-mode"] = ["simclr"]
                parameter["select-positive"] =  [conditions["select_positive"]]
            else:
                parameter["select-positive"] =  ["only_inst_dis"]
            # parameter["valid-samples"] = [10, 5, 3, 2, 1]
            parameter["valid-samples"] = [10]
            parameter["resume"] = [os.path.join(resume_path, dir, 'checkpoint_best.pt')]

            grid_search('main.py', parameter)
            # break
            # random_search('main.py', parameter, 1)


if __name__ == '__main__':
    parameter = {
        # 'momentum': (0.85, 0.95),
        # 'snr': np.arange(-4, 12, 2).tolist(),
        # 'lr': np.logspace(-6, -1, 6).tolist(),
        # 'train-samples': ["100 -o 0.43", "300 -o 0.81", "660 -o 0.92", "1000 -o -0.943", "2000 -o -0.972"],
        # 'data': ["'../../dataset/JNU bearing dataset'"],
        # 'data-mode': ["load1"], # "load1", "load2", "load3"
        # 'num-classes': [4],
        # 'data': ["'../../dataset/CWRU_10_classes'"], # CWRU_10_classes, CWRU_4_classes, CWRU
        # 'data-mode': ["'0HP 1HP 2HP 3HP'"], # "'0HP 1HP 2HP 3HP'"
        # 'num-classes': [10],
        'data': ["'../dataset/PU bearing dataset'"],
        'data-mode': ["df"], # "artificial", "reallife" , "df"
        'finetuneset': ["df"], # "reallife" , "reallife_subset", "valid_subset1", "df"
        'num-classes': [6],
        # 'dataaug': ['\t'],
        # 'normalize-type': ["mean-std"],
        'train-samples': [1000],
        # 'pretrainset': ['valid'],
        # 'finetune-testset': ['train'],
        # 'valid-samples': [1,2,3,5,10],
        'test-samples': [1000],
        'batch-size': [128],
        # 'backbone': ["mlp", "lenet", "cnn1d", "wdcnn", "resnet18_1d", "resnet50_1d", "tfcnn"],
        'backbone': ["tfcnn"],
        'mlp-hidden-size': [512],
        'projection-size': [2048],
        'max-epochs': [100],
        # 'pretrain-lr-scheduler': ["\t"],
        'finetune-epochs': [100],
        'train-mode': ["time-frequency"], # finetune, time-frequency, evaluate, analysis
        # 'temperature': [0.7],
        'resume-mode': ["time-frequency"],
        # 'finetune-mode': ["eval"],
        # 'resume': ['../runs/2024-04/04/15-53-14/checkpoint_best.pt'], # 21-09-07
        'lr': [1e-3],
        # 'weight-decay': [0],
        # 'base-lr': [0.1], 3e-3, 1e-3, 1e-4
        'ftlr': [0.01],
        'ftwd': [1e-5],
        'active-log': ['\t'],
        # 'gpu-index': [1],
        # 'freeze': ['\t'],
        'remark': ['"tfcnn v6"'],
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    for i in range(2):
        grid_search('main.py', parameter)
