import os
from typing import Any, Optional, Tuple
from collections import Counter
import faiss
import numpy as np
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from data.CWRU import CWRUDataset
from data.JNU import JNUDataset
from data.PU import PUDataset
from data.transforms import *
from data.multi_view_data_injector import MultiViewDataInjector, MultiSimData, DisSimData
from utils import time_signal, classification_report_


class OneDimDataset(Dataset):

    def __init__(self, mode, args):
        self.args = args

        # Data source decision
        if "CWRU" in args.data and "classes" not in args.data:
            # when only choose CWRU, use All CWRU data for pretrain and CWRU 10 classes for test
            data = CWRUDataset(args=self.args)
            train_x, train_y = data.slice_enc_all()
            self.args.data = os.path.join(os.path.dirname(args.data), "CWRU_10_classes")
            _, test_x, valid_x, _, test_y, valid_y = data.slice_enc()
        elif "CWRU" in args.data:
            data = CWRUDataset(args=self.args)
            train_x, test_x, valid_x, train_y, test_y, valid_y = data.slice_enc()
        elif "JNU" in args.data:
            data = JNUDataset(args=self.args)
            train_x, test_x, valid_x, train_y, test_y, valid_y = data.slice_enc()
        elif "PU" in args.data:
            # if args.train_mode == 'finetune':
            #     args.data_mode = 'reallife'
            #     args.num_classes = 15
            data = PUDataset(args=self.args)
            train_x, test_x, valid_x, train_y, test_y, valid_y = data.slice_enc()
        else:
            raise "Your path mask has data name, or no such data!!!"
        
        if args.train_mode in ['time-frequency']:
            self.transform = MultiViewDataInjector([time_signal_transforms(args), time_signal_transforms_to_freq(args)])
        elif args.resume_mode == 'time-frequency':
            self.transform = MultiViewDataInjector([time_signal_transforms(args), time_signal_transforms_to_freq(args)])
        else:
            self.transform = time_signal_transforms(args)
        
        # 用字符选择需要的数据集
        if mode not in ['train', 'test', 'valid']:
            raise 'there is no such dataset!!!'
        signal, label = None, None
        if mode == 'train':
            signal, label = train_x, train_y
        if mode == 'test':
            signal, label = test_x, test_y
        if mode == 'valid':
            signal, label = valid_x, valid_y

        # 改变变量类型以适应模型的输入
        self.signal = signal.astype(np.float32)
        self.label = label.astype(np.int64)
        self._len = len(self.signal)
        
    def log_label_distribution(self, label1, label2, args):
        if args.active_log:
            distribute1 = self.kmeans_label(label1, label2)
            distribute2 = self.kmeans_label(label2, label1)
            with open('results/temp.txt', 'a') as f:
                for count in distribute1:
                    f.writelines(str(count) + "\n")
                f.write('\n')
                f.write('\n')
                for count in distribute2:
                    f.writelines(str(count) + "\n")
                f.write('\n')
                f.write('\n')
    
    def log_init_temp(self, args):
        if args.active_log:
            with open('results/temp.txt', 'w') as f:
                f.writelines("log:\n")

    def kmeans_label(self, label1, label2):
        distribute = []
        for i in set(label1):
            distribute.append(Counter(label2[label1 == i]))
        return distribute

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        lab: Optional[int]
        if self.label is not None:
            if self.args.train_mode == 'pm':
                sig, lab = self.signal[idx], self.label[idx]
            else:
                sig, lab = self.signal[idx], int(self.label[idx])
        else:
            sig, lab = self.signal[idx], -1
        
        if self.args.train_mode == 'self-cnn':
            sig = self_cnn_transforms(self.args, sig, lab)
            sig = torch.Tensor(sig)
        elif self.transform is not None:
            sig = self.transform(sig)
        
        return sig, lab

    def __len__(self):
        return self._len
    
    def change_len(self):
        self._len = len(self.signal)
