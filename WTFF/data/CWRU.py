"""
Created on Sun Sep 26 10:12:08 2021

@author: weiyang
"""

import numpy as np
import scipy.io
import math
import os

from utils import *


class CWRUDataset:

    def __init__(self, args=None):
        '''Initialize class CWRUDataset

        Args:
            file_root (string): path to dataset source file, (i.e. './dataset')

            args (Dictionary): Program Arguments
        '''
        # def __init__(self, file_root="../../dataset/CWRU/1HP", length=2048, number=685, step=174):
        # w/o overlap
        # length=2048, number=58, step=2048
        # w overlap
        # length=2048, number=710, step=165
        # 12k DE DE 的最小信号长度: 121265
        self.args = args
        self.minilength_de_de = 121265
        # self.length = length
        # self.number = number
        # self.step = step
        self.length = args.length
        # self.step = int(args.length * (1 - args.overlap_rate))
        self.number = args.train_samples + args.test_samples + args.valid_samples
        self.signal_noise = False
        self.signal_std = False
        self.signal_nm = False
        self.sample_noise = False
        self.sample_std = False
        self.sample_nm = False
    
    def capture_cwru_DE_12k(self):
        files = {}
        matfiles = []
        
        file_root = os.path.join(os.path.dirname(self.args.data), "CWRU_10_classes")
        for load in self.args.data_mode.split():
            # 将文件夹下的数据读取到列表中
            filenames = os.listdir(os.path.join(file_root, load))
            # 便利文件名列表读入 mat 数据
            for name in filenames:
                # 获取文件路径
                mat_path = os.path.join(file_root, load, name)
                # 读取 mat 文件
                mat_file = scipy.io.loadmat(mat_path)
                matfiles.append(mat_file)

                # 法一： 构造 key 读取文件，不方便，且在特殊情况下难以实现
                # 对文件名进行拆分，保留 . 以前的字符，即去除 .mat
                # data_name = name.split('.')[0]
                # 构造 key 读取 mat 文件
                # key = 'X{}_DE_time'.format(data_name)

                # 法二： 根据字典中包含的特定字符读取文件
                # 取出 mat_file 的keys
                file_keys = mat_file.keys()
                # 遍历 file_keys
                for key in file_keys:
                    # 判断 key 字符是否包含特定字符：
                    if 'DE' in key:
                        files[name] = mat_file[key].ravel()
        return files

    def capture_cwru_10_classes(self):
        files = self.capture_cwru_DE_12k()
        # sort your data, make sure the order is the same and printed it out
        # after sorting, the files_sorted come out as [B007, B014, B021, IR007, IR014, IR021, OR007, OR014, OR021, normal]
        # files_sorted = {}
        # for i in sorted(files):
        #     files_sorted[i] = files[i]
        # print(files_sorted)
        # let the order come print(files.keys())up with
        # [normal, IR007, IR014, IR021, B007, B014, B021, OR007, OR014, OR021]
        signals = []
        files_sorted = {}
        key_order = ["normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        # for i in files.keys() i in key_order:
        #     files_sorted[i] = files[i]
        # for i in files_sorted.keys(): print(i)
        # print(files_sorted.keys())
        return files_sorted
    
    def capture_cwru_9_classes(self):
        files = self.capture_cwru_DE_12k()
        signals = []
        files_sorted = {}
        key_order = ["IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        return files_sorted

    def capture_cwru_4_classes(self):
        files = self.capture_cwru_DE_12k()
        files_sorted = {}
        key_order = ["normal", "IR007", "OR007", "B007"]
        signals = []
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        return files_sorted
    
    def capture_cwru_all(self):
        files = {}

        file_dirs = ["Drive End 12k", "Drive End 48k", "Fan End 12k", "Normal"]
        for file_dir in file_dirs:
            for file in os.listdir(os.path.join(self.args.data, file_dir)):
                mat_file = scipy.io.loadmat(os.path.join(self.args.data, file_dir, file))
                for key in mat_file.keys():
                    if "time" in key:
                        files[key] = mat_file[key]
        return files
    
    def capture_cwru_normal(self):
        files = {}

        if os.path.exists("../../dataset/CWRU/Normal"):
            path = "../../dataset/CWRU/Normal"
        elif os.path.exists("../../autodl-tmp/CWRU/Normal"):
            path = "../../autodl-tmp/CWRU/Normal"
        else:
            assert "no normal dataset file"

        for file in os.listdir(path):
            mat_file = scipy.io.loadmat(os.path.join(path, file))
            for key in mat_file.keys():
                if "DE" in key:
                    files[key] = mat_file[key]
        return files

    # sort your data, make sure the order is the same and printed it out
    # 顺序读取数据的函数
    # 样本个数
    # Length控制单个样本的长度
    # Contact_num 重合数据数量 Contact_ratio 重合率为 contact_num / Length = (Length-step) / Length
    # 注意样本的总长度只有120000
    def slice_enc(self):
        if 'CWRU_10_classes' in self.args.data:
            mat_data = self.capture_cwru_10_classes()
        elif 'CWRU_9_classes' in self.args.data:
            mat_data = self.capture_cwru_9_classes()
        elif 'CWRU_4_classes' in self.args.data:
            mat_data = self.capture_cwru_4_classes()
        else:
            raise 'no such cwru slice mode!!!'
        keys = mat_data.keys()

        label = 0
        train_dataset, test_dataset, valid_dataset = [], [], []
        train_labels, test_labels ,valid_labels = [], [], []
        # 遍历字典，读取数据
        for name in keys:
            class_train_dataset, class_test_dataset, class_valid_dataset = [], [], []
            for mat_single_data in mat_data[name]:
                # 是否进行归一化 (normalization) 以及标准化 (standardization)
                if self.signal_noise:
                    mat_single_data = self.Add_noise(mat_single_data, snr=10)
                if self.signal_std:
                    mat_single_data = self.standardization(mat_single_data)
                if self.signal_nm:
                    mat_single_data = self.normalization(mat_single_data)
                valid_length = (self.args.valid_samples//len(mat_data[name])+1) * self.length
                valid_length = valid_length if valid_length < 25600 else 25600
                valid_data = mat_single_data[:valid_length]
                test_length = math.ceil(len(mat_single_data) * self.args.test_samples / self.number)
                test_length = test_length if test_length > 25600 else 25600
                test_data = mat_single_data[valid_length : valid_length + test_length]
                train_data = mat_single_data[test_length + valid_length :]
                # 根据要获取的样本数量对数据进行切片
                class_train_dataset.append(self.slice_data(train_data, self.args.train_samples//len(mat_data[name])+1))
                class_test_dataset.append(self.slice_data(test_data, self.args.test_samples//len(mat_data[name])+1))
                class_valid_dataset.append(self.slice_data(valid_data, self.args.valid_samples//len(mat_data[name])+1))
            train_dataset.append(np.asarray(class_train_dataset).reshape((-1, 1, 2048))[:self.args.train_samples]),
            test_dataset.append(np.asarray(class_test_dataset).reshape((-1, 1, 2048))[:self.args.test_samples]),
            valid_dataset.append(np.asarray(class_valid_dataset).reshape((-1, 1, 2048))[:self.args.valid_samples]),
            # 根据每种样本的数量生成标签
            train_labels += [label] * self.args.train_samples
            test_labels += [label] * self.args.test_samples
            valid_labels += [label] * self.args.valid_samples
            label += 1
            
        return np.asarray(train_dataset).reshape((-1, 1, 2048)),\
            np.asarray(test_dataset).reshape((-1, 1, 2048)),\
            np.asarray(valid_dataset).reshape((-1, 1, 2048)),\
            np.asarray(train_labels), np.asarray(test_labels),\
            np.asarray(valid_labels)
    
    def slice_enc_all(self):
        mat_data = self.capture_cwru_all()

        label = 0
        train_dataset = []
        train_labels = []
        for name in mat_data.keys():
            train_data = mat_data[name]
            train_dataset.append(self.slice_data(train_data, self.args.train_samples))
            train_labels += [label] * self.args.train_samples
            label += 1
        
        return np.asarray(train_dataset).reshape((-1, 1, 2048)), np.asarray(train_labels)

    def slice_data(self, data, number):
        datasets = []
        # 根据信号长度改变步长
        self.step = int((data.shape[0] - self.length) / (self.number - 1))
        for x in range(number):
            datas = data[self.step * x:self.step * x + self.length]
            if self.sample_noise:
                datas = self.Add_noise(datas, snr=10)
            if self.sample_std:
                datas = self.standardization(datas)
            if self.sample_nm:
                datas = self.normalization(datas)
            datasets.append(datas)
        return np.asarray(datasets)

    # 数据归一化
    def normalization(self, data):
        # 判断 data 是否为 list，不是 list 将会把列表转换为数组
        if isinstance(data, list):
            data = np.asarray(data)
        # _range = np.max(data) - np.min(data)
        # return (data - np.min(data)) / _range
        # 需要归一化后的范围为 [-1, 1]
        _range = np.max(abs(data))
        return data / _range
        # 可以用于对比 numpy 的效率
        # return (max(data)-data)/(max(data)-min(data))

    # 标准化
    def standardization(self, data):
        if isinstance(data, list):
            data = np.asarray(data)
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    # 制作白噪声
    # x为输入信号，snr为信噪比
    def wgn(self, x, snr):
        """ create white noise 

        Aigs:
            x (Array, List): signal data
            snr (int): signal-to-noise ratio
        
        Returns:
            noise (Array)
        """
        P_signal = np.sum(abs(x) ** 2) / len(x)
        P_noise = P_signal / 10 ** (snr / 10.0)
        return np.random.randn(len(x)) * np.sqrt(P_noise)

    # 添加噪声信号到原始信号中
    # x为输入信号，d为噪声原始信号，snr为信噪比
    def Add_noise(self, x, snr):
        """ add noise to signal x 
        
        Aigs:
            x (array, list): signal data
            snr (int): signal-to-noise ratio
        
        Returns:
            noise_signal (Array): signal with noise
        """
        if isinstance(x, list):
            x = np.asarray(x)
        d = self.wgn(x, snr)
        P_signal = np.sum(abs(x) ** 2)
        P_d = np.sum(abs(d) ** 2)
        P_noise = P_signal / 10 ** (snr / 10)
        noise = np.sqrt(P_noise / P_d) * d
        noise_signal = x + noise
        return noise_signal
