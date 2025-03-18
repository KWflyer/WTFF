import os
import math
import pandas as pd

from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from utils import *
from data.sequence_aug import *

#Three working conditions
WC1 = ["n600_3_2.csv","ib600_2.csv","ob600_2.csv","tb600_2.csv"]
WC2 = ["n800_3_2.csv","ib800_2.csv","ob800_2.csv","tb800_2.csv"]
WC3 = ["n1000_3_2.csv","ib1000_2.csv","ob1000_2.csv","tb1000_2.csv"]

label1 = [i for i in range(0,4)]
label2 = [i for i in range(4,8)]
label3 = [i for i in range(8,12)]


class JNUDataset:
    def __init__(self, args) -> None:
        self.args = args
        self.file_root = args.data
        # self.length = length
        # self.number = number
        # self.step = step
        self.length = args.length
        # self.step = int(args.length * (1 - args.overlap_rate))
        self.number = args.train_samples + args.test_samples + \
            args.valid_samples
        self.signal_noise = False
        self.signal_std = False
        self.signal_nm = False
        self.mat_data = self.capture_files()

    #generate Training Dataset and Testing Dataset
    def capture_files(self):
        '''
        This function is used to generate the final training set and test set.
        root:The location of the data set
        '''

        data = {}
        if self.args.data_mode == "load1":
            MatData = WC1
        elif self.args.data_mode == "load2":
            MatData = WC2
        elif self.args.data_mode == "load3":
            MatData = WC3
        elif self.args.data_mode == "all":
            MatData = WC1 + WC2[1:] + WC3[1:]
        elif self.args.data_mode == "fault":
            MatData = WC1[1:] + WC2[1:] + WC3[1:]
        else:
            raise "no such data mode!!!"
        
        start = time.time()
        for k in tqdm(range(len(MatData))):
            path=os.path.join(self.file_root,MatData[k])
            fl = np.loadtxt(path)
            data[MatData[k]] = [fl]
        end = time.time() - start
        print("Read data time:%.2f"%end + "s")
        return data

    def capture_normal_files(self):
        data = {}
        MatData = [WC1[0],WC2[0],WC3[0]]
        for i in range(len(MatData)):
            path=os.path.join(self.file_root,MatData[i])
            fl = np.loadtxt(path)
            data[MatData[i]] = fl
        return data
    
    # sort your data, make sure the order is the same and printed it out
    # 顺序读取数据的函数
    # 样本个数
    # Length控制单个样本的长度
    # Contact_num 重合数据数量 Contact_ratio 重合率为 contact_num / Length = (Length-step) / Length
    # 注意样本的总长度只有120000
    def slice_enc(self):
        # 获取保存读取数据字典的 keys
        keys = self.mat_data.keys()

        label = 0
        train_dataset, test_dataset, valid_dataset = [], [], []
        train_labels, test_labels ,valid_labels = [], [], []
        # 遍历字典，读取数据
        for name in keys:
            class_train_dataset, class_test_dataset, class_valid_dataset = [], [], []
            for mat_single_data in self.mat_data[name]:
                valid_length = (self.args.valid_samples//len(self.mat_data[name])+1) * self.length
                valid_data = mat_single_data[:valid_length]
                test_length = math.ceil(len(mat_single_data) * self.args.test_samples / self.number)
                test_length = test_length if test_length > 25600 else 25600
                test_data = mat_single_data[valid_length : valid_length + test_length]
                train_data = mat_single_data[test_length + valid_length :]
                # 根据要获取的样本数量对数据进行切片
                class_train_dataset.append(self.slice_data(train_data, self.args.train_samples//len(self.mat_data[name])+1))
                class_test_dataset.append(self.slice_data(test_data, self.args.test_samples//len(self.mat_data[name])+1))
                class_valid_dataset.append(self.slice_data(valid_data, self.args.valid_samples//len(self.mat_data[name])+1))
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

    def slice_data(self, data, number):
        datasets = []
        # 根据信号长度改变步长
        self.step = int((data.shape[0] - self.length) / (self.number - 1))
        for x in range(number):
            datas = data[self.step * x:self.step * x + self.length]
            datasets.append(datas)
        return datasets

