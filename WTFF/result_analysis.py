import os
import re
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import change_name


def train_result(file_tree, writer):
    file = os.path.join(file_tree, 'training.log')
    assert os.path.exists(file), 'File not exists'
    # 读取文件
    with open(file, 'r') as f:
        # 读取文件内容
        txt = f.read()
    # 使用正则表达式取出内容
    epochs = re.findall('Epoch:(.*?)\t', txt)
    loss = re.findall('Loss:(.*?)\n', txt)
    setting = re.findall(':(.*?), ', txt)
    # 使用pandas将内容保存到不同的xlsx的不同文件夹中
    df = pd.DataFrame({'epochs': epochs, 'loss': loss})
    df2 = pd.DataFrame({'setting': setting})
    df.to_excel(writer, 'result', index=False)
    df2.to_excel(writer, 'setting', index=False)


def test_result(file_tree, writer):
    # 查看该目录下的文件
    folders = os.listdir(file_tree)
    for i, folder in enumerate(folders):
        print(folder)
        # 读取文件夹，去除文件
        if '-' in folder:
            # 生成 testing.log 路径
            file = os.path.join(file_tree, folder, 'testing.log')
            # 判断文件是否存在，存在便进行读取
            if os.path.exists(file):
                with open(file, 'r') as f:
                    # 读取文件内容
                    txt = f.read()
                # 使用正则表达式取出内容
                epochs = re.findall('Epoch:(.*?)\t', txt)
                trian_loss = re.findall('Train Loss:(.*?)\t', txt)
                train_acc = re.findall('Train accuracy:(.*?)\n', txt)
                test_loss = re.findall('Test Loss:(.*?)\t', txt)
                test_acc = re.findall('Test accuracy:(.*?)\n', txt)
                setting = re.findall('root:(.*?)\.\n', txt)
                # 使用pandas将内容保存到不同的xlsx的不同文件夹中
                df = pd.DataFrame({'epochs': epochs, 'trian_loss': trian_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc})
                df2 = pd.DataFrame({'setting': setting})
                df.to_excel(writer, f'test result{i}', index=False)
                df2.to_excel(writer, f'test setting{i}', index=False)
            else:
                print('No such file')


def result_to_np(root):
    files = os.listdir(root)
    for file in files:
        path = os.path.join(root, file, 'testing.log')
        if os.path.exists(path):
            with open(path) as f:
                txt = f.read()
            epochs = re.findall('Epoch:(.*?)\t', txt)
            trian_loss = np.array(re.findall('Train Loss:(.*?)\t', txt))
            train_acc = np.array(re.findall('Train accuracy:(.*?)\t', txt)).astype(float)
            test_loss = np.array(re.findall('Test Loss:(.*?)\t', txt))
            test_acc = np.array(re.findall('Test accuracy:(.*?)\n', txt))
            setting = re.findall('INFO:root:(.*?)\n', txt)
            print(np.max(train_acc))
            plt.ylim((0, 100))
            plt.plot(train_acc)
            plt.text(0.5, 0.5, f'max: {np.max(train_acc)}')
            # plt.legend(f'{np.max(train_acc)}')
            plt.show()


def results_to_execl_1day(root='../results/16'):
    expr = os.listdir(root)

    df = pd.DataFrame()
    for e in expr:
        if os.path.exists(os.path.join(root, e, 'training.log')):
            with open(os.path.join(root, e, 'training.log')) as f:
                train_res = f.read()
            backbone = re.findall('bone:(.*?),', train_res)
            mlp_hidden_size = re.findall('mlp_hidden_size:(.*?),', train_res)
            projection_size = re.findall('projection_size:(.*?)\.', train_res)
            train_bs = re.findall('batch_size:(.*?),', train_res)
            train_lr = re.findall('lr:(.*?),', train_res)
            train_wd = re.findall('wd:(.*?),', train_res)
            train_m = re.findall('m:(.*?),', train_res)
            train_temperature = re.findall('temperature:(.*?),', train_res)
            train_ep = re.findall('max-epochs: (.*?) workers', train_res)
            workers = re.findall('workers: (.*?),', train_res)
            gpu_index = re.findall('gpu index: (.*?)\.', train_res)
            train_load = re.findall('train load: (.*?),', train_res)
            length = re.findall('length: (.*?),', train_res)
            overlap_rate = re.findall('overlap rate: (.*?),', train_res)
            number = re.findall('for every load: (.*?),', train_res)
            snr = re.findall('snr: (.*?),', train_res)
            dropout = re.findall('dropout: (0\.\d)', train_res)
            save_path = re.findall('saved at (.*?), model', train_res)
            optimizer = re.findall('optimizer: <class (.*?)>', train_res)
            train_rt = re.findall('runtime(.*?s)\.', train_res)
            if snr == []:
                snr = np.NaN
            if dropout == []:
                dropout = np.NaN
            train_loss = np.array(re.findall('Loss: (.*?)\.\n', train_res)).astype(float)
            # print(train_loss)

            train_min_l = np.min(train_loss)
            train_last_l = train_loss[-1]

            test = os.listdir(os.path.join(root, e))
            for t in test:
                if '-' in t and os.path.exists(os.path.join(root, e, t, 'testing.log')):
                    with open(os.path.join(root, e, t, 'testing.log')) as f:
                        test_res = f.read()
                    test_start = re.findall('at (.*?) training', test_res)
                    test_epochs = re.findall('for (.*?) epochs', test_res)
                    test_model = re.findall('from (.*?)\n', test_res)
                    test_bs = re.findall('batch_size (.*?),', test_res)
                    test_lr = re.findall('lr (.*?)\n', test_res)
                    test_tr_load = re.findall('train load: (.*?),', test_res)
                    test_te_load = re.findall('test load: (.*?),', test_res)
                    test_length = re.findall('length: (.*?),', test_res)
                    test_overlap_rate = re.findall('overlap rate: (.*?),', test_res)
                    test_number = re.findall('every load: (.*?),', test_res)
                    test_device = re.findall('device: (.*?)\.', test_res)
                    test_load_time = re.findall('data time: (.*?s)\.', test_res)
                    test_run_time = re.findall('runtime (.*?s)\.', test_res)
                    test_tr_loss = np.array(re.findall('Train Loss:(.*?)\t', test_res)).astype(float)
                    test_tr_acc = np.array(re.findall('Train accuracy:(.*?)\t', test_res)).astype(float)
                    test_te_loss = np.array(re.findall('Test Loss:(.*?)\t', test_res)).astype(float)
                    test_te_acc = np.array(re.findall('Test accuracy:(.*?)\n', test_res)).astype(float)

                    # print(test_tr_loss.shape)
                    if test_tr_loss.shape == (0,):
                        test_tr_loss = np.zeros((1,))
                    if test_tr_acc.shape == (0,):
                        test_tr_acc = np.zeros((1,))
                    if test_te_loss.shape == (0,):
                        test_te_loss = np.zeros((1,))
                    if test_te_acc.shape == (0,):
                        test_te_acc = np.zeros((1,))
                    # print(test_tr_loss)

                    if test_load_time == []:
                        test_load_time = np.NaN
                    if test_run_time == []:
                        test_run_time = np.NaN

                    test_tr_loss_min = np.min(test_tr_loss)
                    test_tr_loss_last = test_tr_loss[-1]
                    test_tr_acc_max = np.max(test_tr_acc)
                    test_tr_acc_last = test_tr_acc[-1]
                    test_te_loss_min = np.min(test_te_loss)
                    test_te_loss_last = test_te_loss[-1]
                    test_te_acc_max = np.max(test_te_acc)
                    test_te_acc_last = test_te_acc[-1]

                    df2 = pd.DataFrame({
                        'backbone': backbone,
                        'mlp_hidden_size': mlp_hidden_size,
                        'projection_size': projection_size,
                        'workers': workers,
                        'gpu_index': gpu_index,
                        'train_load': train_load,
                        'length': length,
                        'overlap_rate': overlap_rate,
                        'number': number,
                        'snr': snr,
                        'dropout': dropout,
                        'save_path': save_path,
                        'optimizer': optimizer,
                        'test_start': test_start,
                        'train_bs': train_bs,
                        'train_lr': train_lr,
                        'train_wd': train_wd,
                        'train_m': train_m,
                        'train_temperature': train_temperature,
                        'train_ep': train_ep,
                        'train_rt': train_rt,
                        'train_min_l': train_min_l,
                        'train_last_l': train_last_l,
                        'test_epochs': test_epochs,
                        'test_model': test_model,
                        'test_bs': test_bs,
                        'test_lr': test_lr,
                        'test_tr_load': test_tr_load,
                        'test_te_load': test_te_load,
                        'test_length': test_length,
                        'test_overlap_rate': test_overlap_rate,
                        'test_number': test_number,
                        'test_device': test_device,
                        'test_load_time': test_load_time,
                        'test_run_time': test_run_time,
                        'test_tr_loss_min': test_tr_loss_min,
                        'test_tr_loss_last': test_tr_loss_last,
                        'test_tr_acc_max': test_tr_acc_max,
                        'test_tr_acc_last': test_tr_acc_last,
                        'test_te_loss_min': test_te_loss_min,
                        'test_te_loss_last': test_te_loss_last,
                        'test_te_acc_max': test_te_acc_max,
                        'test_te_acc_last': test_te_acc_last
                        })

                    df = pd.concat([df, df2])

    df.to_excel(change_name(os.path.join(root, 'result.xlsx')), sheet_name='all_result', index=True)
    print(df.head())


def find_resumefile(file, dir):
    with open(file) as f:
        txt = f.read()
    reg = "INFO:root:resume: (.*?)\n" # find train_mode
    return re.findall(reg, txt)


def find_max_acc(file, dir):
    with open(file) as f:
        txt = f.read()
    reg = "INFO:root:train_mode: (.*?)\n" # find train_mode
    train_mode = re.findall(reg, txt)
    reg = "INFO:root:resume_mode: (.*?)\n" # find train_mode
    resume_mode = re.findall(reg, txt)
    reg = "INFO:root:data_mode: (.*?)\n"
    data_mode = re.findall(reg, txt)
    reg = "INFO:root:train_samples: (.*?)\n"
    train_samples = re.findall(reg, txt)
    reg = "INFO:root:backbone: (.*?)\n"
    backbone = re.findall(reg, txt)
    reg = "INFO:root:batch_size: (.*?)\n"
    batch_size = re.findall(reg, txt)
    reg = "INFO:root:ftlr: (.*?)\n"
    ftlr = re.findall(reg, txt)
    reg = "INFO:root:valid_samples: (.*?)\n"
    valid_samples = re.findall(reg, txt)
    reg = "INFO:root:resume: (.*?)\n"
    resume = re.findall(reg, txt)
    reg = "INFO:root:freeze: (.*?)\n"
    freeze = re.findall(reg, txt)
    reg = "INFO:root:loss: (.*?)\n"
    lossfunc = re.findall(reg, txt)
    reg = "INFO:root:ncentroids: (.*?)\n"
    ncentroids = re.findall(reg, txt)
    reg = "INFO:root:select_data: (.*?)\n"
    select_data = re.findall(reg, txt)
    reg = "INFO:root:select_positive: (.*?)\n"
    select_positive = re.findall(reg, txt)
    reg = "INFO:root:remark: (.*?)\n"
    remark = re.findall(reg, txt)
    if not len(select_positive):
        select_positive = ["no record"]
    print("train_mode,", train_mode[0],"data_mode,", data_mode[0],"backbone,", backbone[0],
     "valid_samples,", valid_samples[0], "freeze,", freeze[0], "NO" if resume[0] == "None" else "YES")
    reg = "\[eval\].*?acc: (.*?)," # maximize test accuracy
    acc = re.findall(reg, txt)
    acc = [float(x) for x in acc]
    if len(acc) != 0:
        if train_mode[0] == "self-cnn" and len(acc) > 100:
            acc = acc[100:]
        acc.reverse()
        print("Highest Test Acc:", max(acc), len(acc), len(acc) - (acc.index(max(acc))))
        reg = "\[eval\].*?loss: (.*?)," # minimize test loss
        loss = re.findall(reg, txt)
        loss = [float(x) for x in loss]
        loss.reverse()
        print("Lowest Test Loss:", min(loss), len(loss), len(loss) - (loss.index(min(loss))))
        return dir, train_mode[0], lossfunc[0], select_positive[0], ncentroids[0], select_data[0], resume_mode[0], freeze[0], train_samples[0], data_mode[0], \
            backbone[0], batch_size[0], ftlr[0], valid_samples[0], "NO" if resume[0] == "None" else "YES", \
            max(acc), len(acc), len(acc) - (acc.index(max(acc))), min(loss), len(loss), len(loss) - (loss.index(min(loss))), resume, remark
    else:
        return dir, train_mode[0], lossfunc[0], select_positive[0], ncentroids[0], select_data[0], resume_mode[0], freeze[0], train_samples[0], data_mode[0], \
            backbone[0], batch_size[0], ftlr[0], valid_samples[0], "NO" if resume[0] == "None" else "YES", \
            0, 0, 0, 0, 0, 0, resume, remark

def find_best_performance(filedir):
    info = []
    for i, dir in enumerate(sorted(os.listdir(filedir))):
        filename = os.path.join(filedir, dir, 'training.log')
        # resume_file = find_resumefile(filename, dir)
        # if resume_file[0] != "None" and resume_file[0][16:18]=="13":
        #     print(i,dir,resume_file[0][16:18])
        
        print(i,dir)
        # backbone, acc = find_max_acc(filename)
        info.append((find_max_acc(filename, dir)))
    df = pd.DataFrame(info)
    with pd.ExcelWriter("results/path.xlsx") as writer:
        df.to_excel(writer, sheet_name="one")


def find_last_acc(file, dir):
    with open(file) as f:
        txt = f.read()
    reg = "INFO:root:train_mode: (.*?)\n" # find train_mode
    train_mode = re.findall(reg, txt)
    reg = "INFO:root:data: (.*?)\n" # find train_mode
    data = re.findall(reg, txt)
    reg = "INFO:root:resume_mode: (.*?)\n" # find train_mode
    resume_mode = re.findall(reg, txt)
    reg = "INFO:root:data_mode: (.*?)\n"
    data_mode = re.findall(reg, txt)
    reg = "INFO:root:train_samples: (.*?)\n"
    train_samples = re.findall(reg, txt)
    reg = "INFO:root:backbone: (.*?)\n"
    backbone = re.findall(reg, txt)
    reg = "INFO:root:batch_size: (.*?)\n"
    batch_size = re.findall(reg, txt)
    reg = "INFO:root:ftlr: (.*?)\n"
    ftlr = re.findall(reg, txt)
    reg = "INFO:root:valid_samples: (.*?)\n"
    valid_samples = re.findall(reg, txt)
    reg = "INFO:root:resume: (.*?)\n"
    resume = re.findall(reg, txt)
    reg = "INFO:root:freeze: (.*?)\n"
    freeze = re.findall(reg, txt)
    reg = "INFO:root:loss: (.*?)\n"
    lossfunc = re.findall(reg, txt)
    reg = "INFO:root:ncentroids: (.*?)\n"
    ncentroids = re.findall(reg, txt)
    reg = "INFO:root:select_data: (.*?)\n"
    select_data = re.findall(reg, txt)
    reg = "INFO:root:select_positive: (.*?)\n"
    select_positive = re.findall(reg, txt)
    reg = "INFO:root:remark: (.*?)\n"
    remark = re.findall(reg, txt)
    if not len(select_positive):
        select_positive = ["no record"]
    print("train_mode,", train_mode[0],"data_mode,", data_mode[0],"backbone,", backbone[0],
     "valid_samples,", valid_samples[0], "freeze,", freeze[0], "NO" if resume[0] == "None" else "YES")
    reg = "\[eval\].*?acc: (.*?)," # maximize test accuracy
    acc = re.findall(reg, txt)
    acc = [float(x) for x in acc]
    if len(acc) != 0:
        if train_mode[0] == "self-cnn" and len(acc) > 100:
            acc = acc[100:]
        acc.reverse()
        print("Lastest Test Acc:", acc[0], len(acc))
        reg = "\[eval\].*?loss: (.*?)," # minimize test loss
        loss = re.findall(reg, txt)
        loss = [float(x) for x in loss]
        loss.reverse()
        print("Lowest Test Loss:", min(loss), len(loss), len(loss) - (loss.index(min(loss))))
        return dir, data, train_mode[0], lossfunc[0], select_positive[0], ncentroids[0], select_data[0], resume_mode[0], freeze[0], train_samples[0], data_mode[0], \
            backbone[0], batch_size[0], ftlr[0], valid_samples[0], "NO" if resume[0] == "None" else "YES", \
            max(acc), len(acc), len(acc) - (acc.index(max(acc))), min(loss), len(loss), len(loss) - (loss.index(min(loss))), resume, remark
    else:
        return dir, data, train_mode[0], lossfunc[0], select_positive[0], ncentroids[0], select_data[0], resume_mode[0], freeze[0], train_samples[0], data_mode[0], \
            backbone[0], batch_size[0], ftlr[0], valid_samples[0], "NO" if resume[0] == "None" else "YES", \
            0, 0, 0, 0, 0, 0, resume, remark


def find_last_accuracy(filedir):
    info = []
    for i, dir in enumerate(sorted(os.listdir(filedir))):
        filename = os.path.join(filedir, dir, 'training.log')
        
        print(i,dir)
        info.append((find_last_acc(filename, dir)))
    df = pd.DataFrame(info)
    with pd.ExcelWriter("results/path.xlsx") as writer:
        df.to_excel(writer, sheet_name="one")


def check_pretrain_model(filedir):
    for dir in sorted(os.listdir(filedir)):
        filename = os.path.join(filedir, dir, "training.log")
        if os.path.exists(os.path.join(filedir, dir, 'checkpoint_best.pt')):
            print(dir)
            find_max_acc(filename)


def fix_data():
    df = pd.read_excel('path.xlsx')
    df2 = df.loc[:, [1,2,9]]
    # df3 = df.iloc[2]
    df2.sort_values(by=2)
    print(df2)

def status_with_condition():
    condition = ["ncentroid", "selected_data"]
    df = pd.read_excel('results/path.xlsx', index_col=0)
    ndf = df.iloc[:,4]
    print(ndf.head())


def find_pretrain_checkpoint(filedir):
    for dir in sorted(os.listdir(filedir)):
        filename = f'/home/wangkai/6tdisk/wy/fault_diagnosis/runs/2022-08/14/{dir}/training.log'
        if os.path.exists(f'/home/wangkai/6tdisk/wy/fault_diagnosis/runs/2022-08/14/{dir}/checkpoint_best.pt'):
            print(dir)
            find_max_acc(filename)


def del_files(file_tree):
    # 批量删除文件夹
    for f in os.listdir(file_tree):
        if '-' in f:
            for i in os.listdir(os.path.join(file_tree, f)):
                if '-' in i:
                    shutil.rmtree(os.path.join(file_tree, f, i))


def clustering_acc(file):
    df = pd.read_excel(os.path.join("results", file), header=0)
    # print(df)
    # compute max in every column
    df2 = df.max(axis=0)
    # compute sum of df2
    Li = df2.sum()
    N = df.sum()
    N = N.sum()
    return Li, N, Li/N


if __name__ == '__main__':
    filedir = '../runs/2025-03/13'
    find_best_performance(filedir)
    # find_last_accuracy(filedir)
    # status_with_condition()
    # check_pretrain_model(filedir)
    # result_to_np(args.file_tree)
    # results_to_execl_1day(args.file_tree)
    # for file in os.listdir("./results"):
    #     if "confusion matrix" in file:
    #         print(file)
    #         print(clustering_acc(file))
    # break
