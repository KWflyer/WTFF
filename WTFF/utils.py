import os
import sys
import re
import time
import random

import pandas as pd
import torch
import yaml
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copyfile, copytree

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score


def time_file(temp):
    year_month = time.strftime('%Y-%m', time.localtime())
    day = time.strftime('%d', time.localtime())
    hour_min = time.strftime('%H-%M-%S', time.localtime())
    # 用时间创建文件夹
    file_root = '../runs/{}/{}/{}'.format(year_month, day, hour_min)

    # 判断路径是否存在，不存在创建文件夹
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    else:
        file_root += "_"
        os.makedirs(file_root)
    if not os.path.exists('../runs/temp'):
        os.makedirs('../runs/temp')
    with open(f'../runs/temp/{temp}.txt', 'w') as f:
        f.write(file_root)
    return file_root


def _create_model_training_folder(writer, files_to_save):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_save:
            if '.' in file:
                copyfile(file, os.path.join(
                    model_checkpoints_folder, os.path.basename(file)))
            else:
                copytree(file, os.path.join(
                    model_checkpoints_folder, os.path.basename(file)))


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


# 假如保存的文件重名，则在名字后面增加“_”
def change_name(path):
    while os.path.exists(path):
        if os.path.isdir(path):
            path = path + "_"
        else:
            dir_ = re.findall("(.*)\.", path)
            file_type = re.findall("(\.[a-z].*)", path)
            path = os.path.join(dir_[0] + '_' + file_type[0])
    return path


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 网格搜索，trainer: 需要运行的文件名， param: 指定的超参数范围
def grid_search(trainer, param, test=False, track=False):
    keys = [key for key in param.keys()]
    campare_keys, search_keys = [], []
    dim = 1
    len_param = [len(param[key]) for key in param.keys()]
    for i, l in enumerate(len_param):
        if l == 1:
            trainer += f' --{keys[i]} {param[keys[i]][0]}'
            campare_keys.append(keys[i])
        else:
            search_keys.append(keys[i])
            dim *= l
    campare_keys.extend(search_keys)
    trainer = 'python ' + trainer
    if len(search_keys) == 0:
        os.system(trainer)
    trainers, midtrainers = [], [trainer]
    for trainer in midtrainers:
        reset = trainer
        for key in search_keys:
            for i in param[key]:
                trainer += f' --{key} {i}'
                cond = re.findall('--(.*?) ', trainer)
                if cond == campare_keys:
                    trainers.append(trainer)
                    if test == False:
                        os.system(trainer)
                if len(trainers) == dim and track == True:
                    with open(change_name('../runs/grid_research.txt'), 'w') as f:
                        f.write(f'number of param set: {len(trainers)}\n')
                        f.write('parameters:\n')
                        for l in trainers:
                            f.write(f'{l}\n')
                    # sys.exit()
                    break
                midtrainers.append(trainer)
                trainer = reset
            if len(trainers) == dim:
                break
        # time.sleep(1)
        # print("no break")
        if len(trainers) == dim:
            break


# 随机搜索， trainer: 需要运行的文件名， param: 指定的超参数范围，limit: 随机搜索的时长，单位为秒
def random_search(trainer, param, limit):
    end = time.time()
    trainer = 'python ' + trainer
    reset = trainer
    trainers = []
    while (time.time() - end) < limit:
        for key in param.keys():
            if isinstance(param[key], list):
                trainer += f' --{key} {random.choice(param[key])}'
            if isinstance(param[key], tuple) and len(param[key]) == 2:
                trainer += f' --{key} {random.uniform(param[key][0], param[key][1])}'
        if trainer not in trainers:
            trainers.append(trainer)
            # time.sleep(1)
            os.system(trainer)
        trainer = reset
    with open(change_name('../runs/random_research.txt'), 'w') as f:
        f.write(f'number of param set: {len(trainers)}\n')
        f.write('parameters:\n')
        for l in trainers:
            f.write(f'{l}\n')


# -------------------施工中--------------------
def hyperband(trainer, param, m, n):
    trainer = 'python ' + trainer
    reset = trainer
    trainers = []
    end = 4 * n
    while n < end:
        for i in range(m):
            for key in param.keys():
                if isinstance(param[key], list):
                    trainer += f' --{key} {random.choice(param[key])}'  
                if isinstance(param[key], tuple) and len(param[key]) == 2:
                    trainer += f' --{key} {random.uniform(param[key][0], param[key][1])}'
            trainer += f'--epoch {n}'
            if trainer not in trainers:
                trainers.append(trainer)
                time.sleep(1)
            trainer = reset


def time_signal(signal, name=None, imshow=True):
    plt.figure(figsize=(10, 2))
    signal = signal[0].reshape(-1)
    x = range(0, len(signal))
    plt.plot(x, signal, lw=0.5, c='blue')
    if name is not None:
        plt.savefig('{}.jpg'.format(name))
    if imshow == True:
        plt.show()
    plt.close()


def classification_report_(label, pred, save_path):
    with open(change_name(os.path.join(save_path, 'classification_report.txt')), 'w') as f:
        f.write(classification_report(label, pred, digits=4))
    pd.DataFrame(confusion_matrix(label, pred)).to_excel(
        change_name(os.path.join(save_path, 'confusion matrix.xlsx')), index=True)


def tsne(feature, label, save_path):
    if isinstance(feature, list):
        feature = np.array(feature)

    print("start TSNE")

    # sklearn 版本
    Y = TSNE(n_components=2, perplexity=50, learning_rate=200,
             init='pca').fit_transform(feature)
    # tsnecuda 版本
    # Y = TSNE(n_iter=1000, verbose=1, num_neighbors=64).fit_transform(feature)
    print(Y.shape)

    scatterdata = []
    for a in range(len(set(label))):
        label_data = []
        for b in range(len(feature)):
            if label[b] == a:
                label_data.append(Y[b])
        label_data = np.array(label_data)
        scatterdata.append(label_data)

    plt.figure(figsize=(10, 10))

    c = [
        '#000080', '#0000ff', '#0061ff', '#00d5ff', '#4dffaa',
        '#aaff4d', '#ffe600', '#ff7a00', '#ff1300', '#800000',
        '#fff799', '#f9d3e3', '#ba5b49', '#d2af9d', '#3271ae',
        '#284852', '#007175', '#8a1874', '#13393e', '#9ebc19',
    ]
    scaplot = []
    for i, spdata in enumerate(scatterdata):
        g = plt.scatter(spdata[:, 0], spdata[:, 1], c=c[i])
        scaplot.append(g)

    # labels = np.arange(0, len(set(label)), 1).tolist()

    # plt.legend(scaplot, labels, bbox_to_anchor=(0, 0), ncol=len(set(label)))
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    plt.savefig(change_name(os.path.join(save_path, 'TSNE.png')))
    plt.close()


def normalize(f):
    norm2 = np.linalg.norm(f, ord=2, axis=1, keepdims=True)
    norm2[norm2==0] = 1e-12
    return f / norm2


def similarity_heat_map(feature, label, save_path):
    feature = normalize(feature)
    sim = np.matmul(feature, feature.T)

    label_nums = len(set(label))
    label_ = np.arange(label_nums)
    class_nums = len(label)//(label_nums)
    # initialize a empty dataframe
    df = pd.DataFrame(np.zeros((label_nums,label_nums)), index=label_, columns=label_)
    for i in range(label_nums):
        for j in range(label_nums):
            df[i][j] = sim[i*class_nums:(i+1)*class_nums, j*class_nums:(j+1)*class_nums].mean()
    ax = sns.heatmap(df, cmap='RdBu_r', vmax=1, square=True, fmt='.2f')
    heat_map = ax.get_figure()
    heat_map.savefig(change_name(os.path.join(save_path,'similarity_heatmap.png')))


def scatter_distributed_map(feature, label, save_path):
    feature = normalize(feature)
    sim = np.matmul(feature, feature[0].T)
    
    fig, ax = plt.subplots()
    ax.scatter(sim, label)

    ax.set_xlabel('similarity', fontsize=15)
    ax.set_ylabel('label', fontsize=15)
    ax.set_title('similarity distribution')

    ax.grid(True)
    fig.tight_layout()
    fig.savefig(change_name(os.path.join(save_path,'scatter_map.png')))


def cluster_eval(feature, label):
    if isinstance(feature, list):
        feature = np.array(feature)
    start = time.time()
    # 导入PCA类
    # from sklearn.decomposition import PCA

    # # 创建一个PCA对象，指定目标维度为24
    # pca = PCA(n_components=24)

    # # 对数据进行降维，得到一个24维的numpy数组X_pca
    # X_pca = pca.fit_transform(feature)
    feature = preprocessing.scale(feature, axis=0)

    # sc_score = silhouette_score(feature, label)
    ch_score = calinski_harabasz_score(feature, label)
    db_score = davies_bouldin_score(feature, label)
    print("ch:", ch_score, ", db:", db_score, ", time:", time.time()-start)


def splot(train_dataset, args):
    """
    对数据集中的信号进行可视化，1）时序的波形图
    """
    plt.figure(figsize=(12, 2))
    plt.xlim((0, 2048))

    # 查看数据维度
    print(train_dataset[args.sid][0].shape)
    print(train_dataset[args.sid][0].reshape(-1).shape)

    # 样本的id
    sid = 0
    y = train_dataset[sid][0].reshape(-1)
    plt.plot(y, c="blue", lw=0.5)

    # 对图像进行保存
    save_path = "../runs/pic"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 图片版本id
    vid = 0
    file_name = f"../runs/pic/{args.trainset}_{args.train_load}sid_{sid}vid_{vid}"
    while os.path.exists(file_name):
        vid += 1
        file_name = f"../runs/pic/{args.trainset}_{args.train_load}sid_{sid}_vid{vid}"
    plt.savefig(file_name)


def hsplot():
    '''
    plot the whole signal
    '''
    # 读入数据
    # 将文件夹下的数据读取到列表中
    file_root = "../dataset/CWRU/3HP"
    filenames = os.listdir(file_root)

    files = {}
    # 便利文件名列表读入 mat 数据
    for name in filenames:
        # 获取文件路径
        mat_path = os.path.join(file_root, name)
        # 读取 mat 文件
        mat_file = scipy.io.loadmat(mat_path)

        # 取出 mat_file 的keys
        file_keys = mat_file.keys()
        # 遍历 file_keys
        for key in file_keys:
            # 判断 key 字符是否包含特定字符：
            if 'DE' in key:
                files[name] = mat_file[key].ravel()
                print(mat_file[key].shape)
                time_signal(mat_file[key].reshape(-1), 'ori_signal')


def load_moco(model, args):
    """ Loads the pre-trained MoCo model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """
    print("\nLoading the model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['model']
    if args.resume_mode == 'moco':
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif args.resume_mode =='byol':
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('model') and not k.startswith('model.fc'):
                # remove prefix
                state_dict[k[len("model."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif args.resume_mode =='simclr':
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder') and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    else:
        raise "load paramenter error, no such resume mode!"

    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model


def load_timefreq(time_encoder, freq_encoder, args):
    print("\nLoading the model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename moco pre-trained keys
    time_dict = checkpoint['time_encoder']
    freq_dict = checkpoint['freq_encoder']
    
    for k in list(time_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('fc'):
            # delete renamed or unused k
            del time_dict[k]
    
    for k in list(freq_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('fc'):
            # delete renamed or unused k
            del freq_dict[k]
    
    # Load the encoder parameters
    time_encoder.load_state_dict(freq_dict, strict=False)
    freq_encoder.load_state_dict(freq_dict, strict=False)

    return time_encoder, freq_encoder


def load_classifier(model, args):
    print("\nLoading the model: {}\n".format(args.resume))
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint['classifier']
    model.load_state_dict(state_dict, strict=False)
    return model


def load_multi_modal(model, args):
    """ Loads the pre-trained MoCo model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """
    print("\nLoading the model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['time_encoder']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('fc'):
            # delete renamed or unused k
            del state_dict[k]
    
    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model

def load_time_multi_modal(model, args):
    print("\nLoading the time model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['time_encoder']
    
    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model

def load_freq_multi_modal(model, args):
    print("\nLoading the frequency model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['freq_encoder']
    
    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model
    


def load_sup(model, args, key=None):
    """ Loads the pre-trained supervised model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the supervised base_encoder.
    """
    # print("\n\nLoading the model: {}\n\n".format(args.resume))
    print("\nLoading the model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume)

    key = key if key else 'model'
    # rename moco pre-trained keys
    state_dict = checkpoint[key]
    if args.train_mode == "finetune":
        for k in list(state_dict.keys()):
            if k in ["fc.weight", "fc.bias"]:
                # delete renamed or unused k
                del state_dict[k]

    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model


def load_ae(model, args):
    # Load the pretrained model
    if args.resume:
        checkpoint = torch.load(args.resume)
        print("\nLoading the model: {}\n".format(args.resume))
    else:
        checkpoint = torch.load(args.checkpoint)
        print("\nLoading the model: {}\n".format(args.checkpoint))

    # rename moco pre-trained keys
    state_dict = checkpoint['model']
    if args.train_mode == "finetune":
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k in [
                    # 'classifier.fc1.1.weight', 'classifier.fc1.1.bias',
                    # 'classifier.fc2.1.weight', 'classifier.fc2.1.bias',
                    'classifier.fc3.1.weight', 'classifier.fc3.1.bias'
                     ]:
                # delete renamed or unused k
                del state_dict[k]
    # elif args.train_mode == "pm":
    #     for k in list(state_dict.keys()):
    #         if k.startswith('encoder'):
    #             # remove prefix
    #             state_dict[k[len("encoder."):]] = state_dict[k]

    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':

    # 使用随机噪声进行测试
    signal = np.random.randn(2048)
    # splot()

    # 使用随机噪声进行测试
    # signal = np.random.randn(64, 2048)
    # time_signal(signal, 'save_pic')

    # 提取结果的时间路径
    # default='runs/2021-10/27/15-09-28/checkpoints/model.pth'
    #
    # a = default.split('/')[:-2]
    # file_root = os.getcwd()
    # for b in a:
    #     file_root = os.path.join(file_root, b)
    #     print(file_root)
    # shutil.copytree('models', './runs/model')
