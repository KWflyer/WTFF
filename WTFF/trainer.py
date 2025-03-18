import os
import time
import math
import torch
import logging
import numpy as np
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from thop import profile, clever_format

import models
from optimizers import get_optimizer
from utils import *

class TimeFreqTrainer:
    def __init__(self, args, time_encoder, freq_encoder, time_projector, freq_projector):
        self.start = time.time()
        self.args = args
        self.time_encoder = time_encoder.to(args.device)
        self.freq_encoder = freq_encoder.to(args.device)
        self.time_projector = time_projector.to(args.device)
        self.freq_projector = freq_projector.to(args.device)
        self.optimizer = get_optimizer((self.time_encoder, self.freq_encoder, self.time_projector, self.freq_projector), args=args)

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=0,
                                  drop_last=False,
                                  shuffle=True)
        if self.args.pretrain_lr_scheduler:
            if self.args.warm > 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                T_max=(self.args.max_epochs-self.args.warm)*len(train_loader), eta_min=1e-5, last_epoch=-1)
                self.optimizer.param_groups[0]['lr'] = 1e-5
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                T_max=self.args.max_epochs*len(train_loader), eta_min=1e-5, last_epoch=-1)
            
        if self.args.active_log:
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.writer.log_dir}, model: "
                        f"{self.time_encoder.__class__}, optimizer: {self.optimizer.__class__}")
            logging.info(f"Start Model at {time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}")
            logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        best_train_loss = np.inf

        for epochs in range(self.args.max_epochs):
            data_iter = tqdm(enumerate(train_loader),
                             total=len(train_loader),
                             desc='Epoch %d' % epochs)
            total_loss = 0
            cur_loss = 0

            for i, ((batch_view_1, batch_view_2), _) in data_iter:
                batch_view_1 = batch_view_1.to(self.args.device)
                batch_view_2 = batch_view_2.to(self.args.device)

                if self.args.backbone == "tfcnn":
                    prediction1, time_features = self.time_encoder(batch_view_1)
                    prediction2, freq_features = self.freq_encoder(batch_view_2)

                    # logits, labels = self.InfoNCE_logits(prediction1, prediction2)
                    # loss = criterion(logits, labels).mean()

                    for j in range(len(time_features)):
                        time_features[j] = self.time_projector(time_features[j].squeeze(1))
                        freq_features[j] = self.freq_projector(freq_features[j].squeeze(1))
                    
                    # time_features[0] = self.pth_fft(time_features[0].squeeze(1))
                    # time_features[-1] = self.pth_fft(time_features[-1].squeeze(1))
                        
                    # time_features[-1] = time_features[-1].squeeze(1)
                    # logits_res0, labels_res0 = self.InfoNCE_logits(time_features[-1], freq_features[-1].squeeze(1))
                    # logits_res1, labels_res1 = self.InfoNCE_logits(time_features[-1], freq_features[-1].squeeze(1))
                    
                    logits_res0, labels_res0 = self.InfoNCE_logits(time_features[0], freq_features[-1])
                    logits_res1, labels_res1 = self.InfoNCE_logits(time_features[-1], freq_features[0])
                    loss = criterion(logits_res0, labels_res0).mean() + criterion(logits_res1, labels_res1).mean()
                    
                    # logits_res0, labels_res0 = self.InfoNCE_logits(time_features[5], freq_features[-1].squeeze(1))
                    # logits_res1, labels_res1 = self.InfoNCE_logits(time_features[-1], freq_features[5].squeeze(1))
                    # loss = criterion(logits_res0, labels_res0).mean() + criterion(logits_res1, labels_res1).mean()
                    
                    # time_features[1] = time_features[1].squeeze(1)
                    # time_features[-2] = time_features[-2].squeeze(1)
                    # logits_res2, labels_res2 = self.InfoNCE_logits(time_features[1], freq_features[-1].squeeze(1))
                    # logits_res3, labels_res3 = self.InfoNCE_logits(time_features[-1], freq_features[1].squeeze(1))
                    
                    # time_features[2] = time_features[2].squeeze(1)
                    # time_features[-3] = time_features[-3].squeeze(1)
                    # logits_res4, labels_res4 = self.InfoNCE_logits(time_features[2], freq_features[-3].squeeze(1))
                    # logits_res5, labels_res5 = self.InfoNCE_logits(time_features[-3], freq_features[2].squeeze(1))
                    
                    # loss = criterion(logits_res2, labels_res2).mean() + criterion(logits_res3, labels_res3).mean()
                    # loss = criterion(logits_res0, labels_res0).mean() + criterion(logits_res1, labels_res1).mean() + \
                    #     criterion(logits_res2, labels_res2).mean() + criterion(logits_res3, labels_res3).mean() + \
                    #     criterion(logits_res4, labels_res4).mean() + criterion(logits_res5, labels_res5).mean()
                    # loss = criterion(logits_res0, labels_res0).mean() + criterion(logits_res1, labels_res1).mean() + criterion(logits, labels).mean()
                    
                    total_loss += loss
                    cur_loss = loss
                else:
                    prediction1 = self.time_encoder(batch_view_1)
                    prediction2 = self.freq_encoder(batch_view_2)
                    logits, labels = self.InfoNCE_logits(prediction1, prediction2)
                    loss = criterion(logits, labels).mean()
                    total_loss += loss
                    cur_loss = loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0 and self.args.active_log:
                    self.args.writer.add_scalar('train loss', cur_loss, epochs * len(train_loader) + i)
                    self.args.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epochs * len(train_loader) + i)

                # warmup for the first setting epochs
                if self.args.pretrain_lr_scheduler:
                    if self.args.warm > 0 and epochs+1 <= self.args.warm:
                        wu_lr = (self.args.lr - self.args.base_lr) * \
                            (float(epochs * len(train_loader) + i) / \
                             self.args.warm / len(train_loader)) + self.args.base_lr
                        self.optimizer.param_groups[0]['lr'] = wu_lr
                    else:
                        scheduler.step()

            if self.args.active_log:
                logging.debug(f"Epoch: {epochs}, \tLoss: {total_loss}")
                print(f"Epoch: {epochs}, \tLoss: {total_loss}")
                self.save_model(epochs, self.args.checkpoint_dir)
                # save current best checkpoint with minium loss
                if total_loss <= best_train_loss:
                    self.save_model(epochs, self.args.checkpoint_dir[:-3] + "_best.pt")
                    best_train_loss = total_loss
        
        if self.args.active_log:
            runtime = divmod(time.time()-self.start, 60)
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.writer.log_dir}, "
                        f"with runtime {runtime[0]}m {runtime[1]}s.")
    
    def InfoNCE_logits(self, f1, f2):
        # Normalize the feature representations
        f1 = nn.functional.normalize(f1, dim=1)
        f2 = nn.functional.normalize(f2, dim=1)

        sim = torch.matmul(f1, f2.T)

        # generate mask for sim
        mask = torch.eye(sim.shape[0], dtype=torch.int64).to(self.args.device)

        pos = sim[mask].view(sim.shape[0], -1)
        neg = sim[~mask].view(sim.shape[0], -1)

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.args.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        return logits, labels
    
    def pth_fft(self, x):
        return torch.abs(torch.fft.fft(x, dim=-1)) / x.shape[-1]

    def save_model(self, epochs, save_path):
        # save checkpoint
        torch.save({
            'time_encoder': self.time_encoder.state_dict(),
            'freq_encoder': self.freq_encoder.state_dict(),
            'time_projector': self.time_projector.state_dict(),
            'freq_projector': self.freq_projector.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'epoch' : epochs,
        }, save_path)

# function print parameter with grad in model
def print_model_parameters_with_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            logging.info(name)

def finetune_tffusion(time_encoder, freq_encoder, classifier, train_dataset, val_dataset, args, time_projector=None, freq_projector=None):
    ''' Finetune script

        Freeze the encoders and train the supervised Linear Evaluation head with a Cross Entropy Loss.
    '''
    time_encoder = time_encoder.to(args.device)
    freq_encoder = freq_encoder.to(args.device)
    classifier = classifier.to(args.device)
    if time_projector is not None and freq_projector is not None:
        time_projector = time_projector.to(args.device)
        freq_projector = freq_projector.to(args.device)
    
    feature_result = []
    handle_feature_time = None
    handle_feature_freq = None

    def get_features_hook(self, input, output):
        feature_result.append(output)

    # 根据模型对应结构的所需节点处，添加钩子函数
    # if args.backbone == 'resnet18_1d':
    #     handle_feature_time = time_encoder.avgpool.register_forward_hook(get_features_hook)
    #     handle_feature_freq = freq_encoder.avgpool.register_forward_hook(get_features_hook)

    print_model_parameters_with_grad(time_encoder)
    print_model_parameters_with_grad(freq_encoder)
    print_model_parameters_with_grad(classifier)

    train_loader = DataLoader(train_dataset,
                            batch_size=args.finetune_batch_size,
                            num_workers=0,
                            drop_last=False,
                            shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.finetune_batch_size,
                            num_workers=0,
                            drop_last=False,
                            shuffle=False)

    ''' optimizers '''
    # Only optimise the supervised head
    optimizer = get_optimizer((time_encoder, freq_encoder, classifier), args)
    if time_projector is not None and freq_projector is not None:
        optimizer = get_optimizer((time_encoder, freq_encoder, time_projector, freq_projector, classifier), args)

    ''' Schedulers '''
    if args.finetune_lr_scheduler:
        # Cosine LR Decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    args.finetune_epochs, eta_min=1e-5)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # initilize Variables
    best_train_loss = np.inf
    best_valid_acc = 0.0
    saved_valid_loss = np.inf

    ''' Pretrain loop '''
    for epoch in range(args.finetune_epochs):

        time_encoder.train()
        freq_encoder.train()
        classifier.train()
        if time_projector is not None and freq_projector is not None:
            time_projector.train()
            freq_projector.train()
        
        run_loss = 0
        run_top1 = 0.0

        # Print setup for distributed only printing on one node.
        if args.active_log:
            train_dataloader = tqdm(train_loader,
                            total=len(train_loader),
                            desc='Epoch %d' % epoch)
        else:
            train_dataloader = train_loader

        ''' epoch loop '''
        for i, ((inputs_t, inputs_f), target) in enumerate(train_dataloader):
            inputs_t = inputs_t.to(args.device)
            inputs_f = inputs_f.to(args.device)
            target = target.to(args.device)

            feature_result = []

            # Forward pass
            optimizer.zero_grad()

            # Do not compute the gradients for the frozen encoder
            if args.backbone == 'tfcnn':
                output_t, features_t = time_encoder(inputs_t)
                output_f, features_f = freq_encoder(inputs_f)
                if time_projector is not None and freq_projector is not None:
                    features_t[0] = time_projector(features_t[0].squeeze(1))
                    features_f[0] = freq_projector(features_f[0].squeeze(1))
                    features_t[-1] = time_projector(features_t[-1].squeeze(1))
                    features_f[-1] = freq_projector(features_f[-1].squeeze(1))
                output_t = features_t[0].squeeze(1) + features_t[-1].squeeze(1)
                output_f = features_f[0].squeeze(1) + features_f[-1].squeeze(1)
                # output_t = features_t[-1].squeeze(1)
                # output_f = features_f[-1].squeeze(1)
            else:
                output_t = time_encoder(inputs_t)
                output_f = freq_encoder(inputs_f)
                # if args.backbone == 'resnet18_1d':
                #     output_t1 = feature_result[0].squeeze(2)
                #     output_f1 = feature_result[1].squeeze(2)
            output = classifier(torch.cat((output_t, output_f), dim=1))

            # Take pretrained encoder representations
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            run_top1 += accuracy(output, target)[0][0]
            

        epoch_finetune_loss = run_loss / len(train_loader)
        epoch_finetune_acc = run_top1 / len(train_loader)
        ''' Update Schedulers '''
        if args.finetune_lr_scheduler:
            # Decay lr with CosineAnnealingLR
            scheduler.step()

        ''' Printing '''
        if args.active_log:
            logging.info('Epoch {}/{}, [Finetune] loss: {:.4f},\t acc: {:.4f}, \t'.format(
                epoch+1, args.finetune_epochs, epoch_finetune_loss, epoch_finetune_acc))
            args.writer.add_scalar('finetune_epoch_loss_train', epoch_finetune_loss, epoch+1)
            args.writer.add_scalar('finetune_epoch_acc_train', epoch_finetune_acc, epoch+1)
            args.writer.add_scalar('finetune_lr_train', optimizer.param_groups[0]['lr'], epoch+1)

        valid_loss, valid_acc = evaluate_tffusion(
            time_encoder, freq_encoder, classifier, val_loader, epoch, args, time_projector, freq_projector)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if run_loss <= best_train_loss:
            best_epoch = epoch + 1
            best_valid_acc = valid_acc
            best_train_loss = run_loss
            saved_valid_loss = valid_loss

            
            if time_projector is not None and freq_projector is not None:
                state = {
                    'time_encoder': time_encoder.state_dict(),
                    'time_projector': time_projector.state_dict(),
                    'freq_encoder': freq_encoder.state_dict(),
                    'freq_projector': freq_projector.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
            else:
                state = {
                    'time_encoder': time_encoder.state_dict(),
                    'freq_encoder': freq_encoder.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }

            if args.active_log:
                torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))

        epoch_finetune_loss = None  # reset loss
        epoch_finetune_acc = None
    print('Epoch {}/{}, [Finetune] best train loss: {:.4f},\t valid loss {:.4f}, acc: {:.4f}'.format(
                best_epoch, args.finetune_epochs, best_train_loss, saved_valid_loss, best_valid_acc))
    
    del state
    # if handle_feature_time: handle_feature_time.remove()
    # if handle_feature_freq: handle_feature_freq.remove()


def evaluate_tffusion(time_encoder, freq_encoder, classifier, val_loader, epoch, args, time_projector=None, freq_projector=None):
    time_encoder = time_encoder.to(args.device)
    freq_encoder = freq_encoder.to(args.device)
    classifier = classifier.to(args.device)
    if time_projector is not None and freq_projector is not None:
        time_projector = time_projector.to(args.device)
        freq_projector = freq_projector.to(args.device)

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # Evaluate both encoder and class head
    time_encoder.eval()
    freq_encoder.eval()
    classifier.eval()
    if time_projector is not None and freq_projector is not None:
        time_projector.eval()
        freq_projector.eval()
    
    feature_result = []
    handle_feature_time = None
    handle_feature_freq = None

    def get_features_hook(self, input, output):
        feature_result.append(output)

    # 根据模型对应结构的所需节点处，添加钩子函数
    # if args.backbone == 'resnet18_1d':
    #     handle_feature_time = time_encoder.avgpool.register_forward_hook(get_features_hook)
    #     handle_feature_freq = freq_encoder.avgpool.register_forward_hook(get_features_hook)

    # initilize Variables
    run_loss = 0
    run_top1 = 0.0
    
    eval_dataloader = tqdm(val_loader,
                        total=len(val_loader),
                        desc='Epoch %d' % epoch)

    ''' epoch loop '''
    for i, ((inputs_t, inputs_f), target) in enumerate(eval_dataloader):
        # Do not compute gradient for encoder and classification head
        time_encoder.zero_grad()
        freq_encoder.zero_grad()
        classifier.zero_grad()

        inputs_t = inputs_t.to(args.device)
        inputs_f = inputs_f.to(args.device)
        target = target.to(args.device)

        # Forward pass
        if args.backbone == 'tfcnn':
            output_t, features_t = time_encoder(inputs_t)
            output_f, features_f = freq_encoder(inputs_f)
            if time_projector is not None and freq_projector is not None:
                features_t[0] = time_projector(features_t[0].squeeze(1))
                features_f[0] = freq_projector(features_f[0].squeeze(1))
                features_t[-1] = time_projector(features_t[-1].squeeze(1))
                features_f[-1] = freq_projector(features_f[-1].squeeze(1))
            output_t = features_t[0].squeeze(1) + features_t[-1].squeeze(1)
            output_f = features_f[0].squeeze(1) + features_f[-1].squeeze(1)
            # output_t = features_t[-1].squeeze(1)
            # output_f = features_f[-1].squeeze(1)
        else:
            output_t = time_encoder(inputs_t)
            output_f = freq_encoder(inputs_f)
            # if args.backbone == 'resnet18_1d':
            #     output_t = feature_result[0].squeeze(2)
            #     output_f = feature_result[1].squeeze(2)
        output = classifier(torch.cat((output_t, output_f), dim=1))
        loss = criterion(output, target)

        run_loss += loss.item()
        run_top1 += accuracy(output, target)[0][0]
        feature_result = []

    epoch_valid_loss = run_loss / len(val_loader) 
    epoch_valid_acc = run_top1 / len(val_loader)

    ''' Printing '''
    if args.active_log:
        logging.info('[eval] loss: {:.4f},\t acc: {:.4f},\t '.format(
            epoch_valid_loss, epoch_valid_acc))
        if args.train_mode == 'finetune':
            args.writer.add_scalar('finetune_epoch_loss_eval', epoch_valid_loss, epoch+1)
            args.writer.add_scalar('finetune_epoch_acc_eval', epoch_valid_acc, epoch+1)
    
    # if handle_feature_time: handle_feature_time.remove()
    # if handle_feature_freq: handle_feature_freq.remove()

    return epoch_valid_loss, epoch_valid_acc


def analysis(model, finetune_trainset, finetune_testset, args):
    val_train_loader = torch.utils.data.dataloader.DataLoader(finetune_trainset,
                                                batch_size=args.finetune_batch_size,
                                                num_workers=0,
                                                drop_last=False,
                                                shuffle=False)
    val_test_loader = torch.utils.data.dataloader.DataLoader(finetune_testset,
                                                batch_size=args.finetune_batch_size,
                                                num_workers=0,
                                                drop_last=False,
                                                shuffle=False)

    # Loss / Criterion
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # 建立钩子函数，创建数组接收结果
    feature_result = []

    def get_features_hook(self, input, output):
        feature_result.append(input[0].detach().cpu().numpy())

    # 根据模型对应结构的所需节点处，添加钩子函数
    # handle_feature = model.wd[21].register_forward_hook(get_features_hook)

    # Evaluate both encoder and class head
    model.eval()

    # initilize Variables
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0
    pred_list = []
    label_list = []

    if args.classifier in ['knn', 'svm', 'mlp']:
        eval_dataloader = tqdm(val_train_loader,
                            total=len(val_train_loader))

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(eval_dataloader):
            # Do not compute gradient for encoder and classification head
            model.zero_grad()

            inputs = inputs.to(args.device)
            target = target.to(args.device)

            output = model(inputs)

            label_list += target.detach().cpu().numpy().tolist()
        
        train_feature = np.concatenate(feature_result, axis=0)
        train_label = label_list

        feature_result = []
        label_list = []
    
    eval_dataloader = tqdm(val_test_loader,
                        total=len(val_test_loader))

    for i, (inputs, target) in enumerate(eval_dataloader):
        # Do not compute gradient for encoder and classification head
        model.zero_grad()

        inputs = inputs.to(args.device)
        target = target.to(args.device)

        # Forward pass
        output = model(inputs)
        loss = criterion(output, target)

        torch.cuda.synchronize()

        run_loss += loss.item()
        predicted = output.argmax(-1)
        acc = accuracy(output, target)
        run_top1 += acc[0]
        pred_list += predicted.detach().cpu().numpy().tolist()
        label_list += target.detach().cpu().numpy().tolist()
    
    # 取出中间层的内容
    # feature = feature_result[0]
    feature = np.concatenate(feature_result, axis=0)
    
    # 去除钩子
    # handle_feature.remove()
    
    if args.classifier in ['knn', 'svm', 'mlp']:
        test_feature = feature
        test_label = np.array(label_list)

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train_feature, train_label)
        pred = neigh.predict(test_feature)
        knn_top1 = (pred == test_label).sum() / len(test_label)
        logging.info('[eval] acc: {:.4f}'.format(knn_top1))
        print('[eval] acc: {:.4f}'.format(knn_top1))
    elif args.classifier =='svm':
        clf = SVC(kernel='rbf', C=9, gamma=0.1)
        clf.set_params(kernel='rbf', probability=True).fit(train_feature, train_label)
        test_pre = clf.predict(test_feature)
        SVM_AC = accuracy_score(test_label, test_pre)
        # SVM_f1 = f1_score(test_label, test_pre, average='macro')
        logging.info('[eval] acc: {:.4f}'.format(SVM_AC))
        print('[eval] acc: {:.4f}'.format(SVM_AC))
    elif args.classifier == 'mlp':
        MLP = MLPClassifier(solver='lbfgs', alpha=1e-4, batch_size=64, learning_rate_init=0.01,
                        hidden_layer_sizes=(256), random_state=1, max_iter=2000)
        MLP.fit(train_feature, train_label)
        MLP_predict = MLP.predict(test_feature)
        MLP_AC = accuracy_score(test_label, MLP_predict)
        # MLP_f1 = f1_score(test_label, MLP_predict, average='macro')
        logging.info('[eval] acc: {:.4f}'.format(MLP_AC))
        print('[eval] acc: {:.4f}'.format(MLP_AC))

    # classification_report_(pred_list, label_list, args.resume.split("check")[0])
    # tsne(feature, label_list, save_path=args.resume.split("check")[0])
    # classification_report_(pred_list, label_list, "results")
    # tsne(feature, label_list, save_path="results")
    # similarity_heat_map(feature, label_list, save_path="results")
    scatter_distributed_map(feature, label_list, save_path="results")
    # cluster_eval(feature, label_list)

    return feature


