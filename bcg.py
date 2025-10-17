from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import sys
import numpy as np
import time
import torch.nn.functional as F
from sklearn.cluster import KMeans
import copy
import time
import sklearn
from torch.nn import DataParallel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from models import model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data import data_preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--dataroot', default=None)
parser.add_argument('--syn_num', type=int, default=None, help='number features to generate per class')
parser.add_argument('--batch_size', type=int, default=None, help='detector batch size')
parser.add_argument('--resSize', type=int, default=None, help='size of visual features')
parser.add_argument('--attSize', type=int, default=None, help='size of semantic features')
parser.add_argument('--nz', type=int, default=None, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=None, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=None, help='size of the hidden units in discriminator')
parser.add_argument('--big_iter', type=int, default=None, help='number of epochs to train for')
parser.add_argument('--lr_detector', type=float, default=None, help='learning rate to train GANs ')
parser.add_argument('--lr_condition', type=float, default=None, help='learning rate to train softmax classifier')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--split', type=int, default=None, help='split id')
parser.add_argument('--gpu', type=int, default=0, help='split id')

opt = parser.parse_args()

class FeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data[0].shape[0]


    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

class FeatDataset2(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data[0].shape[0]

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][self.data[1][idx]]


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x, features=False):
        if features:
            return self.fc(x)
        o = self.logic(self.fc(x))
        return o

class LINEAR_LOGSOFTMAX2(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX2, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x, features=False):

            x = torch.nn.functional.relu(self.fc1(x))
            # x = torch.nn.functional.sigmoid(self.fc1(x))
            x = self.fc2(x)
            if features:
                return x
            else:
                return self.logic(x)


def compute_roc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class BCG:
    def __init__(self, data):

        global opt
        self.device = opt.device
        self.data = data
        self.nclass = data.seenclasses.size(0)
        self.input_dim = opt.resSize
        self.semantics = data.train_att.to(self.device)
        self.semantics_dim = self.semantics.shape[1]
        self.syn_feature_all = None
        self.syn_label_all = None

        self.batchsize = opt.batchsize_detector
        self.epoch_detector = opt.epoch_detector
        self.epoch_detector_initial = opt.epoch_detector_initial
        self.lr_detector = opt.lr_detector
        self.lr_bcg_initial = opt.lr_bcg_initial
        self.lr_bcg = opt.lr_bcg
        self.epoch_bcg_initial_milestone = opt.epoch_bcg_initial_milestone
        self.epoch_bcg_milestone = opt.epoch_bcg_milestone

        self.big_iter = opt.big_iter
        self.epoch_bcg_initial = opt.epoch_bcg_initial
        self.epoch_bcg = opt.epoch_bcg
        self.boundary_nclass = 0
        self.update_boundary_nclass = opt.update_boundary_nclass
        self.ibnum = opt.ibnum


        self.boundary_class = None
        self.maxmargin = opt.maxmargin
        self.minmargin = opt.minmargin
        self.dissim = opt.dissim
        self.sim = opt.sim
        self.diversity_weight = opt.diversity_weight
        self.hard_weight = opt.hard_weight

        self.netG = model.MLP_G(opt)
        SAVE_PARA_PATH = './checkpoints/' + "gan_g" +str(opt.epoch_cgan)+ '.pth'
        print(SAVE_PARA_PATH)
        # SAVE_PARA_PATH = './checkpoints/' + "gan_g" + str(5) + '.pth'
        # self.netG.load_state_dict(torch.load(SAVE_PARA_PATH, map_location={'cuda:0': "cuda:" + str(int(opt.device))}))
        self.netG.load_state_dict(torch.load(SAVE_PARA_PATH))

        # self.netD = model.MLP_CRITIC(opt)
        # SAVE_PARA_PATH = './checkpoints/' + "gan_d" +str(opt.epoch_cgan)+ '.pth'
        # SAVE_PARA_PATH = './checkpoints/' + "gan_d" + str(5) + '.pth'
        # self.netD.load_state_dict(torch.load(SAVE_PARA_PATH))

        self.best_roc = 0
        self.best_oscr = 0
        self.best_f1 = 0

        # self.netD = DataParallel(self.netD)
        # self.netG = DataParallel(self.netG)
        # self.netD.to(self.device)
        self.netG.to(self.device)


        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass + self.nclass * (self.update_boundary_nclass*(self.big_iter-1))).to(self.device)
        # self.model = LINEAR_LOGSOFTMAX2(self.input_dim, self.nclass + self.nclass * (
        #             self.update_boundary_nclass * (self.big_iter - 1))).to(self.device)

        self.model.apply(weights_init)
        if not opt.openauc:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = OpenAUCLoss(num_classes=self.nclass)
        self.criterion2 = nn.NLLLoss()

        # setup optimizer
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_detector, betas=(0.5, 0.999))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr_detector, momentum=0.9,
                                         weight_decay=5e-4,
                                         nesterov=True)

        data.train_feature = data.train_feature.to(self.device)
        data.train_label = data.train_label.to(self.device)
        data.train_att = data.train_att.to(self.device)
        data.test_seen_feature = data.test_seen_feature.to(self.device)
        data.test_seen_label = data.test_seen_label.to(self.device)
        data.test_unseen_feature = data.test_unseen_feature.to(self.device)
        data.test_unseen_label = data.test_unseen_label.to(self.device)

        self.trainset = FeatDataset(data=[data.train_feature, data.train_label])
        self.traindataloader = DataLoader(self.trainset, batch_size=opt.batchsize_detector, shuffle=True, num_workers=0)

        self.testset = FeatDataset(data=[data.test_seen_feature, data.test_seen_label])
        self.testdataloader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=0)

        self.openset = FeatDataset(data=[data.test_unseen_feature, data.test_unseen_label])
        self.opendataloader = DataLoader(self.openset, batch_size=128, shuffle=False, num_workers=0)

        self.fit()


    def fit(self):



        for iter in range(self.big_iter):

            print("iter: %d,  boundary_nclass: %d" % (iter, self.boundary_nclass))

            if iter > 0:
                # self.classifier_epoch = 100
                training_epoch = self.epoch_detector



            else:
                training_epoch = self.epoch_detector_initial
                training_epoch = 1




            for epoch in range(training_epoch):

                if iter > 0:

                    self.trainset = FeatDataset(data=[self.data.train_feature, self.data.train_label])
                    self.traindataloader = DataLoader(self.trainset, batch_size=opt.batchsize_detector_2, shuffle=True,
                                                      num_workers=0)


                train_correct = AverageMeter()
                for batch_idx, samples in enumerate(self.traindataloader):

                    # inputv, labelv, attv = samples
                    inputv, labelv = samples
                    self.model.zero_grad()

                    if not opt.openauc:
                        output = self.model(inputv, features=True)[:,:self.boundary_nclass*self.nclass+self.nclass]
                        loss1 = self.criterion(output, labelv)
                    else:
                        output = self.model(inputv, features=True)
                        _, loss1 = self.criterion(output, labelv, self.model.fc, inputv)


                    # # 无预热，效果一般般，至少收敛缓慢
                    # output = self.model(inputv, features = True)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)]
                    # loss1 = self.criterion(output, labelv)

                    # # BEST 收敛速度和结果都略胜一筹的，应该归功于对剩余类的更强预训练。
                    # output = self.model(inputv)
                    # loss1 = self.criterion2(output, labelv)

                    # 一般预热OKAY
                    # output = self.model(inputv)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)]
                    # loss1 = self.criterion2(output, labelv)

                    _, preds = torch.max(output, 1)
                    train_correct.update(preds.eq(labelv).sum().item(), len(inputv))

                    # if iter > 0:
                    if False:

                        this_batchsize = len(self.syn_label_all)
                        # this_batchsize = opt.batchsize_detector

                        index = torch.randperm(len(self.syn_label_all))
                        this_input = Variable(self.syn_feature_all[index[:this_batchsize]])
                        this_label = Variable(self.syn_label_all[index[:this_batchsize]])

                        # BEST 8477
                        output2 = self.model(this_input, features=True)[:,:self.boundary_nclass*self.nclass+self.nclass]
                        loss2 = self.criterion(output2, this_label)
                        # loss2 = 0


                        # # 无预热，效果一般般
                        # output2 = self.model(self.syn_feature_all, features = True)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)]
                        # loss2 = self.criterion(output2, self.syn_label_all)

                        # # BETTER 收敛速度和结果都略胜一筹的，应该归功于对剩余类的预训练。
                        # output2 = self.model(self.syn_feature_all)
                        # loss2 = self.criterion2(output2, self.syn_label_all)

                        # # 一般预热OKAY
                        # output2 = self.model(self.syn_feature_all)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)]
                        # loss2 = self.criterion2(output2, self.syn_label_all)


                        _, preds2 = torch.max(output2, 1)
                        loss = loss1+ loss2
                        train_correct.update(preds2.eq(this_label).sum().item(), len(this_label))

                    # test manifold_mixup
                    # if True:
                    #     self.generate_mixup_samples(inputv)
                    #     # self.syn_feature_all = self.syn_feature_all[:2048]
                    #     # self.syn_label_all = self.syn_label_all[:2048]
                    #
                    #     output2 = self.model(self.syn_feature_all)
                    #     loss2 = self.criterion(output2, self.syn_label_all)
                    #     _, preds2 = torch.max(output2, 1)
                    #
                    #     loss = loss1 + loss2

                    else:
                        loss = loss1

                    loss.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    acc_seen, auc_score_av, auc_score_softmax, oscr, f1_score = self.val(iter, False)



                if (iter==0) and (epoch+1 == training_epoch):
                    SAVE_PARA_PATH = './checkpoints/' + "detector_softmax" + '.pth'
                    torch.save(self.model.state_dict(), SAVE_PARA_PATH)


                if self.best_roc<auc_score_softmax :
                    self.best_roc = auc_score_softmax
                    SAVE_PARA_PATH = './checkpoints/' + "detector_best" + '.pth'
                    torch.save(self.model.state_dict(), SAVE_PARA_PATH)

                if self.best_oscr<oscr:
                    self.best_oscr = oscr

                if self.best_f1 < f1_score:
                    self.best_f1 = f1_score


                print("epoch:%d, train_acc:%.4f, acc:%.4f, av_roc:%.4f, softmax_roc:%.4f, best_softmax_roc:%.4f,                     oscr:%.4f, best_oscr:%.4f,f1:%.4f, best_f1:%.4f," % (
                    epoch, train_correct.avg, acc_seen, auc_score_av, auc_score_softmax, self.best_roc, oscr, self.best_oscr, f1_score, self.best_f1))

                print("\n")

            if opt.openauc:
                exit()

            if iter == self.big_iter-1:
                if opt.f1:
                    with torch.no_grad():
                        print("computing F1")

                        print("softmax:")

                        SAVE_PARA_PATH = './checkpoints/' + "detector_softmax" + '.pth'
                        self.model.load_state_dict(torch.load(SAVE_PARA_PATH))
                        acc_seen, auc_score_av, auc_score_softmax, oscr, f1_score = self.val(iter, True)
                        print(
                            "epoch:%d, train_acc:%.4f, acc:%.4f, av_roc:%.4f, softmax_roc:%.4f, best_softmax_roc:%.4f,                     oscr:%.4f, best_oscr:%.4f,f1:%.4f, best_f1:%.4f," % (
                                epoch, train_correct.avg, acc_seen, auc_score_av, auc_score_softmax, self.best_roc, oscr,
                                self.best_oscr, f1_score, self.best_f1))

                        print("bcg:")

                        SAVE_PARA_PATH = './checkpoints/' + "detector_best" + '.pth'
                        self.model.load_state_dict(torch.load(SAVE_PARA_PATH))
                        acc_seen, auc_score_av, auc_score_softmax, oscr, f1_score = self.val(iter, True)
                        print(
                            "epoch:%d, train_acc:%.4f, acc:%.4f, av_roc:%.4f, softmax_roc:%.4f, best_softmax_roc:%.4f,                     oscr:%.4f, best_oscr:%.4f,f1:%.4f, best_f1:%.4f," % (
                                epoch, train_correct.avg, acc_seen, auc_score_av, auc_score_softmax, self.best_roc, oscr,
                                self.best_oscr, f1_score, self.best_f1))

                print("all finished")

                exit()



            #######################更新边界类################################
            self.update_boundary_class = Variable(torch.rand([self.nclass, self.update_boundary_nclass, self.semantics_dim]).to(self.device))
            self.update_boundary_class.requires_grad = True
            self.update_boundary_optimizer = torch.optim.SGD([self.update_boundary_class], lr=self.lr_bcg_initial, momentum=0, weight_decay=0,
                                                  nesterov=False)
            train_scheduler = optim.lr_scheduler.MultiStepLR(self.update_boundary_optimizer, milestones=self.epoch_bcg_initial_milestone, gamma=0.1)

            #######################初始化边界类################################
            for k in range(self.epoch_bcg_initial):
                loss = self.bcg_loss()
                self.update_boundary_optimizer.zero_grad()
                loss.backward()
                self.update_boundary_optimizer.step()
                train_scheduler.step()



            # ############### check ###################
            #             dis = torch.abs(self.update_boundary_class - self.semantics[0])
            #             print("所有反事实类距离类0的语义距离，要求都在3以上,第一行则在3~10之间，其他行大于3")
            #             print(torch.sum((dis > self.dissim) * 1, dim=2))
            #             print("类0的反事实类与类0的不相似距离，要求在10以内")
            #             print(dis.shape[2] - torch.sum((dis[0] < self.sim) * 1, dim=1))
            #             print("一组同类的反事实类的类间距，越大越好，比如都大于10")
            #             print(torch.sum(torch.abs(self.update_boundary_class[0] - self.update_boundary_class[0][0]) > self.dissim, dim=1))
            #             print("------------------")
            #             print(self.semantics[0])
            #             print(self.update_boundary_class[0][0])
            #             print(self.update_boundary_class[0][1])
            #
            #             print("finish_boundary_class_initiation!!!!")
            #             exit()
            # ############## check ###################

            self.update_boundary_optimizer = torch.optim.SGD([self.update_boundary_class], lr=self.lr_bcg, momentum=0,
                                                             weight_decay=0,
                                                             nesterov=False)
            train_scheduler = optim.lr_scheduler.MultiStepLR(self.update_boundary_optimizer,
                                                             milestones=self.epoch_bcg_milestone, gamma=0.1)

            #######################更新边界类################################

            for epoch in range(self.epoch_bcg):

                self.generate_update_boundary_samples()
                # print(self.syn_label_all)
                # exit()
                output = self.model(self.syn_feature_all)
                #####使用最大可能已知类###########这个sota
                known_output = output[:,:self.nclass]
                _, index_sort = torch.sort(known_output, dim=1, descending=True)
                known_label = index_sort[:,0]
                #####使用中心已知类###########
                # known_label = self.syn_label_all


                # loss = 1 * self.criterion2(output, known_label) + 1 * self.bcg_loss()
                loss = self.hard_weight * self.criterion2(output, known_label) + 1 * self.bcg_loss()

                self.update_boundary_optimizer.zero_grad()
                loss.backward()
                self.update_boundary_optimizer.step()
                train_scheduler.step()

                ###############check

                if epoch == self.epoch_bcg-1:


                    dis = torch.abs(self.update_boundary_class - self.semantics[0])
                    print("--------------")
                    print("所有反事实类距离类0的不相似距离，要求都在3以上")
                    print(torch.sum((dis > self.dissim) * 1, dim=2))
                    print("类0的边界条件与类0的不相似距离，要求在10以内")
                    print(dis.shape[2] - torch.sum((dis[0] < self.sim) * 1, dim=1))
                    print("一组同类的反事实类的类间距，越大越好，比如都大于10")
                    print(
                        torch.sum(torch.abs(self.update_boundary_class[0] - self.update_boundary_class[0][0]) > self.dissim, dim=1))
                    print("------------------")
                    print("原型")
                    print(self.semantics[0])
                    print("条件一")
                    print(self.update_boundary_class[0][0])
                    print("条件二")
                    print(self.update_boundary_class[0][1])

                    check = torch.sort(torch.nn.functional.softmax(output, dim=1)[:, :self.nclass], dim=1)[0][:, -1].shape[0]
                    print("最大可能已知类中位数", torch.sort(
                        torch.sort(torch.nn.functional.softmax(output, dim=1)[:, :self.nclass])[0][:, -1])[
                              0][int(check * 0.50-10):int(check * 0.50) + 10])

                    print("最大可能平均数",
                        torch.sum(torch.sort(torch.nn.functional.softmax(output, dim=1)[:, :self.nclass])[0][:, -1])/check)
                    # exit()


                    score = [torch.nn.functional.softmax(output, dim=1)[i, self.syn_label_all[i]].item() for i in
                           range(len(output))]
                    score.sort()
                    print("中心已知类难度  75%分位数", score[len(score)//4*2])
                    print("中心已知类难度 平均数", sum(score)/len(score))
                    # if input()=="x":
                    #     exit()
                ##############check



            self.generate_anti_feature()

            if self.boundary_class == None:
                self.boundary_class = self.update_boundary_class
            else:
                self.boundary_class = torch.cat([self.boundary_class,self.update_boundary_class],dim=1)

            self.boundary_nclass = self.boundary_nclass + self.update_boundary_nclass

    def val(self, iter, f1_flag):

        known_scores_av = []
        known_scores_softmax = []
        known_labels = []
        known_preds = []
        unknown_preds = []
        test_correct = AverageMeter()

        for batch_idx, samples in enumerate(self.testdataloader):

            inputv, labelv = samples

            output = torch.nn.functional.softmax(
                self.model(inputv, features=True)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)],
                dim=1)[:, :self.nclass]

            # output = torch.nn.functional.softmax(
            #     self.model(inputv, features=True)[:, :self.nclass + self.nclass * (self.update_boundary_nclass * 1)],
            #     dim=1)[:, :self.nclass]

            # OKAY TOO
            # output = torch.nn.functional.softmax(
            #     self.model(inputv, features=True),
            #     dim=1)[:, :self.nclass]

            av_output = self.model(inputv, features=True)[:,
                        :self.nclass]

            known_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            known_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            known_labels.extend([labelv[i].cpu().numpy() for i in range(len(inputv))])

            _, preds = torch.max(av_output, 1)

            known_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

            test_correct.update(preds.eq(labelv).sum().item(), len(inputv))



        unknown_scores_av = []
        unknown_scores_softmax = []

        for batch_idx, samples in enumerate(self.opendataloader):

            inputv, labelv = samples

            output = torch.nn.functional.softmax(
                self.model(inputv, features=True)[:,:self.nclass + self.nclass * (self.update_boundary_nclass*iter)],
                dim=1)[:, :self.nclass]

            # output = torch.nn.functional.softmax(
            #     self.model(inputv, features=True)[:, :self.nclass + self.nclass * (self.update_boundary_nclass * 1)],
            #     dim=1)[:, :self.nclass]

            # OKAY TOO
            # output = torch.nn.functional.softmax(
            #     self.model(inputv, features=True),
            #     dim=1)[:, :self.nclass]

            av_output = self.model(inputv, features=True)[:,
                        :self.nclass]

            unknown_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            unknown_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            _, preds = torch.max(av_output, 1)

            unknown_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

        distances = sorted(known_scores_softmax)
        self.threshold_list = [distances[int((i / 100) * len(distances))] for i in range(0, 100, 1)]


        auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
        auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)
        oscr = compute_oscr(np.array(known_scores_softmax), np.array(unknown_scores_softmax), np.array(known_preds), np.array(known_labels))

        all_labels = known_labels+[-1 for i in range(len(unknown_scores_av))]
        all_score = known_scores_softmax + unknown_scores_softmax
        all_preds_buff = known_preds+unknown_preds
        f1_score = -1

        if f1_flag:
        # if False:
            for threshold in range(1,1000):
                all_pred = [all_preds_buff[i] if all_score[i]>(threshold/10) else -1  for i in range(len(all_preds_buff))  ]
                if sklearn.metrics.f1_score(all_labels, all_pred, average="macro")>f1_score:
                    f1_score = sklearn.metrics.f1_score(all_labels, all_pred, average="macro")

        best_acc = 0
        best_threshold = 0
        best_ratio = 0
        p_acc = 0
        n_acc = 0

        for jj, threshold in enumerate(self.threshold_list):

            TP = sum((known_scores_softmax > threshold))
            TN = sum((unknown_scores_softmax < threshold))

            p_account = len(known_scores_softmax)
            n_account = len(unknown_scores_softmax)
            account = p_account + n_account

            # print("ACC:   %3f, %3f, %3f, %3f, %d" % (TP / p_account, TN / n_account, (TP + TN) / account, threshold, jj))

            if (TP + TN) / account > best_acc:
                best_acc = (TP + TN) / account
                best_threshold = threshold
                best_ratio = jj
                p_acc = TP / p_account
                n_acc = TN / n_account

        print(
              "ACC2:", best_acc.item(), "最佳阈值:", best_threshold, "已知类比例:", 100 - best_ratio, "%", "P_ACC:", p_acc,
              "N_ACC:", n_acc)







        return test_correct.avg, auc_score_av, auc_score_softmax, oscr, f1_score

    def bcg_loss(self):

        minmargin_loss = 0
        maxmargin_loss = 0
        diversity_loss = 0

        maxmargin_loss_weight = 1/ (self.update_boundary_class.shape[0]*self.update_boundary_class.shape[1] )
        minmargin_loss_weight = 1/ ( self.update_boundary_class.shape[0]*self.update_boundary_class.shape[0]*self.update_boundary_class.shape[1])
        diversity_loss_weight = 1/ ((self.update_boundary_class.shape[1]*(self.update_boundary_class.shape[1]-1)/2*self.update_boundary_class.shape[0]))

        # margin loss
        for k in range(len(self.semantics)):
            dis = torch.abs(self.update_boundary_class - self.semantics[k])
            value, index = torch.sort(dis, dim=2)
            max_value = value[:, :, -self.minmargin:]
            min_value = value[:, :, :(dis.shape[2] - self.maxmargin)]



            minmargin_loss = torch.sum(((max_value - self.dissim) < 0) * (self.dissim - max_value)) + minmargin_loss
            maxmargin_loss = torch.sum(((min_value[k]) > self.sim) * (min_value[k])) + maxmargin_loss

        # diversity loss
        for i in range(self.update_boundary_class.shape[0]):
            for j in range(self.update_boundary_class.shape[1]):
                dis = torch.abs(self.update_boundary_class[i]*(self.update_boundary_class[i]<=1)*(self.update_boundary_class[i]>=0) - self.update_boundary_class[i][j]*(self.update_boundary_class[i][j]<=1)*(self.update_boundary_class[i][j]>=0))
                diversity_loss = torch.sum(((dis - self.dissim) < 0) * (self.dissim - dis)) + diversity_loss

        maxmargin_loss = maxmargin_loss*maxmargin_loss_weight
        minmargin_loss = minmargin_loss*minmargin_loss_weight
        diversity_loss = diversity_loss * diversity_loss_weight


        loss = maxmargin_loss+minmargin_loss+diversity_loss*self.diversity_weight

        return loss

    def generate_update_boundary_samples(self):

        num = (self.ibnum * self.batchsize) / self.update_boundary_nclass / self.nclass  # 每个边界类样本数量
        boundary_class = None
        self.syn_feature_all = None
        self.syn_label_all = None

        for k in range(len(self.update_boundary_class)):
            if boundary_class == None:
                boundary_class = self.update_boundary_class[k]
            else:
                boundary_class = torch.cat([boundary_class, self.update_boundary_class[k]])



        for k in range(len(boundary_class)):

            boundary_class_att = torch.ones(int(num), len(boundary_class[k])).to(self.device)
            boundary_class_att = boundary_class_att * boundary_class[k]
            # syn_label = (torch.ones(len(boundary_class_att)) * (self.nclass + k)).to(self.device)
            syn_label = (torch.ones(len(boundary_class_att)) * (k//self.nclass)).to(self.device)
            syn_noise = torch.FloatTensor(len(boundary_class_att), opt.nz).to(self.device)

            syn_noise.normal_(0, 1)

            # output = self.netG(Variable(syn_noise), Variable(boundary_class_att))
            output = self.netG(syn_noise, boundary_class_att)

            if self.syn_feature_all == None:
                self.syn_feature_all = output
                self.syn_label_all = syn_label
            else:
                self.syn_feature_all = torch.cat([self.syn_feature_all, output])
                self.syn_label_all = torch.cat([self.syn_label_all, syn_label])


        self.syn_feature_all = self.syn_feature_all.to(self.device)
        self.syn_label_all = self.syn_label_all.int().to(self.device)


        # self.input = torch.cat((self.input, syn_feature_all), 0)
        # self.label = torch.cat((self.label, syn_label_all), 0)
        self.syn_feature_all = self.syn_feature_all
        self.syn_label_all = self.syn_label_all.to(torch.int64)

        return self.syn_feature_all, self.syn_label_all

    def generate_boundary_samples(self):

        # num = (self.ibnum * self.batchsize) / self.boundary_nclass / self.nclass  # 每个边界类样本数量
        num = 40
        boundary_class = None
        self.syn_feature_all = None
        self.syn_label_all = None

        for k in range(len(self.boundary_class)):
            if boundary_class == None:
                boundary_class = self.boundary_class[k]
            else:
                boundary_class = torch.cat([boundary_class, self.boundary_class[k]])



        for k in range(len(boundary_class)):

            boundary_class_att = torch.ones(int(num), len(boundary_class[k])).to(self.device)
            boundary_class_att = boundary_class_att * boundary_class[k]
            syn_label = (torch.ones(len(boundary_class_att)) * (self.nclass + k)).to(self.device)
            syn_noise = torch.FloatTensor(len(boundary_class_att), opt.nz).to(self.device)

            syn_noise.normal_(0, 1)

            # output = self.netG(Variable(syn_noise), Variable(boundary_class_att))
            output = self.netG(syn_noise, boundary_class_att)

            if self.syn_feature_all == None:
                self.syn_feature_all = output
                self.syn_label_all = syn_label
            else:
                self.syn_feature_all = torch.cat([self.syn_feature_all, output])
                self.syn_label_all = torch.cat([self.syn_label_all, syn_label])

        check_att = {}
        for k in range(self.nclass):
            check_att[k] = self.data.train_att[k]
        for k in range(len(boundary_class)):
            check_att[k + self.nclass] = boundary_class[k]

        self.syn_feature_all = self.syn_feature_all.to(self.device)
        self.syn_label_all = self.syn_label_all.int().to(self.device)


        # self.input = torch.cat((self.input, syn_feature_all), 0)
        # self.label = torch.cat((self.label, syn_label_all), 0)
        self.syn_feature_all = self.syn_feature_all
        self.syn_label_all = self.syn_label_all.to(torch.int64)

        # print(self.syn_feature_all.shape)
        # exit()

        return self.syn_feature_all, self.syn_label_all

    def generate_anti_feature(self):
        # def generate_anti_feature(classes, attribute, num):

        num = 40#每个边界类样本数量
        if opt.tsne:
            num = 5  # 每个边界类样本数量
            # num = 20  # 每个边界类样本数量

        boundary_class = None
        syn_feature_all = None
        syn_label_all = None

        for k in range(len(self.update_boundary_class)):
            if boundary_class == None:
                boundary_class = self.update_boundary_class[k]
            else:
                boundary_class = torch.cat([boundary_class, self.update_boundary_class[k]])

        for k in range(len(boundary_class)):
            boundary_class_att = torch.ones(num, len(boundary_class[k])).to(self.device)
            boundary_class_att = boundary_class_att * boundary_class[k]
            # boundary_class_att = boundary_class_att * self.semantics[0]
            syn_label = (torch.ones(len(boundary_class_att)) * (self.nclass+self.nclass*self.boundary_nclass + k)).to(self.device).int()
            syn_noise = torch.FloatTensor(len(boundary_class_att), opt.nz).to(self.device)

            syn_noise.normal_(0, 1)
            with torch.no_grad():
                output = self.netG(Variable(syn_noise), Variable(boundary_class_att))

            if syn_feature_all == None:
                syn_feature_all = output
                syn_label_all = syn_label
            else:
                syn_feature_all = torch.cat([syn_feature_all, output])
                syn_label_all = torch.cat([syn_label_all, syn_label])

        syn_feature_all = syn_feature_all.cpu()
        syn_label_all = syn_label_all.cpu()

        print(syn_label_all)

        print(self.data.train_feature.shape)

        ####################################### TSNE
        if opt.tsne:
            self.data.train_feature = self.data.train_feature[:1200]
            self.data.train_label = self.data.train_label[:1200]
        ######################################### TSNE
        # print(self.syn_feature_all.shape)

        SAVE_PARA_PATH = './checkpoints/' + 'bcg_train' + '.pth'
        torch.save([self.data.train_feature, self.data.train_label], SAVE_PARA_PATH)

        SAVE_PARA_PATH = './checkpoints/' + 'bcg_test' + '.pth'
        torch.save([self.data.test_seen_feature, self.data.test_seen_label], SAVE_PARA_PATH)

        SAVE_PARA_PATH = './checkpoints/' + 'bcg_anti' + '.pth'
        torch.save([syn_feature_all, syn_label_all], SAVE_PARA_PATH)

        self.data.train_feature = torch.cat((self.data.train_feature.cpu(), syn_feature_all), 0)
        self.data.train_label = torch.cat((self.data.train_label.cpu(), syn_label_all), 0)
        # if self.syn_feature_all==None:
        #     self.syn_feature_all = syn_feature_all
        #     self.syn_label_all = syn_label_all
        # else:
        #     self.syn_feature_all = torch.cat((self.syn_feature_all.cpu(), syn_feature_all), 0)
        #     self.syn_label_all = torch.cat((self.syn_label_all.cpu(), syn_label_all), 0)
        print(self.data.train_feature.shape)
        # print(self.syn_feature_all.shape)

        self.data.train_feature = self.data.train_feature.cuda()
        self.data.train_label = self.data.train_label.cuda()

        if opt.tsne:
            ####################################### TSNE
            print('Begining......bcg')  # 时间会较长，所有处理完毕后给出finished提示
            tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
            result_2D = tsne_2D.fit_transform(self.data.train_feature.cpu())
            fig1 = plot_embedding_2D(result_2D, self.data.train_label.cpu(), 't-SNE')

            print('Finished......')
            exit()
            ####################################### TSNE



        # if self.boundary_class == None:
        #     self.boundary_class = self.update_boundary_class
        # else:
        #     self.boundary_class = torch.cat([self.boundary_class, self.update_boundary_class], dim=1)
        #
        # self.boundary_nclass = self.boundary_nclass + self.update_boundary_nclass

def plot_embedding_2D(data, label, title):
    fig = plt.figure(figsize=(5, 4), dpi=300)
    color_set = ["#e50000","#0343df","#15b01a","#9a0eea","#f97306","#fac205"] #设置已知类颜色
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], s=20, c=plt.cm.Set1(int(label[i])), marker='o', alpha=0.5)
        if int(label[i])<len(color_set):
            plt.scatter(data[i, 0], data[i, 1], s=20, c=color_set[int(label[i])], marker='o', alpha=1)
        else:
            plt.scatter(data[i, 0], data[i, 1], s=20, c=[ "#929591"], marker='o', alpha=0.5)
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(int(label[i])),
        #          fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./' + 'TSNE' + "_" + "bcg" + '.jpg', dpi=300)
    plt.close()
    # plt.title(title)
    return fig


def compute_oscr(x1, x2, pred, labels):

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w


    return OSCR


class OpenAUCLoss(nn.Module):
    def __init__(self, num_classes, **options):
        super().__init__()
        self.loss_close = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.alpha = 2
        self.lambd = 0.1  #0.1~0.6

    def loss_open(self, outputs_real, labels, outputs_fake, mask):
        pred_neg, predictions = outputs_real.max(axis=1)
        if not opt.mmixup:
            hit = (predictions.data == labels.data).float() * mask
        else:
            hit = (predictions.data == predictions.data).float() * mask
        pred_pos, _ = outputs_fake.max(axis=1)

        return (hit * (pred_pos - pred_neg + 1) ** 2).sum(), hit.sum()

    def forward(self, logits, labels, f_post, manifolds):
        loss = self.loss_close(logits, labels)

        half_lenth = manifolds.size(0) // 2
        if 2 * half_lenth != manifolds.size(0):
            return logits, loss
        laterhalf_manifolds = manifolds[half_lenth:]
        laterhalf_labels = labels[half_lenth:]

        shuffle_ix = np.random.permutation(np.arange(half_lenth))
        shuffle_ix = torch.tensor(list(shuffle_ix)).int().cuda()
        shuffle_laterhalf_labels = torch.index_select(laterhalf_labels, 0, shuffle_ix)
        shuffle_laterhalf_manifolds = torch.index_select(laterhalf_manifolds, 0, shuffle_ix)
        mask = (shuffle_laterhalf_labels.data != laterhalf_labels.data).float()

        lam = np.random.beta(self.alpha, self.alpha)
        mixup_manifolds = lam * laterhalf_manifolds + (1 - lam) * shuffle_laterhalf_manifolds
        outputs_fake = f_post(mixup_manifolds)

        loss_fake, n = self.loss_open(logits[:half_lenth], labels[:half_lenth], outputs_fake, mask)
        loss = loss + loss_fake / n * self.lambd if n > 0 else loss

        return logits, loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_hyperparameters():

    global opt
    global data

    opt.dataset = "cifar10"
    # opt.dataset = "cifar+50"
    # opt.dataset = "mnist"
    # opt.dataset = "svhn"
    opt.check = False


    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    opt.device = torch.device("cuda:"+str(int(opt.gpu)) if torch.cuda.is_available() else "cpu")


    if opt.dataset == "cifar10" or opt.dataset == "cifar+50":

        opt.openauc = True
        opt.openauc = False
        opt.mmixup = True
        opt.mmixup = False

        opt.tsne = True
        opt.tsne = False

        opt.f1 = True
        opt.f1 = False

        if True:
            if opt.split == None:
                opt.split = 0
                opt.split = 1
                opt.split = 2
                opt.split = 3
                # opt.split = 4
            opt.data = data_preprocess.DATA_PREPROCESS(opt)
            opt.nz = 312
            opt.ngh = 4096
            opt.ndh = 4096
            opt.attSize = opt.data.train_att.shape[1]
            opt.resSize = 2048


            opt.epoch_cgan = 39 #sota
            # opt.epoch_cgan = 5  # sota


            opt.batchsize_detector = 32
            opt.batchsize_detector_2 = 4096
            # opt.batchsize_detector_2 = 32
            # opt.epoch_detector = 1
            opt.epoch_detector = 30
            # opt.epoch_detector = 50
            opt.epoch_detector_initial = 10
            # opt.epoch_detector_initial = 1
            opt.lr_detector = 0.001


            opt.lr_bcg_initial = 10
            opt.lr_bcg = 10
            opt.epoch_bcg_initial = 100
            opt.epoch_bcg_initial_milestone = [50,70,90]
            opt.epoch_bcg = 30
            # opt.epoch_bcg = 100
            opt.epoch_bcg_milestone = [20,30,40]
            # opt.epoch_bcg_milestone = [50, 70, 90]


            # opt.big_iter = 5
            opt.big_iter = 1
            opt.big_iter = opt.big_iter+1


            opt.update_boundary_nclass = 50
            # opt.update_boundary_nclass = 5
            # opt.ibnum = 40
            opt.ibnum = 40
            # opt.ibnum = 4000

            opt.maxmargin = 20
            if opt.tsne:
                opt.maxmargin = 2
            # opt.maxmargin = 5
            opt.minmargin = 1
            # opt.minmargin = 20
            # opt.dissim = 0.8
            opt.dissim = 0.8
            # opt.sim = 0.1
            opt.sim = 0.01
            opt.diversity_weight = 0.01
            opt.hard_weight = 0.0
            # opt.hard_weight = 0.5
            # opt.hard_weight = 0.0
            # opt.hard_weight = 0.2





            # R 消融
            # # opt.maxmargin = 50
            # opt.maxmargin = 10
            # opt.minmargin = 1
            # opt.dissim = 5
            # # opt.epoch_bcg_initial = 1000
            # opt.hard_weight = 0
            # opt.lr_bcg_initial = 100
            # opt.lr_bcg = 100
            # opt.epoch_detector = 15
            #
            # opt.maxmargin = 5
            # # opt.maxmargin = 5
            # opt.minmargin = 1
            #
            # opt.lr_bcg_initial = 100
            # opt.lr_bcg = 100


    # if opt.dataset == "mnist":
    #
    #     opt.openauc = True
    #     opt.openauc = False
    #     opt.mmixup = True
    #     opt.mmixup = False
    #
    #     opt.tsne = True
    #     opt.tsne = False
    #
    #     opt.f1 = True
    #     opt.f1 = False
    #
    #     if True:
    #         if opt.split == None:
    #             opt.split = 0
    #             # opt.split = 1
    #             # opt.split = 2
    #             # opt.split = 3
    #             # opt.split = 4
    #         opt.data = util.DATA_LOADER(opt)
    #         opt.nz = 312
    #         opt.ngh = 3200
    #         opt.ndh = 3200
    #         opt.attSize = opt.data.train_att.shape[1]
    #         opt.resSize = 1600
    #
    #
    #
    #         opt.epoch_cgan = 5  # sota
    #
    #
    #         opt.batchsize_detector = 32
    #         opt.batchsize_detector_2 = 4096
    #         opt.batchsize_detector_2 = 32
    #         opt.epoch_detector = 20
    #         # opt.epoch_detector = 50
    #         opt.epoch_detector_initial = 10
    #         # opt.epoch_detector_initial = 1
    #         opt.lr_detector = 0.001
    #
    #
    #         opt.lr_bcg_initial = 10
    #         opt.lr_bcg = 10
    #         opt.epoch_bcg_initial = 100
    #         opt.epoch_bcg_initial_milestone = [50,70,90]
    #         opt.epoch_bcg = 30
    #         # opt.epoch_bcg = 100
    #         opt.epoch_bcg_milestone = [20,30,40]
    #         # opt.epoch_bcg_milestone = [50, 70, 90]
    #
    #
    #         # opt.big_iter = 5
    #         opt.big_iter = 1
    #         opt.big_iter = opt.big_iter+1
    #
    #
    #         opt.update_boundary_nclass = 10
    #         # opt.update_boundary_nclass = 20
    #         # opt.update_boundary_nclass = 5
    #         # opt.ibnum = 40
    #         opt.ibnum = 40
    #         # opt.ibnum = 4000
    #
    #         opt.maxmargin = 5
    #         if opt.tsne:
    #             opt.maxmargin = 2
    #         opt.minmargin = 1
    #         # opt.minmargin = 20
    #         # opt.dissim = 0.8
    #         opt.dissim = 0.8
    #         # opt.sim = 0.1
    #         opt.sim = 0.01
    #         opt.diversity_weight = 0.01
    #         opt.hard_weight = 0.1
    #         # opt.hard_weight = 1
    #         # opt.hard_weight = 0.5
    #         # opt.hard_weight = 0.0
    #         # opt.hard_weight = 0.2
    #
    #
    #
    #
    #         # R 消融
    #         # # opt.maxmargin = 50
    #         # opt.maxmargin = 10
    #         # opt.minmargin = 1
    #         # opt.dissim = 5
    #         # # opt.epoch_bcg_initial = 1000
    #         # opt.hard_weight = 0
    #         # opt.lr_bcg_initial = 100
    #         # opt.lr_bcg = 100
    #         # opt.epoch_detector = 15
    #         #
    #         # opt.maxmargin = 5
    #         # # opt.maxmargin = 5
    #         # opt.minmargin = 1
    #         #
    #         # opt.lr_bcg_initial = 100
    #         # opt.lr_bcg = 100

    # if opt.dataset == "svhn":
    #
    #     opt.openauc = True
    #     opt.openauc = False
    #     opt.mmixup = True
    #     opt.mmixup = False
    #
    #     opt.tsne = True
    #     opt.tsne = False
    #
    #     opt.f1 = True
    #     opt.f1 = False
    #
    #     if True:
    #         if opt.split == None:
    #             opt.split = 0
    #             opt.split = 1
    #             opt.split = 2
    #             opt.split = 3
    #             opt.split = 4
    #         opt.data = util.DATA_LOADER(opt)
    #         opt.nz = 312
    #         opt.ngh = 4096
    #         opt.ndh = 4096
    #         opt.attSize = opt.data.train_att.shape[1]
    #         opt.resSize = 2048
    #
    #
    #
    #         opt.epoch_cgan = 39  # sota
    #         opt.epoch_cgan = 5  # sota
    #
    #
    #         opt.batchsize_detector = 32
    #         # opt.batchsize_detector_2 = 4096
    #         opt.batchsize_detector_2 = 32
    #         opt.epoch_detector = 5
    #         # opt.epoch_detector = 50
    #         opt.epoch_detector_initial = 10
    #         # opt.epoch_detector_initial = 1
    #         opt.lr_detector = 0.001
    #
    #
    #         opt.lr_bcg_initial = 10
    #         opt.lr_bcg = 10
    #         opt.epoch_bcg_initial = 100
    #         opt.epoch_bcg_initial_milestone = [50,70,90]
    #         opt.epoch_bcg = 30
    #         # opt.epoch_bcg = 100
    #         opt.epoch_bcg_milestone = [20,30,40]
    #         # opt.epoch_bcg_milestone = [50, 70, 90]
    #
    #
    #         # opt.big_iter = 5
    #         opt.big_iter = 1
    #         opt.big_iter = opt.big_iter+1
    #
    #
    #         # opt.update_boundary_nclass = 10
    #         # opt.update_boundary_nclass = 20
    #         opt.update_boundary_nclass = 5
    #         # opt.ibnum = 40
    #         opt.ibnum = 40
    #         # opt.ibnum = 400
    #
    #         opt.maxmargin = 2
    #         if opt.tsne:
    #             opt.maxmargin = 2
    #         opt.minmargin = 1
    #         # opt.maxmargin = 40
    #         # opt.dissim = 0.8
    #         opt.dissim = 0.1
    #         # opt.sim = 0.1
    #         opt.sim = 0.01
    #         opt.diversity_weight = 0.01
    #         opt.hard_weight = 0.1
    #         # opt.hard_weight = 1
    #         # opt.hard_weight = 0.5
    #         # opt.hard_weight = 0.0
    #         # opt.hard_weight = 0.2
    #
    #
    #
    #
    #         # R 消融
    #         # # opt.maxmargin = 50
    #         # opt.maxmargin = 10
    #         # opt.minmargin = 1
    #         # opt.dissim = 5
    #         # # opt.epoch_bcg_initial = 1000
    #         # opt.hard_weight = 0
    #         # opt.lr_bcg_initial = 100
    #         # opt.lr_bcg = 100
    #         # opt.epoch_detector = 15
    #         #
    #         # opt.maxmargin = 5
    #         # # opt.maxmargin = 5
    #         # opt.minmargin = 1
    #         #
    #         # opt.lr_bcg_initial = 100
    #         # opt.lr_bcg = 100








if __name__ == '__main__':

    set_hyperparameters()
    obj = BCG(opt.data)
    print("finished!")
    exit()