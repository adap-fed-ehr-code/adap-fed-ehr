# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models_code
import numpy as np

from torch.autograd import grad

import time
from sklearn import metrics

class Trainer(object):
    def __init__(self, option, model_config, logger, num_classes=2, num_domains=2):
        self.option = option
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.logger = logger
        self._build_model(model_config, num_classes, num_domains)
        self._set_optimizer()

    def _build_model(self, model_config, num_classes=2, num_domains=2):
        self.net = models_code.FS_C(self.option, model_config, num_classes, num_domains)
        self.loss = nn.NLLLoss()

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()

    def _set_optimizer(self):
        self.optim_g1 = optim.Adam(self.net.G1.parameters(), lr=self.option.lr, weight_decay=0.0)
        self.optim_g2 = optim.Adam(self.net.G2.parameters(), lr=self.option.lr, weight_decay=0.0)
        self.optim_m_s = optim.Adam(self.net.M_s.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_ms_s = []
        for idx in range(len(self.net.Ms_s)):
            self.optim_ms_s.append(optim.Adam(self.net.Ms_s[idx].parameters(), lr=self.option.lr_2,
                                              weight_decay=self.option.weight_decay_2))
        self.optim_ms_p = []
        for idx in range(len(self.net.Ms_p)):
            self.optim_ms_p.append(
                optim.Adam(self.net.Ms_p[idx].parameters(), lr=self.option.lr_2, weight_decay=self.option.weight_decay_2))

    def _reset_grad(self):
        self.optim_g1.zero_grad()
        self.optim_g2.zero_grad()
        self.optim_m_s.zero_grad()
        for idx in range(len(self.optim_ms_s)):
            self.optim_ms_s[idx].zero_grad()
        for idx in range(len(self.optim_ms_p)):
            self.optim_ms_p[idx].zero_grad()

    def _group_step(self, optim_list):
        for i in range(len(optim_list)):
            optim_list[i].step()

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            for idx in range(len(self.net.Ms_s)):
                self.net.Ms_s[idx].train()
            for idx in range(len(self.net.Ms_p)):
                self.net.Ms_p[idx].train()
        else:
            self.net.eval()
            for idx in range(len(self.net.Ms_s)):
                self.net.Ms_s[idx].eval()
            for idx in range(len(self.net.Ms_p)):
                self.net.Ms_p[idx].eval()

    # @profile
    def _train_step(self, step, data_loaders):
        self._mode_setting(is_train=True)
        time_sum = 0.0
        for (d_idx, data_loader) in enumerate(data_loaders):
            for i, (x, y, d) in enumerate(data_loader):
                start_time = time.time()
                x = self._get_variable(x)
                y = self._get_variable(y)

                self._reset_grad()

                feat_s, feat_d, all_y_pred_s_s,all_y_pred_s_i = self.net.forward_s_d(x,d_idx)
                loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                loss_pred_y_s_i = self.loss(all_y_pred_s_i, y[:, 0])
                loss = 1.0 * loss_pred_y_s_i + 1.0 * loss_pred_y_s_s
                loss.backward()
                self._group_step([self.optim_m_s, self.optim_ms_s[d_idx]])
                self._reset_grad()

                # feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i = self.net.forward_s_d(x, d_idx)
                # # #loss_ortho = (torch.tensordot(torch.transpose(feat_s, 1, 0), feat_d, 1)).mean()
                # loss_dicre = ((all_y_pred_s_s - all_y_pred_s_i) ** 2).mean()
                # loss = self.option.lamb_discre * loss_dicre
                # #
                # loss.backward()
                # self._group_step([self.optim_g2])
                # self._reset_grad()
                #
                # feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i = self.net.forward_s_d(x, d_idx)
                # loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                # g = grad(loss_pred_y_s_s,self.net.M_s.fc1.weight)[0]
                # loss = 0.0 * g.mean()
                # loss.requires_grad_()
                # loss.backward()
                # self._group_step([self.optim_g2])
                # self._reset_grad()

                feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i, all_y_pred_p, _ = self.net.forward_d(x,d_idx)
                loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                loss_pred_y_p = self.loss(all_y_pred_p, y[:, 0])
                loss_discre = ((all_y_pred_s_s - all_y_pred_s_i) ** 2).mean()
                loss_grad = grad(loss_pred_y_s_s, self.net.M_s.parameters(), create_graph=True)[0].pow(2).mean()
                loss_stable = loss_discre + loss_grad

                loss = 1.0 * loss_pred_y_s_s + loss_pred_y_p + self.option.lamb_stable * loss_stable + self.option.lamb_sparse * self.net.regularization()

                # print(torch.norm(g[0],2))
                loss.backward()
                self._group_step([self.optim_g1, self.optim_g2, self.optim_ms_p[d_idx]])
                self._reset_grad()
                time_sum += time.time() - start_time

        g1_mask = self.net.G1.sample_mask()[0, :]
        print(g1_mask.mean(),g1_mask.std())
        g2_mask = self.net.G2.sample_mask()[0, :]
        print(g2_mask.mean(), g2_mask.std())

    # @profile
    def _valid(self, data_loaders, step, valid_flag):
        self._mode_setting(is_train=False)
        all_auc_s,all_auc_i,all_auc_p,all_auc_r = [],[],[],[]
        msg = "Step: %d" % step
        self.logger.info(msg)

        if valid_flag == 0:
            valid_str = "Train"
        elif valid_flag == 1:
            valid_str = "Valid"
        else:
            valid_str = "test"
        msg = "[VALID-%s]  (epoch %d)" % (valid_str, step)
        self.logger.info(msg)
        all_loss_pred_y_s_i = []
        all_loss_pred_y_s_s = []
        all_loss_pred_y_p = []
        all_loss_pred_y_r = []

        all_pred_y = []
        all_true_label = []
        for (d_idx,data_loader) in enumerate(data_loaders):
            all_pred_y_s_i = []
            all_pred_y_s_s = []
            all_pred_y_p = []
            all_pred_y_r = []
            new_feat_s = []
            new_labels = []

            x = self._get_variable(data_loader.dataset.X)
            y = self._get_variable(data_loader.dataset.y[:, None])
            d = self._get_variable(data_loader.dataset.d[:, None])

            feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i, all_y_pred_p, all_y_pred_r = self.net.forward_d_valid(x,d_idx)

            loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
            loss_pred_y_s_i = self.loss(all_y_pred_s_i, y[:, 0])
            loss_pred_y_p = self.loss(all_y_pred_p, y[:, 0])
            loss_pred_y_r = self.loss(all_y_pred_r, y[:, 0])

            all_loss_pred_y_s_i.append(loss_pred_y_s_i.data.item())
            all_loss_pred_y_s_s.append(loss_pred_y_s_s.data.item())
            all_loss_pred_y_p.append(loss_pred_y_p.data.item())
            all_loss_pred_y_r.append(loss_pred_y_r.data.item())

            all_pred_y_s_i.append(self._get_numpy_from_variable(all_y_pred_s_i)[:, 1])
            all_pred_y_s_s.append(self._get_numpy_from_variable(all_y_pred_s_s)[:, 1])
            all_pred_y_p.append(self._get_numpy_from_variable(all_y_pred_p)[:, 1])
            all_pred_y_r.append(self._get_numpy_from_variable(all_y_pred_r)[:, 1])

            new_feat_s.append(self._get_numpy_from_variable(feat_s))
            new_labels.append(self._get_numpy_from_variable(y))

            all_pred_y_s_i = np.concatenate(all_pred_y_s_i, 0)
            all_pred_y_s_s = np.concatenate(all_pred_y_s_s, 0)
            all_pred_y_p = np.concatenate(all_pred_y_p, 0)
            all_pred_y_r = np.concatenate(all_pred_y_r, 0)
            new_labels = np.concatenate(new_labels, 0)
            auc_i, auc_s = metrics.roc_auc_score(new_labels, all_pred_y_s_i),metrics.roc_auc_score(new_labels, all_pred_y_s_s)
            auc_p, auc_r = metrics.roc_auc_score(new_labels, all_pred_y_p),metrics.roc_auc_score(new_labels, all_pred_y_r)
            all_auc_s.append(auc_s)
            all_auc_i.append(auc_i)
            #if auc_p > auc_s:
            #all_auc_p.append(auc_p)
            #else:
            all_auc_p.append(auc_p)
            all_auc_r.append(auc_r)
            all_pred_y.extend(all_pred_y_p)
            all_true_label.extend(new_labels)

        micro_auc = metrics.roc_auc_score(all_true_label, np.array(all_pred_y))
        msg = "Average AUCROC-I: %.3f  S: %.3f P: %.3f R: %.3f" % (np.array(all_auc_i).mean(), np.array(all_auc_s).mean(), np.array(all_auc_p).mean(), np.array(all_auc_r).mean())
        self.logger.info(msg)
        return np.array(all_auc_i).mean(), np.array(all_auc_s).mean(), np.array(all_auc_p).mean(),all_auc_s,all_auc_p, micro_auc

    def val_shap(self, train_X, test_X, test_d, s_idx, dir_name):
        self._mode_setting(is_train=False)
        for d_idx in range(test_d.max() + 1):
            test_X_d = test_X[test_d == d_idx, :]
            # pred = self.net.predict_numpy_d(test_X_d, d_idx)
            f = lambda x: self.net.predict_numpy_d(x, d_idx)[:, 1]
            med = train_X.mean(axis=0).reshape((1, train_X.shape[1]))
            # med = X_train.mean().values.reshape((1,X_train.shape[1]))
            explainer = shap.KernelExplainer(f, med)
            shap_values = explainer.shap_values(test_X_d, nsamples=1000)
            np.save('{}/shap_{}_{}'.format(dir_name, s_idx, d_idx), shap_values)
            np.save('{}/X_{}_{}'.format(dir_name, s_idx, d_idx), test_X_d)

        return

            # @profile
    def train(self, train_loaders, val_loaders=None, test_loaders=None):
        self._mode_setting(is_train=True)
        start_epoch = 0
        best_valid_roc_i, best_valid_roc_s, best_valid_roc_p = 0.0,0.0,0.0
        best_test_roc_i, best_test_roc_s, best_test_roc_p = 0.0,0.0,0.0
        best_test_micro_auc = 0.0
        best_test_auc_array = None
        best_step_i,best_step_s, best_step_p,stop_cnt = -1,-1,-1, 0
        best_g1_mask, best_g2_mask = None, None
        for step in range(start_epoch, self.option.change_step):
            self._train_step(step, train_loaders)
            self._valid(train_loaders, step, 0)
            valid_roc_i, valid_roc_s, valid_roc_p, all_valid_roc_s, all_valid_roc_p, valid_micro_auc = self._valid(val_loaders, step ,1)
            test_roc_i, test_roc_s, test_roc_p, all_test_roc_s, all_test_roc_p, test_micro_auc = self._valid(test_loaders, step, 2)
            opti_valid_roc, opti_test_roc = [],[]
            for idx in range(len(all_valid_roc_p)):
                if all_valid_roc_p[idx] > all_valid_roc_s[idx]:
                    opti_valid_roc.append(all_valid_roc_p[idx])
                    opti_test_roc.append(all_test_roc_p[idx])
                else:
                    opti_valid_roc.append(all_valid_roc_s[idx])
                    opti_test_roc.append(all_test_roc_s[idx])
            if valid_roc_i > best_valid_roc_i:
                best_valid_roc_i = valid_roc_i
                best_test_roc_i = test_roc_i
                best_step_i = step
            if valid_roc_s > best_valid_roc_s:
                best_valid_roc_s = valid_roc_s
                best_test_roc_s = test_roc_s
                best_step_s = step
            if np.array(opti_valid_roc).mean() > best_valid_roc_p:
                best_g1_mask = self.net.G1.sample_mask()[0, :].detach().numpy()
                best_g2_mask = self.net.G2.sample_mask()[0, :].detach().numpy()
                best_valid_roc_p = np.array(opti_valid_roc).mean()
                best_test_roc_p = np.array(opti_test_roc).mean()
                best_test_micro_auc = test_micro_auc
                best_test_auc_array = opti_test_roc
                best_step_p = step
                stop_cnt = 0
            else:
                stop_cnt += 1
            msg = "[VALID-BEST-I]: %.3f, %.3f, %d" % (best_valid_roc_i, best_test_roc_i,best_step_i)
            self.logger.info(msg)
            msg = "[VALID-BEST-S]: %.3f, %.3f, %d" % (best_valid_roc_s, best_test_roc_s,best_step_s)
            self.logger.info(msg)
            msg = "[VALID-BEST-P]: %.3f, %.3f, %d, %d" % (best_valid_roc_p, best_test_roc_p,best_step_p,stop_cnt)
            self.logger.info(msg)
            if stop_cnt >= 5:
                # self.logger.info("G1")
                # for idx in range(len(best_g1_mask)):
                #     self.logger.info(best_g1_mask[idx])
                # self.logger.info("G2")
                # for idx in range(len(best_g2_mask)):
                #     self.logger.info(best_g2_mask[idx])
                break
        return best_test_roc_i, best_test_roc_s, best_test_roc_p, best_test_micro_auc, best_test_auc_array

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

    def _get_numpy_from_variable(self, input):
        if self.option.cuda:
            return input.data.cpu().numpy()
        return input.data.numpy()



class Trainer_Global(object):
    def __init__(self, option, model_config, logger, num_classes=2, num_domains=2):
        self.option = option
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.logger = logger
        self._build_model(model_config, num_classes, num_domains)
        self._set_optimizer()

    def _build_model(self, model_config, num_classes=2, num_domains=2):
        self.net = models_code.FS_C(self.option, model_config, num_classes, num_domains)
        self.loss = nn.NLLLoss()

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()

    def _set_optimizer(self):
        self.optim_g1 = optim.Adam(self.net.G1.parameters(), lr=self.option.lr, weight_decay=0.0)
        self.optim_g2 = optim.Adam(self.net.G2.parameters(), lr=self.option.lr, weight_decay=0.0)
        self.optim_m_s = optim.Adam(self.net.M_s.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_ms_s = []
        for idx in range(len(self.net.Ms_s)):
            self.optim_ms_s.append(optim.Adam(self.net.Ms_s[idx].parameters(), lr=self.option.lr_2,
                                              weight_decay=self.option.weight_decay_2))
        self.optim_ms_p = []
        for idx in range(len(self.net.Ms_p)):
            self.optim_ms_p.append(
                optim.Adam(self.net.Ms_p[idx].parameters(), lr=self.option.lr_2, weight_decay=self.option.weight_decay_2))

    def _reset_grad(self):
        self.optim_g1.zero_grad()
        self.optim_g2.zero_grad()
        self.optim_m_s.zero_grad()
        for idx in range(len(self.optim_ms_s)):
            self.optim_ms_s[idx].zero_grad()
        for idx in range(len(self.optim_ms_p)):
            self.optim_ms_p[idx].zero_grad()

    def _group_step(self, optim_list):
        for i in range(len(optim_list)):
            optim_list[i].step()

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            for idx in range(len(self.net.Ms_s)):
                self.net.Ms_s[idx].train()
            for idx in range(len(self.net.Ms_p)):
                self.net.Ms_p[idx].train()
        else:
            self.net.eval()
            for idx in range(len(self.net.Ms_s)):
                self.net.Ms_s[idx].eval()
            for idx in range(len(self.net.Ms_p)):
                self.net.Ms_p[idx].eval()

    # @profile
    def _train_step(self, step, data_loaders):
        self._mode_setting(is_train=True)
        time_sum = 0.0
        for (d_idx, data_loader) in enumerate(data_loaders):
            for i, (x, y, d) in enumerate(data_loader):
                start_time = time.time()
                x = self._get_variable(x)
                y = self._get_variable(y)

                self._reset_grad()

                feat_s, feat_d, all_y_pred_s_s,all_y_pred_s_i = self.net.forward_s_d(x,d_idx)
                loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                loss_pred_y_s_i = self.loss(all_y_pred_s_i, y[:, 0])
                loss = 1.0 * loss_pred_y_s_i + 1.0 * loss_pred_y_s_s
                loss.backward()
                self._group_step([self.optim_m_s, self.optim_ms_s[d_idx]])
                self._reset_grad()

                # feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i = self.net.forward_s_d(x, d_idx)
                # # #loss_ortho = (torch.tensordot(torch.transpose(feat_s, 1, 0), feat_d, 1)).mean()
                # loss_dicre = ((all_y_pred_s_s - all_y_pred_s_i) ** 2).mean()
                # loss = self.option.lamb_discre * loss_dicre
                # #
                # loss.backward()
                # self._group_step([self.optim_g2])
                # self._reset_grad()
                #
                # feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i = self.net.forward_s_d(x, d_idx)
                # loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                # g = grad(loss_pred_y_s_s,self.net.M_s.fc1.weight)[0]
                # loss = 0.0 * g.mean()
                # loss.requires_grad_()
                # loss.backward()
                # self._group_step([self.optim_g2])
                # self._reset_grad()

                feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i, all_y_pred_p, _ = self.net.forward_d(x,d_idx)
                loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
                loss_pred_y_p = self.loss(all_y_pred_p, y[:, 0])

                loss = 1.0 * loss_pred_y_s_s + loss_pred_y_p + self.option.lamb_sparse * self.net.regularization()

                # print(torch.norm(g[0],2))
                loss.backward()
                self._group_step([self.optim_g1, self.optim_g2, self.optim_ms_p[d_idx]])
                self._reset_grad()
                time_sum += time.time() - start_time

        g1_mask = self.net.G1.sample_mask()[0, :]
        print(g1_mask.mean(),g1_mask.std())
        g2_mask = self.net.G2.sample_mask()[0, :]
        print(g2_mask.mean(), g2_mask.std())

    # @profile
    def _valid(self, data_loaders, step, valid_flag):
        self._mode_setting(is_train=False)
        all_auc_s,all_auc_i,all_auc_p,all_auc_r = [],[],[],[]
        msg = "Step: %d" % step
        self.logger.info(msg)

        if valid_flag == 0:
            valid_str = "Train"
        elif valid_flag == 1:
            valid_str = "Valid"
        else:
            valid_str = "test"
        msg = "[VALID-%s]  (epoch %d)" % (valid_str, step)
        self.logger.info(msg)
        all_loss_pred_y_s_i = []
        all_loss_pred_y_s_s = []
        all_loss_pred_y_p = []
        all_loss_pred_y_r = []

        all_pred_y = []
        all_true_label = []
        for (d_idx,data_loader) in enumerate(data_loaders):
            all_pred_y_s_i = []
            all_pred_y_s_s = []
            all_pred_y_p = []
            all_pred_y_r = []
            new_feat_s = []
            new_labels = []

            x = self._get_variable(data_loader.dataset.X)
            y = self._get_variable(data_loader.dataset.y[:, None])
            d = self._get_variable(data_loader.dataset.d[:, None])

            feat_s, feat_d, all_y_pred_s_s, all_y_pred_s_i, all_y_pred_p, all_y_pred_r = self.net.forward_d_valid(x,d_idx)

            loss_pred_y_s_s = self.loss(all_y_pred_s_s, y[:, 0])
            loss_pred_y_s_i = self.loss(all_y_pred_s_i, y[:, 0])
            loss_pred_y_p = self.loss(all_y_pred_p, y[:, 0])
            loss_pred_y_r = self.loss(all_y_pred_r, y[:, 0])

            all_loss_pred_y_s_i.append(loss_pred_y_s_i.data.item())
            all_loss_pred_y_s_s.append(loss_pred_y_s_s.data.item())
            all_loss_pred_y_p.append(loss_pred_y_p.data.item())
            all_loss_pred_y_r.append(loss_pred_y_r.data.item())

            all_pred_y_s_i.append(self._get_numpy_from_variable(all_y_pred_s_i)[:, 1])
            all_pred_y_s_s.append(self._get_numpy_from_variable(all_y_pred_s_s)[:, 1])
            all_pred_y_p.append(self._get_numpy_from_variable(all_y_pred_p)[:, 1])
            all_pred_y_r.append(self._get_numpy_from_variable(all_y_pred_r)[:, 1])

            new_feat_s.append(self._get_numpy_from_variable(feat_s))
            new_labels.append(self._get_numpy_from_variable(y))

            all_pred_y_s_i = np.concatenate(all_pred_y_s_i, 0)
            all_pred_y_s_s = np.concatenate(all_pred_y_s_s, 0)
            all_pred_y_p = np.concatenate(all_pred_y_p, 0)
            all_pred_y_r = np.concatenate(all_pred_y_r, 0)
            new_labels = np.concatenate(new_labels, 0)
            auc_i, auc_s = metrics.roc_auc_score(new_labels, all_pred_y_s_i),metrics.roc_auc_score(new_labels, all_pred_y_s_s)
            auc_p, auc_r = metrics.roc_auc_score(new_labels, all_pred_y_p),metrics.roc_auc_score(new_labels, all_pred_y_r)
            all_auc_s.append(auc_s)
            all_auc_i.append(auc_i)
            #if auc_p > auc_s:
            #all_auc_p.append(auc_p)
            #else:
            all_auc_p.append(auc_p)
            all_auc_r.append(auc_r)
            all_pred_y.extend(all_pred_y_p)
            all_true_label.extend(new_labels)

        micro_auc = metrics.roc_auc_score(all_true_label, np.array(all_pred_y))
        msg = "Average AUCROC-I: %.3f  S: %.3f P: %.3f R: %.3f" % (np.array(all_auc_i).mean(), np.array(all_auc_s).mean(), np.array(all_auc_p).mean(), np.array(all_auc_r).mean())
        self.logger.info(msg)
        return np.array(all_auc_i).mean(), np.array(all_auc_s).mean(), np.array(all_auc_p).mean(),all_auc_s,all_auc_p, micro_auc

    def val_shap(self, train_X, test_X, test_d, s_idx, dir_name):
        self._mode_setting(is_train=False)
        for d_idx in range(test_d.max() + 1):
            test_X_d = test_X[test_d == d_idx, :]
            # pred = self.net.predict_numpy_d(test_X_d, d_idx)
            f = lambda x: self.net.predict_numpy_d(x, d_idx)[:, 1]
            med = train_X.mean(axis=0).reshape((1, train_X.shape[1]))
            # med = X_train.mean().values.reshape((1,X_train.shape[1]))
            explainer = shap.KernelExplainer(f, med)
            shap_values = explainer.shap_values(test_X_d, nsamples=1000)
            np.save('{}/shap_{}_{}'.format(dir_name, s_idx, d_idx), shap_values)
            np.save('{}/X_{}_{}'.format(dir_name, s_idx, d_idx), test_X_d)

        return

            # @profile
    def train(self, train_loaders, val_loaders=None, test_loaders=None):
        self._mode_setting(is_train=True)
        start_epoch = 0
        best_valid_roc_i, best_valid_roc_s, best_valid_roc_p = 0.0,0.0,0.0
        best_test_roc_i, best_test_roc_s, best_test_roc_p = 0.0,0.0,0.0
        best_test_micro_auc = 0.0
        best_test_auc_array = None
        best_step_i,best_step_s, best_step_p,stop_cnt = -1,-1,-1, 0
        best_g1_mask, best_g2_mask = None, None
        for step in range(start_epoch, self.option.change_step):
            self._train_step(step, train_loaders)
            self._valid(train_loaders, step, 0)
            valid_roc_i, valid_roc_s, valid_roc_p, all_valid_roc_s, all_valid_roc_p, valid_micro_auc = self._valid(val_loaders, step ,1)
            test_roc_i, test_roc_s, test_roc_p, all_test_roc_s, all_test_roc_p, test_micro_auc = self._valid(test_loaders, step, 2)
            opti_valid_roc, opti_test_roc = [],[]
            for idx in range(len(all_valid_roc_p)):
                if all_valid_roc_p[idx] > all_valid_roc_s[idx]:
                    opti_valid_roc.append(all_valid_roc_p[idx])
                    opti_test_roc.append(all_test_roc_p[idx])
                else:
                    opti_valid_roc.append(all_valid_roc_s[idx])
                    opti_test_roc.append(all_test_roc_s[idx])
            if valid_roc_i > best_valid_roc_i:
                best_valid_roc_i = valid_roc_i
                best_test_roc_i = test_roc_i
                best_step_i = step
            if valid_roc_s > best_valid_roc_s:
                best_valid_roc_s = valid_roc_s
                best_test_roc_s = test_roc_s
                best_step_s = step
            if np.array(opti_valid_roc).mean() > best_valid_roc_p:
                best_g1_mask = self.net.G1.sample_mask()[0, :].detach().numpy()
                best_g2_mask = self.net.G2.sample_mask()[0, :].detach().numpy()
                best_valid_roc_p = np.array(opti_valid_roc).mean()
                best_test_roc_p = np.array(opti_test_roc).mean()
                best_test_micro_auc = test_micro_auc
                best_test_auc_array = opti_test_roc
                best_step_p = step
                stop_cnt = 0
            else:
                stop_cnt += 1
            msg = "[VALID-BEST-I]: %.3f, %.3f, %d" % (best_valid_roc_i, best_test_roc_i,best_step_i)
            self.logger.info(msg)
            msg = "[VALID-BEST-S]: %.3f, %.3f, %d" % (best_valid_roc_s, best_test_roc_s,best_step_s)
            self.logger.info(msg)
            msg = "[VALID-BEST-P]: %.3f, %.3f, %d, %d" % (best_valid_roc_p, best_test_roc_p,best_step_p,stop_cnt)
            self.logger.info(msg)
            if stop_cnt >= 5:
                break
        return best_test_roc_i, best_test_roc_s, best_test_roc_p, best_test_micro_auc, best_test_auc_array

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

    def _get_numpy_from_variable(self, input):
        if self.option.cuda:
            return input.data.cpu().numpy()
        return input.data.numpy()