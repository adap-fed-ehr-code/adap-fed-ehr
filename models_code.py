# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from layers import L0Mask,L0Mask_D
import numpy as np


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

class MLP1(nn.Module):
    def __init__(self,dims = (32,32)):
        super(MLP1,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        # nn.init.constant_(self.fc1.weight, 0)
        self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        nn.init.constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=dims[1])

    def forward(self, x):
        return self.fc1(x)

class ClassiflierS1(nn.Module):
    def __init__(self,dims = (32,2)):
        super(ClassiflierS1,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        nn.init.constant_(self.fc1.bias, 0)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.fc1(x)


class ClassiflierB1(nn.Module):
    def __init__(self,dims = (32,1),if_bias = True):
        super(ClassiflierB1,self).__init__()
        self.fc1 = nn.Linear(dims[0],1,bias=if_bias)
        nn.init.constant_(self.fc1.weight, 0)
        # self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        if if_bias:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x, bias = None):
        out = self.fc1(x)
        if bias is None:
            return torch.cat((torch.zeros_like(out), out),1)
        else:
            return torch.cat((torch.zeros_like(out), out + bias), 1)


class RegressorB1(nn.Module):
    def __init__(self,dims = (32,1),if_bias = True):
        super(RegressorB1,self).__init__()
        self.fc1 = nn.Linear(dims[0],1,bias=if_bias)
        nn.init.constant_(self.fc1.weight, 0)
        if if_bias:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x, bias = None):
        out = self.fc1(x)
        if bias is None:
            return out
        else:
            return out + bias


class Reconstructor1(nn.Module):
    def __init__(self,dims = (32,1),if_bias = True):
        super(Reconstructor1,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        nn.init.constant_(self.fc1.weight, 0)
        if if_bias:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        return self.fc1(x)


class ClassiflierB2(nn.Module):
    def __init__(self,dims = (32,8,1),if_bias = True):
        super(ClassiflierB2,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.fc2= nn.Linear(dims[1],1,bias=if_bias)
        self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        self.fc2.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        nn.init.constant_(self.fc1.bias, 0)
        if if_bias:
            nn.init.constant_(self.fc2.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, x, bias = None):
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        if bias is None:
            return torch.cat((torch.zeros_like(out), out),1)
        else:
            return torch.cat((torch.zeros_like(out), out + bias), 1)


class RMTL_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(RMTL_R,self).__init__()
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)

    def forward(self, input_data):
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_i.append(self.M_s(input_data) + tempM(input_data))
        return class_outputs_s_i

    def forward_d(self, input_data, d):
        return self.M_s(input_data) + self.Ms_s[d](input_data)


class RMTL_C_FL(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(RMTL_C_FL,self).__init__()
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_fl = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_fl.append(tempM)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_train(self, input_data, d):
        return self.logsoftmax(self.Ms_fl[d](input_data) + self.Ms_s[d](input_data))

    def forward_valid(self, input_data, d):
        return self.logsoftmax(self.M_s(input_data) + self.Ms_s[d](input_data))


class RMTL_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(RMTL_C,self).__init__()
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_i.append(self.logsoftmax(self.M_s(input_data) + tempM(input_data)))
        return class_outputs_s_i

    def forward_d(self, input_data, d):
        return self.logsoftmax(self.M_s(input_data) + self.Ms_s[d](input_data))


class FS_R1(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_R1,self).__init__()
        self.E_s = L0Mask(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)

    def forward(self, input_data):
        feature_s = self.E_s(input_data)
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)

        return feature_s, class_outputs_s_s, class_outputs_s_i

    def regularization(self):
        return self.E_s.regularization()

    def cal_feat_s(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i


class FS_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_R, self).__init__()
        self.G1 = L0Mask(model_config['Es_paras'][0])
        self.G2 = L0Mask_D(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)

    def forward(self, input_data):
        feature_s, feature_p = self.G2(self.G1(input_data))
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))
            class_outputs_r.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r

    def forward_d(self, input_data, d):
        feat_s, feat_p = self.G2(self.G1(input_data))
        output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
        output_p, output_r = self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1)),self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_s_d(self,input_data,d):
        feat_s, _ = self.G2(self.G1(input_data))
        return self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])

    def regularization(self):
        return self.G1.regularization()

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i

class FS2_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS2_R, self).__init__()
        self.G1 = L0Mask_D(model_config['Es_paras'][0])
        self.G2 = L0Mask_D(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] * 2 + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] * 2 + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)

    def forward(self, input_data):
        feature_sp, feature_c, m1 = self.G1(input_data)
        feature_s, feature_p, m2 = self.G2(feature_sp)
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(tempM(torch.cat((feature_p,grad_reverse(feature_c,0.5),self.M_s(feature_s)),1)))
            class_outputs_r.append(tempM(torch.cat((feature_p,grad_reverse(feature_c,0.5),self.M_s(feature_s)),1)))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r

    def forward_d(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
        output_p, output_r = self.Ms_p[d](torch.cat((feat_p,grad_reverse(feat_c,1.0),self.M_s(feat_s)),1)),self.Ms_p[d](torch.cat((feat_p,grad_reverse(feat_c,1.0),self.M_s(feat_s)),1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_d_valid(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
        output_p, output_r = self.Ms_p[d](torch.cat((feat_p,0.0 * feat_c,self.M_s(feat_s)),1)),self.Ms_p[d](torch.cat((feat_p,0.0 * feat_c,self.M_s(feat_s)),1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_s_d(self,input_data,d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        return feat_s, feat_p,self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])

    def regularization(self):
        return self.G1.regularization()

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i


class FS_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_R, self).__init__()
        self.G1 = L0Mask(model_config['Es_paras'][0])
        self.G2 = L0Mask_D(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][0] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)

    def forward(self, input_data):
        feature_s, feature_p = self.G2(self.G1(input_data))
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))
            class_outputs_r.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r

    def forward_d(self, input_data, d):
        feat_s, feat_p = self.G2(self.G1(input_data))
        output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
        output_p, output_r = self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1)),self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_s_d(self,input_data,d):
        feat_s, _ = self.G2(self.G1(input_data))
        return self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])

    def regularization(self):
        return self.G1.regularization()

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i


class FS_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_C, self).__init__()
        self.G1 = L0Mask_D(model_config['Es_paras'][0])
        self.G2 = L0Mask_D(model_config['Es_paras'][0])

        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0],num_classes),if_bias = True)
            self.Ms_p.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward_d(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))
        # output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))
        output_r = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def predict_numpy_d(self, input_data, d):
        input_data = torch.Tensor(input_data)
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        # output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))
        # output_p = self.softmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))
        output_p = output_p.detach().numpy()
        return output_p


    def forward_d_valid(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))
        # output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))
        output_r = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_s_d(self,input_data,d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        return feat_s, feat_p,self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))

    def regularization(self):
        return self.G1.regularization()

    def cal_feat(self, input_data):
        return self.E_s(input_data)


class FS_GLOBAL(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_GLOBAL, self).__init__()
        self.G1 = L0Mask_D(model_config['Es_paras'][0])
        self.G2 = L0Mask_D(model_config['Es_paras'][0])

        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0],num_classes),if_bias = True)
            self.Ms_p.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward_d(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.logsoftmax(self.M_s(input_data)),self.logsoftmax(self.Ms_s[d](input_data))
        # output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))
        output_r = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def predict_numpy_d(self, input_data, d):
        input_data = torch.Tensor(input_data)
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        # output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))
        # output_p = self.softmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](input_data) + self.M_s(input_data))
        output_p = output_p.detach().numpy()
        return output_p


    def forward_d_valid(self, input_data, d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        output_s_s, output_s_i = self.logsoftmax(self.M_s(input_data)),self.logsoftmax(self.Ms_s[d](input_data))
        # output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        output_p = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))
        output_r = self.logsoftmax(self.Ms_p[d](feat_p) + self.M_s(feat_s))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r

    def forward_s_d(self,input_data,d):
        feat_sp, feat_c = self.G1(input_data)
        feat_s, feat_p = self.G2(feat_sp)
        return feat_s, feat_p,self.logsoftmax(self.M_s(input_data)),self.logsoftmax(self.Ms_s[d](input_data))

    def regularization(self):
        return self.G1.regularization()

    def cal_feat(self, input_data):
        return self.E_s(input_data)


# class FS_C(nn.Module):
#     def __init__(self, option, model_config, num_classes, num_domains):
#         """
#         Input:
#             E: encoder
#             M: classifier
#             num_classes: the number of classes
#          """
#         super(FS_C, self).__init__()
#         self.G1 = L0Mask_D(model_config['Es_paras'][0])
#         self.G2 = L0Mask_D(model_config['Es_paras'][0])
#
#         domain_bias = np.zeros(num_domains).astype(np.float32)
#         domain_bias = torch.Tensor(domain_bias)
#         if option.cuda:
#             print("cuda bias")
#             self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
#         else:
#             self.domain_bias = torch.nn.Parameter(domain_bias)
#         self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
#         self.Ms_s = []
#         for _ in range(num_domains):
#             if option.cuda:
#                 tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
#             else:
#                 tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
#             self.Ms_s.append(tempM)
#         self.Ms_p = []
#         for _ in range(num_domains):
#             if option.cuda:
#                 tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] * 2 + 1, num_classes), if_bias=True).cuda()
#             else:
#                 tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] * 2 + 1,num_classes),if_bias = True)
#             self.Ms_p.append(tempM)
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def forward_d(self, input_data, d):
#         feat_sp, feat_c = self.G1(input_data)
#         feat_s, feat_p = self.G2(feat_sp)
#         output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
#         output_p, output_r = self.Ms_p[d](torch.cat((feat_p,grad_reverse(feat_c,1.0),self.M_s(feat_s)[:,1][:,None]),1)),self.Ms_p[d](torch.cat((feat_p,grad_reverse(feat_c,1.0),self.M_s(feat_s)[:,1][:,None]),1))
#
#         return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r
#
#     def forward_d_valid(self, input_data, d):
#         feat_sp, feat_c = self.G1(input_data)
#         feat_s, feat_p = self.G2(feat_sp)
#         output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s,self.domain_bias[d])),self.logsoftmax(self.Ms_s[d](feat_s,self.domain_bias[d]))
#         output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,0.0 * feat_c,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,0.0 * feat_c,self.M_s(feat_s)[:,1][:,None]),1)))
#
#         return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r
#
#     def forward_s_d(self,input_data,d):
#         feat_sp, feat_c = self.G1(input_data)
#         feat_s, feat_p = self.G2(feat_sp)
#         return feat_s, feat_p,self.logsoftmax(self.M_s(feat_s,self.domain_bias[d])),self.logsoftmax(self.Ms_s[d](feat_s,self.domain_bias[d]))
#
#     def regularization(self):
#         return self.G1.regularization()
#
#     def cal_feat(self, input_data):
#         return self.E_s(input_data)


class FS_C2(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_C2, self).__init__()
        self.E_s = L0Mask(model_config['Es_paras'][0])

        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        feature_s = self.E_s(input_data)
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)

        return feature_s, class_outputs_s_s, class_outputs_s_i

    def regularization(self):
        return self.E_s.regularization()

    def cal_feat_s(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.logsoftmax(self.M_s(feat_s)))
            class_outputs_s_i.append(self.logsoftmax(tempM(feat_s)))
        return class_outputs_s_s,class_outputs_s_i


class FS_C3(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(FS_C3, self).__init__()
        self.E_s = L0Mask_D(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        feature_s, feature_p = self.E_s(input_data)
        class_outputs_s_s, class_outputs_s_i,score_outputs_s_s,score_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)),1))))
            # print(tempM(feature_p))
            class_outputs_r.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)), 1))))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r,score_outputs_s_s,score_outputs_s_i

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        score_outputs_s_s = []
        score_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.logsoftmax(self.M_s(feat_s,self.domain_bias[d_idx])))
            class_outputs_s_i.append(self.logsoftmax(tempM(feat_s, self.domain_bias[d_idx])))
            score_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            score_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i,score_outputs_s_s,score_outputs_s_i


class FS_C1(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifierf
            num_classes: the number of classes
         """
        super(FS_C1, self).__init__()
        self.E_s = L0Mask_D(model_config['Es_paras'][0])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def regularization(self):
        return self.E_s.regularization()

    def forward(self, input_data):
        feature_s,_ = self.E_s(input_data)
        feature_p = feature_s
        class_outputs_s_s, class_outputs_s_i,score_outputs_s_s,score_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)),1))))
            # print(tempM(feature_p))
            class_outputs_r.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)), 1))))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        score_outputs_s_s = []
        score_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.logsoftmax(self.M_s(feat_s,self.domain_bias[d_idx])))
            class_outputs_s_i.append(self.logsoftmax(tempM(feat_s, self.domain_bias[d_idx])))
            score_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            score_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i,score_outputs_s_s,score_outputs_s_i


class GLOBAL_C_FL(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(GLOBAL_C_FL,self).__init__()
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_valid(self, input_data):
        return self.logsoftmax(self.M_s(input_data))

    def forward_train(self, input_data, d):
        return self.logsoftmax(self.Ms_s[d](input_data))


class GLOBAL_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(GLOBAL_C, self).__init__()
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        return self.logsoftmax(self.M_s(input_data))


class GLOBAL_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(GLOBAL_R, self).__init__()
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        return self.M_s(input_data)


class INDIV_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(INDIV_C, self).__init__()
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True)
            self.Ms_s.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_d(self, input_data, d):
        return self.logsoftmax(self.Ms_s[d](input_data))

    def forward(self, input_data):
        class_outputs = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs.append(self.logsoftmax(tempM(input_data)))
        return class_outputs

    def regularization(self):
        reg = 0.0
        for i in range(len(self.Ms_s)):
            for j in range(len(self.Ms_s)):
                if i != j:
                    reg += ((self.Ms_s[i].fc1.weight - self.Ms_s[j].fc1.weight) ** 2).mean()
        return reg / (len(self.Ms_s) * (len(self.Ms_s) - 1))

    def regularization_d(self,d):
        reg = 0.0
        for j in range(len(self.Ms_s)):
            if d != j:
                reg += ((self.Ms_s[d].fc1.weight - self.Ms_s[j].fc1.weight) ** 2).mean()
        return reg / (len(self.Ms_s) - 1)

    def forward_d(self, input_data, d):
        return self.logsoftmax(self.Ms_s[d](input_data))


class INDIV_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(INDIV_R,self).__init__()
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = True)
            self.Ms_s.append(tempM)

    def regularization(self):
        reg = 0.0
        for i in range(len(self.Ms_s)):
            for j in range(len(self.Ms_s)):
                if i != j:
                    reg += ((self.Ms_s[i].fc1.weight - self.Ms_s[j].fc1.weight) ** 2).mean()
        return reg / (len(self.Ms_s) * (len(self.Ms_s) - 1))

    def regularization_d(self,d):
        reg = 0.0
        for j in range(len(self.Ms_s)):
            if d != j:
                reg += ((self.Ms_s[d].fc1.weight - self.Ms_s[j].fc1.weight) ** 2).mean()
        return reg / (len(self.Ms_s) - 1)

    def forward(self, input_data):
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_i.append(tempM(input_data))
        return class_outputs_s_i

    def forward_d(self, input_data, d):
        return self.Ms_s[d](input_data)


# class FS_C(nn.Module):
#     def __init__(self, option, model_config, num_classes, num_domains):
#         """
#         Input:
#             E: encoder
#             M: classifier
#             num_classes: the number of classes
#          """
#         super(FS_C, self).__init__()
#         self.G1 = L0Mask(model_config['Es_paras'][0])
#         self.G2 = L0Mask_D(model_config['Es_paras'][0])
#
#         domain_bias = np.zeros(num_domains).astype(np.float32)
#         domain_bias = torch.Tensor(domain_bias)
#         if option.cuda:
#             print("cuda bias")
#             self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
#         else:
#             self.domain_bias = torch.nn.Parameter(domain_bias)
#         self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes),if_bias = False)
#         self.Ms_s = []
#         for _ in range(num_domains):
#             if option.cuda:
#                 tempM = ClassiflierB1(dims=(model_config['Es_paras'][0], num_classes), if_bias=False).cuda()
#             else:
#                 tempM = ClassiflierB1(dims=(model_config['Es_paras'][0],num_classes),if_bias = False)
#             self.Ms_s.append(tempM)
#         self.Ms_p = []
#         for _ in range(num_domains):
#             if option.cuda:
#                 tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1, num_classes), if_bias=True).cuda()
#             else:
#                 tempM = ClassiflierB1(dims=(model_config['Ep_paras'][0] + 1,num_classes),if_bias = True)
#             self.Ms_p.append(tempM)
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def regularization(self):
#         return self.G1.regularization()
#
#     def forward(self, input_data):
#         feature_s,feature_p = self.G2(self.G1(input_data))
#         class_outputs_s_s, class_outputs_s_i,score_outputs_s_s,score_outputs_s_i = self.cal_output_feat_s(feature_s)
#         class_outputs_p = []
#         class_outputs_r = []
#         for tempM in self.Ms_p:
#             class_outputs_p.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)),1))))
#             # print(tempM(feature_p))
#             class_outputs_r.append(self.logsoftmax(tempM(torch.cat((feature_p, self.M_s(feature_s)[:,1].unsqueeze(-1)), 1))))
#
#         return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r, score_outputs_s_s,score_outputs_s_i
#
#     def cal_feat(self, input_data):
#         return self.E_s(input_data)
#
#     def cal_output_s(self, input_data):
#         feature_s,feature_p = self.G2(self.G1(input_data))
#         class_outputs_s_s, class_outputs_s_i,score_outputs_s_s,score_outputs_s_i = self.cal_output_feat_s(feature_s)
#
#         return class_outputs_s_s, class_outputs_s_i,score_outputs_s_s,score_outputs_s_i
#
#     def cal_output_feat_s(self, feat_s):
#         class_outputs_s_s = []
#         class_outputs_s_i = []
#         score_outputs_s_s = []
#         score_outputs_s_i = []
#         for (d_idx,tempM) in enumerate(self.Ms_s):
#             class_outputs_s_s.append(self.logsoftmax(self.M_s(feat_s,self.domain_bias[d_idx])))
#             class_outputs_s_i.append(self.logsoftmax(tempM(feat_s, self.domain_bias[d_idx])))
#             score_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
#             score_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
#         return class_outputs_s_s,class_outputs_s_i,score_outputs_s_s,score_outputs_s_i


class DomainClassifier(nn.Module):
    def __init__(self,dims = (32,2)):
        super(DomainClassifier,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        nn.init.constant_(self.fc1.bias, 0)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.logsoftmax(self.fc1(x))


class MAN_R(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(MAN_R, self).__init__()

        self.F_s = MLP1(model_config['Es_paras'])
        self.F_ps = []
        for _ in range(num_domains):
            if option.cuda:
                tempF = MLP1(model_config['Ep_paras']).cuda()
            else:
                tempF = MLP1(model_config['Ep_paras'])
            self.F_ps.append(tempF)

        self.M_s = RegressorB1(dims=(model_config['Es_paras'][1] + model_config['Ep_paras'][1], num_classes))
        self.D = DomainClassifier(dims=(model_config['Es_paras'][1], num_domains))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_d(self, input_data):
        feat_s = self.F_s(input_data)
        return self.D(feat_s)

    def forward_y(self, input_data, d):
        feat_s, feat_d = self.F_s(input_data), self.F_ps[d](input_data)
        return self.M_s(torch.cat((feat_s,feat_d),1))


class MAN_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(MAN_C, self).__init__()

        self.F_s = MLP1(model_config['Es_paras'])
        self.F_ps = []
        for _ in range(num_domains):
            if option.cuda:
                tempF = MLP1(model_config['Ep_paras']).cuda()
            else:
                tempF = MLP1(model_config['Ep_paras'])
            self.F_ps.append(tempF)

        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][1] + model_config['Ep_paras'][1], num_classes))
        self.D = DomainClassifier(dims=(model_config['Es_paras'][1], num_domains))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_d(self, input_data):
        feat_s = self.F_s(input_data)
        return self.D(feat_s)

    def forward_y(self, input_data, d):
        feat_s, feat_d = self.F_s(input_data), self.F_ps[d](input_data)
        return self.logsoftmax(self.M_s(torch.cat((feat_s,feat_d),1)))


class R_C(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(R_C, self).__init__()
        self.E_s = MLP1(model_config['Es_paras'])
        self.E_p = MLP1(model_config['Ep_paras'])
        self.E_c = MLP1(model_config['Ec_paras'])

        self.R = Reconstructor1([model_config['Es_paras'][1] + model_config['Ep_paras'][1] + model_config['Ec_paras'][1],model_config['Es_paras'][0]])
        self.M_s = ClassiflierB1(dims=(model_config['Es_paras'][1], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][1], num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Es_paras'][1],num_classes),if_bias = True)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][1] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = ClassiflierB1(dims=(model_config['Ep_paras'][1] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_d(self, input_data, d):
        feat_s, feat_p, feat_c = self.E_s(input_data), self.E_p(input_data), self.E_c(input_data)
        output_s_s, output_s_i = self.logsoftmax(self.M_s(feat_s)),self.logsoftmax(self.Ms_s[d](feat_s))
        output_p, output_r = self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1))),self.logsoftmax(self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)[:,1][:,None]),1)))
        reconstruct_x = self.R(torch.cat([feat_s, feat_p, feat_c], 1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r, reconstruct_x



class R_R2(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(R_R2, self).__init__()
        self.E_s = MLP1(model_config['Es_paras'])
        self.E_p = MLP1(model_config['Ep_paras'])
        self.E_c = MLP1(model_config['Ec_paras'])

        self.R = Reconstructor1([model_config['Es_paras'][1] + model_config['Ep_paras'][1] + model_config['Ec_paras'][1],model_config['Es_paras'][0]])
        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][1], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][1], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][1],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][1] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][1] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)

    def forward_d(self, input_data, d):
        feat_s, feat_p, feat_c = self.E_s(input_data), self.E_p(input_data), self.E_c(input_data)
        output_s_s, output_s_i = self.M_s(feat_s,self.domain_bias[d]),self.Ms_s[d](feat_s,self.domain_bias[d])
        output_p, output_r = self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1)),self.Ms_p[d](torch.cat((feat_p,self.M_s(feat_s)),1))
        reconstruct_x = self.R(torch.cat([feat_s, feat_p, feat_c], 1))

        return feat_s, feat_p, output_s_s, output_s_i, output_p, output_r, reconstruct_x

    def forward(self, input_data):
        feature_s, feature_p, feature_c = self.E_s(input_data), self.E_p(input_data), self.E_c(input_data)
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))
            class_outputs_r.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))
        reconstruct_x = self.R(torch.cat([feature_s,feature_p,feature_c],1))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r, reconstruct_x

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i


class R_R1(nn.Module):
    def __init__(self, option, model_config, num_classes, num_domains):
        """
        Input:
            E: encoder
            M: classifier
            num_classes: the number of classes
         """
        super(R_R1, self).__init__()
        if len(model_config['Es_paras']) == 2:
            self.E_s = MLP1(model_config['Es_paras'])
            self.E_p = MLP1(model_config['Ep_paras'])
        else:
            self.E_s = MLP2(model_config['Es_paras'])
            self.E_p = MLP2(model_config['Ep_paras'])

        domain_bias = np.zeros(num_domains).astype(np.float32)
        domain_bias = torch.Tensor(domain_bias)
        if option.cuda:
            print("cuda bias")
            self.domain_bias = torch.nn.Parameter(domain_bias.cuda())
        else:
            self.domain_bias = torch.nn.Parameter(domain_bias)
        self.M_s = RegressorB1(dims=(model_config['Es_paras'][1], num_classes),if_bias = False)
        self.Ms_s = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Es_paras'][1], num_classes), if_bias=False).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Es_paras'][1],num_classes),if_bias = False)
            self.Ms_s.append(tempM)
        self.Ms_p = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][1] + 1, num_classes), if_bias=True).cuda()
            else:
                tempM = RegressorB1(dims=(model_config['Ep_paras'][1] + 1,num_classes),if_bias = True)
            self.Ms_p.append(tempM)

    def forward(self, input_data):
        feature_s, feature_p = self.E_s(input_data), self.E_p(input_data)
        class_outputs_s_s, class_outputs_s_i = self.cal_output_feat_s(feature_s)
        class_outputs_p = []
        class_outputs_r = []
        for tempM in self.Ms_p:
            class_outputs_p.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))
            class_outputs_r.append(tempM(torch.cat((feature_p,self.M_s(feature_s)),1)))

        return feature_s, feature_p, class_outputs_s_s, class_outputs_s_i, class_outputs_p, class_outputs_r

    def cal_feat(self, input_data):
        return self.E_s(input_data)

    def cal_output_feat_s(self, feat_s):
        class_outputs_s_s = []
        class_outputs_s_i = []
        for (d_idx,tempM) in enumerate(self.Ms_s):
            class_outputs_s_s.append(self.M_s(feat_s,self.domain_bias[d_idx]))
            class_outputs_s_i.append(tempM(feat_s, self.domain_bias[d_idx]))
        return class_outputs_s_s,class_outputs_s_i


