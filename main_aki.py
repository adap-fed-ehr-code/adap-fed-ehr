import argparse

# Import libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# local utils
from utils import Logger
import os.path as osp
from data_loader import WholeDataLoader
import trainer_fs_c
import torch
import os
import random

def cal_loss(X,Y,model):
    lr_pred_y_prob = model.predict_proba(X)
    lr_y_loss = 0.0
    for idx in range(len(Y)):
        lr_y_loss += -np.log(lr_pred_y_prob[idx, Y[idx]])
    return lr_y_loss / len(Y)

def array2str(t_arr):
    s_arr = [str(t_item) for t_item in t_arr]
    return ",".join(s_arr)

day = 1
window_size = 1
print('=== eICU ===')

patient_medication_dict = {}
with open("data/medication_list_entity_day{}.csv".format(day),"r") as fout:
    for in_line in fout.readlines():
        in_info = in_line.replace("\n","").split(":")
        patient_medication_dict[int(in_info[0])] = [int(item) for item in in_info[1].split(",")]

df_eicu_day1 = pd.read_csv('data/X_eicu_day1_table.csv.gz', sep=',', index_col=0)
MIN_PAT = 3000

hospital_list = df_eicu_day1['hospitalid'].value_counts()
hospital_list = hospital_list[hospital_list>=MIN_PAT].index.values
print('Retaining {} of {} hospitals with at least {} patients.'.format(
    len(hospital_list), df_eicu_day1['hospitalid'].nunique(), MIN_PAT))

df_eicu = pd.read_csv('data/X_eicu_day1_table.csv.gz', sep=',', index_col=0)
df_eicu = df_eicu.loc[df_eicu['hospitalid'].isin(hospital_list), :]

hospitals_train = np.random.permutation(len(hospital_list))
hospitals_train = hospitals_train[0:int(len(hospitals_train)/2)]
hospitals_train = hospital_list[hospitals_train]

var_other = ['hospitalid', 'death', 'hosp_los', 'ventdays', 'hosp_los.1']
# convenient reference to death column
d_eicu_all = df_eicu['hospitalid'].values
X_eicu_all_df = df_eicu.drop(var_other,axis=1)
feature_names = X_eicu_all_df.columns.values.tolist()
X_eicu_all = X_eicu_all_df.values
p_eicu_all = df_eicu.index.values

def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    if not torch.cuda.is_available() and option.cuda:
        option.cuda = False

    if option.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        #os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in option.gpu_ids])
        torch.cuda.manual_seed_all(option.random_seed)
        # cudnn.benchmark = option.cudnn_benchmark
    if option.train_baseline:
        option.is_train = True


parser = argparse.ArgumentParser()
parser.add_argument('--n_class',          default=10,     type=int,   help='number of classes')
parser.add_argument('--input_size',       default=28,     type=int,   help='input size')
parser.add_argument('--batch_size',       default=512,    type=int,   help='mini-batch size')
parser.add_argument('--momentum',         default=0.0,    type=float, help='sgd momentum')
parser.add_argument('--lr',               default=0.01,   type=float, help='initial learning rate')
parser.add_argument('--lr_2',               default=0.04,   type=float, help='initial learning rate')
parser.add_argument('--weight_decay',   default=0.0001, type=float, help='sgd optimizer weight decay')
parser.add_argument('--weight_decay_2',   default=0.1, type=float, help='sgd optimizer weight decay')
parser.add_argument('--lamb_stable',   default=0.2, type=float, help='sgd optimizer weight decay')
parser.add_argument('--lamb_sparse',   default=0.1, type=float, help='sgd optimizer weight decay')
parser.add_argument('--max_step',         default=50,    type=int,   help='maximum step for training')
parser.add_argument('--change_step',      default=50,    type=int,   help='step for training to change loss function')
parser.add_argument('--depth',            default=20,     type=int,   help='depth of network')
parser.add_argument('--seed',             default=2,      type=int,   help='seed index')
parser.add_argument('--test_idx',         default=0,      type=int,   help='test index')


parser.add_argument('--checkpoint',       default=None, help='checkpoint to resume')
parser.add_argument('--log_step',         default=100,     type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=1,     type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',         default='./data/',               help='data directory')
parser.add_argument('--save_dir',         default='./results_new/',               help='save directory for checkpoint')
parser.add_argument('--data_split',       default='test',            help='data split to use')
parser.add_argument('--use_pretrain',     action='store_true',        help='whether it use pre-trained parameters if exists')
parser.add_argument('--train_baseline',   action='store_true',        help='whether it train baseline or unlearning')


parser.add_argument('--random_seed',                      type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
parser.add_argument('--cudnn_benchmark',  default=True,   type=bool,  help='cuDNN benchmark')


parser.add_argument('--cuda',             action='store_true',        help='enables cuda')
parser.add_argument('-d', '--debug',      action='store_true',        help='debug mode')
parser.add_argument('--is_train', default=True, help='whether it is training')

option_fs = parser.parse_args()

bmi_dict = {}
patient = pd.read_csv('data/patient.csv', sep=',', index_col=None).values
for idx in range(patient.shape[0]):
    if float(patient[idx, 8]) > 0:
        bmi_dict[patient[idx, 0]] = float(patient[idx, 22]) / ((float(patient[idx, 8]) / 100.0) ** 2)
    else:
        bmi_dict[patient[idx, 0]] = np.nan

X_bmi = []
for pid in p_eicu_all:
    if pid in bmi_dict:
        X_bmi.append(bmi_dict[pid])
X_bmi = np.array(X_bmi)
X_bmi = X_bmi[:, None]

num_medication = 237
X_med_all = []
for pid in p_eicu_all:
    temp_array = np.zeros((1, num_medication)).astype(np.float)
    if pid in patient_medication_dict:
        for m_item in patient_medication_dict[pid]:
            temp_array[0, m_item] = 1
    X_med_all.append(temp_array)
X_med_all = np.concatenate(X_med_all, 0)
print("Shape:", X_med_all.shape)
print(X_med_all.mean())

aki_label = np.load("data/aki_label.npy")
aki_dict = {}
for idx in range(aki_label.shape[0]):
    aki_dict[int(aki_label[idx, 0])] = aki_label[idx, 1]

X_eicu_all = np.concatenate((X_bmi, X_eicu_all), 1)
X_eicu, d_eicu, p_eicu, y_eicu, X_med = [], [], [], [], []
for (idx, pid) in enumerate(p_eicu_all):
    if pid in aki_dict.keys():
        a_label = aki_dict[pid]
        if a_label >= 0 and a_label < day * 24 * 60:
            continue
        elif a_label >= day * 24 * 60 and a_label < (day + window_size) * 24 * 60:
            y_eicu.append(1)
        else:
            y_eicu.append(0)
        X_eicu.append(X_eicu_all[idx, :][None, :])
        X_med.append(X_med_all[idx, :][None, :])
        d_eicu.append(d_eicu_all[idx])
        p_eicu.append(pid)

X_eicu, X_med = np.concatenate(X_eicu, 0), np.concatenate(X_med, 0)
p_eicu, y_eicu, d_eicu, y_eicu = np.array(p_eicu), np.array(y_eicu), np.array(d_eicu), np.array(y_eicu)
s_idx = []
s_names = []

print("Select_size:", len(s_idx))

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X_eicu)
X_eicu = imputer.transform(X_eicu)
scaler = StandardScaler()
scaler.fit(X_eicu)
X_eicu = scaler.transform(X_eicu)
scaler = StandardScaler()
scaler.fit(X_med)
X_med = scaler.transform(X_med)

X = np.concatenate((X_eicu, X_med), 1)
y = y_eicu
d = d_eicu

print("Sample size: {}; Positive label proportion: {}".format(X_eicu.shape[0], y_eicu.mean()))

u_d = np.unique(d)
d_map = {}
for (idx, s_d) in enumerate(u_d):
    d_map[s_d] = idx
d = np.array([d_map[s_d] for s_d in d])
print(d_map)
print("Total number of features:", X.shape[1])

num_train = int(0.7 * X.shape[0])
num_val = int(0.85 * X.shape[0])

all_roc_s,all_roc_i = [],[]
all_roc_s_2,all_roc_i_2 = [],[]

select_option = option_fs
pre_fix = "eicu_fs_aki/day%d_window%d" % (day, window_size)
result_dir = "results/" + pre_fix
if not osp.exists(result_dir):
    os.makedirs(result_dir)
logger = Logger(result_dir)

s_times = 10
all_macro_auc = []
all_micro_auc = []
all_auc_matrix = np.zeros((s_times, 7))
for s in range(10):
    np.random.seed(s + 40)
    random.seed(s + 40)
    torch.manual_seed(s + 40)
    torch.cuda.manual_seed(s + 40)

    rnd_idx = np.array(range(len(y)))
    np.random.shuffle(rnd_idx)
    # np.save("aki_data_split_day{}_window{}/rnd_idx_{}".format(day, window_size, s), rnd_idx)
    # rnd_idx = np.load("aki_data_split_day{}_window{}/rnd_idx_{}.npy".format(day, window_size, s))
    shuffle_X, shuffle_y, shuffle_d = X[rnd_idx, :], y[rnd_idx], d[rnd_idx]
    print("Total number of features:", X.shape[1])

    num_train = int(0.7 * X.shape[0])
    num_val = int(0.85 * X.shape[0])
    train_X,train_y,train_d = shuffle_X[:num_train, :],shuffle_y[:num_train],shuffle_d[:num_train]
    valid_X,valid_y,valid_d = shuffle_X[num_train:num_val, :],  shuffle_y[num_train:num_val], shuffle_d[num_train:num_val]
    test_X,test_y,test_d = shuffle_X[num_val:, :],shuffle_y[num_val:],shuffle_d[num_val:]

    train_loaders, valid_loaders, test_loaders = [], [], []
    for idx in range(0, test_d.max() + 1):
        train_X_d, train_y_d, train_d_d = train_X[train_d == idx, :], train_y[train_d == idx], train_d[train_d == idx]
        valid_X_d, valid_y_d, valid_d_d = valid_X[valid_d == idx, :], valid_y[valid_d == idx], valid_d[valid_d == idx]
        test_X_d, test_y_d, test_d_d = test_X[test_d == idx, :], test_y[test_d == idx], test_d[test_d == idx]

        train_dataset = WholeDataLoader(train_X_d.astype(np.float32), train_y_d.astype(np.int64), train_d_d)
        valid_dataset = WholeDataLoader(valid_X_d.astype(np.float32), valid_y_d.astype(np.int64), valid_d_d)
        test_dataset = WholeDataLoader(test_X_d.astype(np.float32), test_y_d.astype(np.int64), test_d_d)

        train_loaders.append(torch.utils.data.DataLoader(train_dataset,num_workers = 0,batch_size=select_option.batch_size,shuffle=True))
        valid_loaders.append(torch.utils.data.DataLoader(valid_dataset, num_workers = 0, batch_size=12800, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(test_dataset, num_workers = 0, batch_size=12800, shuffle=False))

    trainer_fs = trainer_fs_c.Trainer(option_fs, model_config={'Es_paras': (X.shape[1], 1),
                                                                           'Ep_paras': (X.shape[1], 1)}, logger =logger,
                                                     num_classes=2, num_domains=train_d.max() + 1)

    train_dataset_all = WholeDataLoader(train_X.astype(np.float32), train_y.astype(np.int), train_d)
    train_loader_all = torch.utils.data.DataLoader(train_dataset_all, num_workers=0, batch_size=option_fs.batch_size, shuffle=True)
    if option_fs.is_train:
        _, _, temp_macro_auc, temp_micro_auc, temp_all_auc = trainer_fs.train(train_loaders, valid_loaders, test_loaders)
        # p1 = trainer_fs.net.G1.sample_mask().detach().numpy()[0,:]
        # p2 = trainer_fs.net.G2.sample_mask().detach().numpy()[0,:]
        # print("Stable propotion:", p1.mean())
        # argsort_idx = np.argsort(p2)
        # s_stable = []
        # s_specific = []
        #
        # for s_idx in argsort_idx[::-1]:
        #     if p1[s_idx] > 0.6:
        #         s_stable.append(s_idx)
        #     if len(s_stable) > 20:
        #         break
        #
        # for s_idx in argsort_idx:
        #     if p1[s_idx] > 0.6:
        #         s_specific.append(s_idx)
        #     if len(s_specific) > 20:
        #         break

        all_macro_auc.append(temp_macro_auc)
        all_micro_auc.append(temp_micro_auc)
        all_auc_matrix[s, :] = temp_all_auc
        print("Current result:")
        logger.info(",".join([str(t_mse) for t_mse in all_macro_auc]))
        logger.info("{} \pm {}, [{},{}]".format(str(np.array(all_macro_auc).mean()),str(np.array(all_macro_auc).std()),
                                                str(np.array(all_macro_auc).mean() - 1.96 * np.array(all_macro_auc).std()),
                                                str(np.array(all_macro_auc).mean() + 1.96 * np.array(all_macro_auc).std())))
        logger.info(",".join([str(t_mse) for t_mse in all_micro_auc]))
        logger.info("{} \pm {}, [{},{}]".format(str(np.array(all_micro_auc).mean()),str(np.array(all_micro_auc).std()),
                                                str(np.array(all_micro_auc).mean() - 1.96 * np.array(all_micro_auc).std()),
                                                str(np.array(all_micro_auc).mean() + 1.96 * np.array(all_micro_auc).std())))
        print("For each hospital")
        for h_idx in range(all_auc_matrix.shape[1]):
            logger.info("{} \pm {}, [{},{}]".format(str(all_auc_matrix[:s+1,h_idx].mean()),str(all_auc_matrix[:s+1,h_idx].std()),
                                                    str(all_auc_matrix[:s+1,h_idx].mean() - 1.96 * all_auc_matrix[:s+1,h_idx].std()),
                                                    str(all_auc_matrix[:s+1,h_idx].mean() + 1.96 * all_auc_matrix[:s+1,h_idx].std())))

print("All result:")
logger.info(",".join([str(t_mse) for t_mse in all_macro_auc]))
logger.info("{} \pm {}, [{},{}]".format(str(np.array(all_macro_auc).mean()), str(np.array(all_macro_auc).std()),
                                        str(np.array(all_macro_auc).mean() - 1.96 * np.array(all_macro_auc).std()),
                                        str(np.array(all_macro_auc).mean() + 1.96 * np.array(all_macro_auc).std())))
logger.info(",".join([str(t_mse) for t_mse in all_micro_auc]))
logger.info("{} \pm {}, [{},{}]".format(str(np.array(all_micro_auc).mean()), str(np.array(all_micro_auc).std()),
                                        str(np.array(all_micro_auc).mean() - 1.96 * np.array(all_micro_auc).std()),
                                        str(np.array(all_micro_auc).mean() + 1.96 * np.array(all_micro_auc).std())))
print("For each hospital")
for h_idx in range(all_auc_matrix.shape[1]):
    logger.info("{} \pm {}, [{},{}]".format(str(all_auc_matrix[:, h_idx].mean()), str(all_auc_matrix[:, h_idx].std()),
                                    str(all_auc_matrix[:, h_idx].mean() - 1.96 * all_auc_matrix[:,h_idx].std()),
                                    str(all_auc_matrix[:, h_idx].mean() + 1.96 * all_auc_matrix[:,h_idx].std())))
logger.close()