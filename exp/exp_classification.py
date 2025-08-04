from utils.tools import EarlyStopping
import torch
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from model.TSCMixer import Model
# from model.TSCMixer_AFB import Model
# from model.TSCMixer_NoAttention import Model
# from model.TSCMixer_NoAttention import Model
# from model.TSCMixer_TFL_TFH import Model
# from model.TSCMixer_TMS import Model
# from model.TSCMixer_TMS_TFH import Model
# from model.TSCMixer_TMS_TFL import Model
from torch.utils.data import DataLoader, RandomSampler
from utils.tools import plt_train_loss
from utils.losses import FocalLoss
warnings.filterwarnings('ignore')


class Exp_Classification():
    def __init__(self, configs):
        super(Exp_Classification, self).__init__()
        self.cfgs = configs
        self.model = Model(self.cfgs)

    def get_data(self, dataset):
        sampler = RandomSampler(dataset)
        shuffled_indices = list(sampler)
        data_loader = DataLoader(dataset=dataset, batch_size=self.cfgs['training']['batch_size'], shuffle=False, num_workers=0, sampler=sampler)
        return data_loader, shuffled_indices

    def _select_optimizer(self):
        if self.cfgs['training']['optimizer'] == 'Adam':
            model_optim = optim.Adam([{'params': self.model.parameters()}, ], lr=self.cfgs['training']['learning_rate'])
            return model_optim

    def _select_criterion(self):
        if self.cfgs['training']['criterion'] == 'CrossEntropyLoss':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            alpha = self.cfgs['data']['num_per_classes']/torch.sum(self.cfgs['data']['num_per_classes']) + 1 / torch.sum(self.cfgs['data']['num_per_classes'])
            criterion = FocalLoss(alpha=alpha, gamma=1)
        return criterion

    def train(self, data_train, data_val, data_test):
        train_loader, _ = self.get_data(data_train)
        vali_loader, _ = self.get_data(data_val)
        test_loader, _ = self.get_data(data_test)
        checkpoints_save_path = self.cfgs['training']['ckp_path']
        if not os.path.exists(checkpoints_save_path):
            os.makedirs(checkpoints_save_path)
        checkpoints_save_path = os.path.join(checkpoints_save_path,
                                             '{}_{}_d{}_M{}_pl{}_Dicp{}_N{}_ep{}_lr{}_bs{}_pt{}_ws{}.pth'.
                                             format(self.cfgs['model_name'],
                                                    self.cfgs['data']['data_dir'].split('/')[-1],
                                                    self.cfgs['model']['d'],
                                                    self.cfgs['model']['M'],
                                                    self.cfgs['model']['patch_len'],
                                                    self.cfgs['model']['Dicp'],
                                                    self.cfgs['model']['N'],
                                                    self.cfgs['training']['epochs'],
                                                    self.cfgs['training']['learning_rate'],
                                                    self.cfgs['training']['batch_size'],
                                                    self.cfgs['training']['patience'],
                                                    self.cfgs['training']['warmup_steps_ratio']
                                                    ))
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.cfgs['training']['patience'], verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        from utils.optim_utils import get_lr_scheduler
        total_steps = len(train_loader) * self.cfgs['training']['epochs']
        if isinstance(self.cfgs['training']['warmup_steps_ratio'], float):
            self.cfgs['training']['warmup_steps'] = int(self.cfgs['training']['warmup_steps_ratio'] * total_steps)
        scheduler = get_lr_scheduler(model_optim, total_steps=total_steps, cfgs=self.cfgs)

        device = torch.device('cuda:0') if self.cfgs['training']['use_gpu'] else torch.device('cpu')
        train_loss_list, test_acc_list = [], []

        for epoch in range(self.cfgs['training']['epochs']):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x = batch['x']
                batch_y = batch['y'].squeeze().to(device)
                model_optim.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()
                scheduler.step()
                train_loss.append(loss.item())
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_list.append(train_loss)
            # raise Exception
            self.model.eval()
            train_acc = self.vail(vali_loader)
            test_acc = self.vail(test_loader)
            test_acc_list.append(test_acc)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train acc: {3:.7f} Test acc: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, train_acc, test_acc))
            early_stopping(train_loss, self.model, checkpoints_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        plt_train_loss(train_loss_list, test_acc_list, 'train_loss.jpg')

    def vail(self, data_loader):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        test_total, correct = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch_x = batch['x']
                batch_y = batch['y'].squeeze().to(device)
                outputs = self.model(batch_x)
                predicted = torch.argmax(outputs, 1)
                test_total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / test_total

    def test(self, dataset):
        test_loader, _ = self.get_data(dataset)
        print('loading model')
        checkpoints_path = os.path.join(self.cfgs['training']['ckp_path'],
                                             '{}_{}_d{}_M{}_pl{}_Dicp{}_N{}_ep{}_lr{}_bs{}_pt{}_ws{}.pth'.
                                             format(self.cfgs['model_name'],
                                                    self.cfgs['data']['data_dir'].split('/')[-1],
                                                    self.cfgs['model']['d'],
                                                    self.cfgs['model']['M'],
                                                    self.cfgs['model']['patch_len'],
                                                    self.cfgs['model']['Dicp'],
                                                    self.cfgs['model']['N'],
                                                    self.cfgs['training']['epochs'],
                                                    self.cfgs['training']['learning_rate'],
                                                    self.cfgs['training']['batch_size'],
                                                    self.cfgs['training']['patience'],
                                                    self.cfgs['training']['warmup_steps_ratio']
                                                    ))
        self.model.load_state_dict(torch.load(checkpoints_path))

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        test_total, correct = 0, 0
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x = batch['x']
                batch_y = batch['y'].squeeze().to(device)
                outputs = self.model(batch_x)
                predicted = torch.argmax(outputs, 1)
                test_total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        print(f'test Acc: {correct/test_total}')
        temp = {'Acc': [correct/test_total], 'Hyperparameters': [checkpoints_path.split('/')[-1][:-4]]}
        temp_df = pd.DataFrame(temp)
        save_dir = f"./csv_results/csv_results_{self.cfgs['model_name']}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"result_{self.cfgs['data']['data_dir'].split('/')[-1]}.csv")
        if not os.path.exists(save_path):
            temp_df.to_csv(save_path, sep=',', index=False, header=True)
        else:
            temp_df.to_csv(save_path, mode='a', sep=',', index=False, header=False)


