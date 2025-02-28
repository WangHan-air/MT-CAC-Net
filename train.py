import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader,random_split
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

# Hypermeters
rand_seed = 42
model_name = 'CAC-Net'
device = 'cuda:0'
input_size= 4
output_size1 = 3
output_size2 = 1
kernel_size = 3
levels = 3
batch_size= 64
hidden_size = 64
learning_rate = 0.001
dropout = 0.3
epochs = 800


# Data Process
def dataprocess(data):
    data.iloc[:, 1]=data.iloc[:,1]*0.084033613
    data.iloc[:, 2]=data.iloc[:,2]*2.5-0.25
    data.iloc[:, 3]=data.iloc[:,3]*0.101321188+0.5
    data.iloc[:, 4]=data.iloc[:,4]*0.011111111
    # data.iloc[:, 5]=data.iloc[:,5]*1.17525938-0.018966336
    # data.iloc[:, 6]=data.iloc[:,6]*5.775339301-0.065578978
    # data.iloc[:, 7]=data.iloc[:,7]*0.647187195+0.849014463

    grouped = data.groupby('condition')
    data_input = []
    data_label = []
    seq_lengths = [] 

    for condition, group in grouped:
        static = group.iloc[0, 1:3].values
        series = group.iloc[:, 3:8].values
        # print(static)

        input_seq = np.hstack([np.tile(static, (series.shape[0], 1)), series[:, :2]])
        label_seq = series[:, 2:]

        data_input.append(input_seq)
        data_label.append(label_seq)
        seq_lengths.append(len(input_seq))
        
    return data_input, data_label, seq_lengths

def slidewindow(data_input, data_label, lengths):
    x_length = 13
    y_length = 1
    y_step = 1
    x = []
    y = []

    for i in range(len(data_input)):
        for start_id in range(0, lengths[i]-x_length-y_length+1-y_step+1, y_length):
            x_data = np.array(data_input[i][start_id: start_id + x_length])
            y_data = np.array(data_label[i][start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            if np.isnan(x_data).any() or np.isnan(y_data).any():
                continue
            x.append(x_data)
            y.append(y_data)

    x = np.array(x)
    y = np.array(y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    return x, y


# Samples generation by sliding windows
data = pd.read_csv("ZDYdata-train-valid.csv")
x, y, len1 = dataprocess(data.iloc[0:37733,0:8])
x, y = slidewindow(x, y, len1)
print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))

test_data = pd.read_csv("ZDYdata-test.CSV")
test_x, test_y, test_len = dataprocess(test_data.iloc[0:3706,0:8])
test_x, test_y = slidewindow(test_x, test_y, test_len)
print('test_x_shape: {}  test_y_shape: {}'.format(test_x.shape, test_y.shape))

dataset = TensorDataset(x, y)
test_dataset = TensorDataset(test_x, test_y)
torch.manual_seed(42)

train_size = 27203
valid_size = 7772

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# Models
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class SEblock(nn.Module):
    def __init__(self, channels, reduction_ratio = 16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze and Excitation
        se = self.fc(self.avg_pool(x).view(x.size(0), -1))  
        se = se.unsqueeze(-1)  
        return x * se

class TCNSEblock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, reduction_ratio=16, dropout=0.3):
        super(TCNSEblock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        self.se = SEblock(n_outputs, reduction_ratio)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res



class TCNSE(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, reduction_ratio=16, dropout=0.3):
        super(TCNSE, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNSEblock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, reduction_ratio=reduction_ratio, dropout=dropout)]
            if i <= 1:
                layers += [nn.ReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class TCNSEMTL(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, reduction_ratio=16):
        super(TCNSEMTL, self).__init__()
        self.tcnse = TCNSE(input_size, num_channels, kernel_size, reduction_ratio=reduction_ratio, dropout=dropout)
        
        self.shared_fc = nn.Linear(num_channels[-1], num_channels[-1]*4)

        self.fc1_CL = nn.Linear(num_channels[-1]*4, num_channels[-1]*2)
        self.fc2_CL = nn.Linear(num_channels[-1]*2, num_channels[-1])
        self.fc3_CL = nn.Linear(num_channels[-1], output_size)

        self.fc1_CD = nn.Linear(num_channels[-1]*4, num_channels[-1]*2)
        self.fc2_CD = nn.Linear(num_channels[-1]*2, num_channels[-1])
        self.fc3_CD = nn.Linear(num_channels[-1], output_size)

        self.fc1_CM = nn.Linear(num_channels[-1]*4, num_channels[-1]*2)
        self.fc2_CM = nn.Linear(num_channels[-1]*2, num_channels[-1])
        self.fc3_CM = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcnse(x.transpose(1, 2)).transpose(1, 2) 
        
        shared_output = torch.relu(self.shared_fc(output))

        output_fc1_CL = torch.relu(self.fc1_CL(shared_output))
        output_fc2_CL = torch.relu(self.fc2_CL(output_fc1_CL))
        output_CL = output_fc2_CL[:, -1, :]  
        pred_CL = self.fc3_CL(output_CL)

        output_fc1_CD = torch.relu(self.fc1_CD(shared_output))
        output_fc2_CD = torch.relu(self.fc2_CD(output_fc1_CD))
        output_CD = output_fc2_CD[:, -1, :]
        pred_CD = self.fc3_CD(output_CD)

        output_fc1_CM = torch.relu(self.fc1_CM(shared_output))
        output_fc2_CM = torch.relu(self.fc2_CM(output_fc1_CM))
        output_CM = output_fc2_CM[:, -1, :]
        pred_CM = self.fc3_CM(output_CM)
        
        pred = torch.stack([pred_CL, pred_CD, pred_CM], dim=1).squeeze(-1)
        
        return pred
    

# Train
def train(net, train_loader, valid_loader, epochs, lr, model_save_pth1):
    
    rmse_valid_list = []
    mean_rmse_valid_list = []
    mean_mae_valid_list = []
    mean_mape_valid_list = []
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')      
    gamma = 0.99
    scheduler = ExponentialLR(optimizer, gamma)
    
    net.to(device) 
    
    for epoch in range(1, epochs + 1):
        net.train()
        mse_train = np.zeros(3)
        cnt = 0

        for batch_idx, (x_input, y_true) in enumerate(train_loader):
            x_input = x_input.to(device)
            y_true = y_true.to(device)
            y_pred = net(x_input)
    
            loss = criterion(y_pred, y_true.squeeze(1))
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            
            mse_train += loss.detach().sum(dim=0).cpu().numpy()  
            cnt += x_input.size(0)
    
        scheduler.step()
            
        mse_train /= cnt
        rmse_train = np.sqrt(mse_train)
        mean_rmse_train = np.mean(rmse_train)    

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, Current learning rate: {current_lr}')      
        
        
        net.eval()  
        mse_valid = np.zeros(3)  
        y_valid_true_final = []
        y_valid_pred_final = []
        cnt = 0
        for batch_idx, (x_input_valid, y_true_valid) in enumerate(valid_loader):
            x_input_valid = x_input_valid.to(device)
            y_true_valid = y_true_valid.to(device)
            y_valid_pred = net(x_input_valid)
        
            y_true_valid = y_true_valid.squeeze(1)
            loss_valid = criterion(y_valid_pred, y_true_valid) 
            
            mse_valid += loss_valid.detach().sum(dim=0).cpu().numpy()

            y_valid_true_final.extend(y_true_valid.data.detach().cpu().numpy())
            y_valid_pred_final.extend(y_valid_pred.data.detach().cpu().numpy())

            cnt += x_input_valid.size(0)

        mse_valid /= cnt
        rmse_valid = np.sqrt(mse_valid)
        mean_rmse_valid = np.mean(rmse_valid) 
        rmse_valid_list.append(rmse_valid)
        mean_rmse_valid_list.append(mean_rmse_valid)
        mean_mae_valid = metrics.mean_absolute_error(y_valid_pred_final, y_valid_true_final)
        mean_mae_valid_list.append(mean_mae_valid)
        
        mae_per_channel = np.zeros(3)
        for i in range(3):
            mae_per_channel[i] = metrics.mean_absolute_error(
                np.array([y[i] for y in y_valid_true_final]), 
                np.array([y[i] for y in y_valid_pred_final])
            )
        
        mape_per_channel = np.zeros(3)
        epsilon = 1e-8
        for i in range(3):  
            mape_per_channel[i] = np.mean(np.abs((np.array([y[i] for y in y_valid_true_final]) - 
                                                   np.array([y[i] for y in y_valid_pred_final])) / 
                                                  (np.array([y[i] for y in y_valid_true_final]) + epsilon))) * 100
        
        mean_mape_valid = np.mean(mape_per_channel)
        mean_mape_valid_list.append(mean_mape_valid)
        
        if mean_mape_valid == np.min(mean_mape_valid_list):
            torch.save(net.state_dict(), model_save_pth1)   
            
        print(f'\n>>> Epoch: {epoch}\n')
        print(f'    RMSE_train: MEAN:{mean_rmse_train:.4f}\n')
        print(f'    RMSE_valid: MEAN:{mean_rmse_valid:.4f}\n')
        print(f'    MAE_valid: MEAN:{mean_mae_valid:.4f}, CL:{mae_per_channel[0]:.4f}, CD:{mae_per_channel[1]:.4f}, CM:{mae_per_channel[2]:.4f}\n')
        print(f'    MAPE_valid: MEAN:{mean_mape_valid:.4f}, CL:{mape_per_channel[0]:.4f}, CD:{mape_per_channel[1]:.4f}, CM:{mape_per_channel[2]:.4f}\n')
        print(f'    RMSE_valid_min: {np.min(mean_rmse_valid_list):.4f}, MAE_valid_min: {np.min(mean_mae_valid_list):.4f}, MAPE_valid_min: {np.min(mean_mape_valid_list):.4f}\n')
       
    return net, mean_rmse_valid_list, mean_mae_valid_list, mean_mape_valid_list

def main():
    model = TCNSEMTL(input_size, output_size2, num_channels=[hidden_size]*levels,  kernel_size=kernel_size, dropout=dropout, reduction_ratio=16)
    folder_name = f'model{model_name}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_save_pth1 = './{}/min_mape.pth'.format(folder_name)
    train(model, train_loader, valid_loader, epochs=epochs, lr=learning_rate, model_save_pth=model_save_pth1)
    

if __name__ == '__main__':
    main()
