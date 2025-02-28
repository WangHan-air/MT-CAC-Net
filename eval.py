import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import os

# Parameters
rand_seed = 42
model_name = 'CAC-Net'
device = 'cuda:0'
input_size= 4
output_size = 1
kernel_size = 3
levels = 3
hidden_size = 64
learning_rate = 0.001
batch_size = 64
dropout = 0.3

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
    # for i in range(0, 165):
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

test_data = pd.read_csv("ZDYdata-test.CSV")
test_x, test_y, test_len = dataprocess(test_data.iloc[0:3706,0:8])
test_x, test_y = slidewindow(test_x, test_y, test_len)
print('test_x_shape: {}  test_y_shape: {}'.format(test_x.shape, test_y.shape))
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True) 

# Model
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
        output = self.tcnse(x.transpose(1, 2)).transpose(1, 2)  # TCN的输出是[batch_size, time_steps, channels]
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

# eval
def eval(test_loader, hidden_size, learning_rate, batch_size, dropout):
    folder_name = f'model{model_name}'
    model_save_pth = './{}/min_mape.pth'.format(folder_name)
    
    net = TCNSEMTL(input_size=input_size, output_size=output_size, num_channels=[hidden_size]*levels, kernel_size=kernel_size, dropout=dropout)
    net.to(device)

    # Load model parameters
    net.load_state_dict(torch.load(model_save_pth,  map_location=torch.device(device)))
    net.eval()

    criterion = nn.MSELoss(reduction='none')       

    mse_test = np.zeros(3)  
    y_test_true_final = []
    y_test_pred_final = []
    cnt = 0
    for batch_idx, (x_input_test, y_true_test) in enumerate(test_loader):
        x_input_test = x_input_test.to(device)
        y_true_test = y_true_test.to(device)
        y_test_pred = net(x_input_test)

        y_true_test = y_true_test.squeeze(1)
        loss_test = criterion(y_test_pred, y_true_test) 

        mse_test += loss_test.detach().sum(dim=0).cpu().numpy()  
        
        y_test_true_final.extend(y_true_test.data.detach().cpu().numpy())
        y_test_pred_final.extend(y_test_pred.data.detach().cpu().numpy())

        cnt += x_input_test.size(0)

        mse_test /= cnt
        rmse_test = np.sqrt(mse_test)
        mean_rmse_test = np.mean(rmse_test) 

        mean_mae_test = metrics.mean_absolute_error(y_test_pred_final, y_test_true_final)
        mae_per_channel = np.zeros(3)
        for i in range(3):
            mae_per_channel[i] = metrics.mean_absolute_error(
                np.array([y[i] for y in y_test_true_final]), 
                np.array([y[i] for y in y_test_pred_final])
            )
            
        mape_per_channel = np.zeros(3)
        epsilon = 1e-8
        for i in range(3):  
            mape_per_channel[i] = np.mean(np.abs((np.array([y[i] for y in y_test_true_final]) - 
                                                    np.array([y[i] for y in y_test_pred_final])) / 
                                                    (np.array([y[i] for y in y_test_true_final]) + epsilon))) * 100

        mean_mape_test = np.mean(mape_per_channel)

        r2_test = metrics.r2_score(y_test_true_final, y_test_pred_final)

        print(f"Hidden size: {hidden_size}, Learning rate: {learning_rate}, Batch size: {batch_size}, Dropout: {dropout}, Model_save_pth:{model_save_pth}")
        print(f'    RMSE_test: MEAN:{mean_rmse_test:.4f}, CL:{rmse_test[0]:.4f}, CD:{rmse_test[1]:.4f}, CM:{rmse_test[2]:.4f}\n')
        print(f'    MAE_test: MEAN:{mean_mae_test:.4f}, CL:{mae_per_channel[0]:.4f}, CD:{mae_per_channel[1]:.4f}, CM:{mae_per_channel[2]:.4f}\n')
        print(f'    MAPE_test: MEAN:{mean_mape_test:.4f}, CL:{mape_per_channel[0]:.4f}, CD:{mape_per_channel[1]:.4f}, CM:{mape_per_channel[2]:.4f}\n')
        print(f'R2_test:{r2_test:.4f}\n')


def main():
    eval(test_loader, hidden_size = hidden_size, learning_rate = learning_rate, batch_size = batch_size, dropout = dropout)

if __name__ == '__main__':
    main()
        
