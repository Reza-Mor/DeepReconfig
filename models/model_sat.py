import torch
import torch.nn as nn
import torch.nn.functional as F

class SATLayer(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fmv_pos = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmc_pos = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmv_neg = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmc_neg = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fuv = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fuc = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )

    def message(self, H_c, H_v, A_pos, A_neg):
        message_v = torch.matmul(A_pos, self.fmv_pos(H_c)) + torch.matmul(A_neg, self.fmv_neg(H_c))
        message_c = torch.matmul(torch.transpose(A_pos), self.fmc_pos(H_v)) + torch.matmul(torch.transpose(A_neg), self.fmc_neg(H_v))
        
        return message_v, message_c
    
    def update(self, h, m):
        message_v, message_c = m
        H_v, H_c = h
        return (self.fuv(torch.cat((H_v, message_v), dim=1)), self.fuc(torch.cat((H_c, message_c), dim=1)))

    def forward(self, h, data):
        m = self.message(h, data)
        H_v, H_c = self.update(h, m)
        return H_v, H_c
    

class SATGNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        stats = False

        self.conv1 = SATLayer(input_size, hidden_size)
        self.bn1_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn1_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.conv2 = SATLayer(hidden_size, hidden_size)
        self.bn2_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn2_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
        )

    def forward(self, H_c, H_v, A_pos, A_neg):

        H_v, H_c = self.conv1(H_c, H_v, A_pos, A_neg)
        H_v, H_c = self.bn1_0(H_v), self.bn1_1(H_c)

        H_v, H_c = self.conv2(H_c, H_v, A_pos, A_neg)
        H_v, H_c = self.bn2_0(H_v), self.bn2_1(H_c)

        aggregation = torch.cat([H_v.mean(1), H_c.mean(1)], dim = 1)
        output = self.decoder(aggregation)
        return output
