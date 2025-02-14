import torch
import torch.nn as nn

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)


class CBR(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class maxpool3d_r(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, stride1, stride2):
        super(maxpool3d_r, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size2, stride=stride2)
        
    def forward(self, x):
        x = self.maxpool1(x)
        x = x.transpose(1,3)
        x = self.maxpool2(x)
        x = x.transpose(1,3)
        return x
        

class LPRNet(nn.Module):
    def __init__(self, class_num):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool2d(kernel_size=3, stride=1),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            # CBR(128, 64),
            maxpool3d_r(kernel_size1=(3, 3), stride1=(1, 2), kernel_size2=(1, 1), stride2=(1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            # CBR(256, 64),
            maxpool3d_r(kernel_size1=(3,3), stride1=(1,2), kernel_size2=(1, 1), stride2=(1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits