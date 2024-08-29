import torch
import torch.nn as nn

class Downsample2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio,
        activation=nn.ReLU,
        normalization=True,
    ):
        super(Downsample2D, self).__init__()
        if kernel_size - ratio < 2:
            raise ValueError("K - S must > 2")
        padding = int((kernel_size - ratio) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, ratio, padding)
        if normalization:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        if activation:
            self.act = activation()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class Upsample2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio,
        activation=nn.ReLU,
        normalization=True,
    ):
        super(Upsample2D, self).__init__()
        if kernel_size - ratio < 2:
            raise ValueError("K - S must > 2")
        padding = int((kernel_size - ratio) / 2)

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, ratio, padding
        )
        if normalization:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        if activation:
            self.act = activation()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class Simple2DSTFTCAE(nn.Module):
    def __init__(self, in_channels=2):
        super(Simple2DSTFTCAE, self).__init__()

        self.encoder = nn.Sequential(
            Downsample2D(in_channels, 16, 32, 2, nn.ReLU, True),
            Downsample2D(16, 32, 16, 2, nn.ReLU, True),
            Downsample2D(32, 64, 8, 2, nn.ReLU, True),
            Downsample2D(64, 64, 4, 2, nn.ReLU, True),
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            torch.nn.Linear(4096, 32),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            torch.nn.Linear(32, 4096),
        )

        self.decoder = nn.Sequential(
            Upsample2D(64, 64, 4, 2, nn.ReLU, True),
            Upsample2D(64, 32, 8, 2, nn.ReLU, True),
            Upsample2D(32, 16, 16, 2, nn.ReLU, True),
            Upsample2D(16, in_channels, 32, 2, False, False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = x.view(-1, 64, 8, 8)
        x = self.decoder(x)

        return x


class LinearAE(nn.Module):
    def __init__(self, in_channels=4):
        super(LinearAE, self).__init__()
        # 인코더: 4차원 입력을 여러 은닉층을 거쳐 3차원 잠재 공간으로 변환
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 10),  # 첫 번째 은닉층
            nn.LeakyReLU(0.01),
            nn.Linear(10, 6),  # 두 번째 은닉층
            nn.LeakyReLU(0.01),
            nn.Linear(6, 3),    # 잠재 공간
            nn.Tanh()
        )
        
        # 디코더: 3차원 잠재 공간을 여러 은닉층을 거쳐 4차원으로 복원
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),   # 첫 번째 은닉층
            nn.LeakyReLU(0.01),
            nn.Linear(6, 10),  # 두 번째 은닉층
            nn.LeakyReLU(0.01),
            nn.Linear(10, in_channels),  # 출력층
            nn.Tanh()       # 출력 값의 범위를 0~1로 제한
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded