import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio,
        activation=nn.ReLU,
        normalization=True,
    ):
        super(Downsample, self).__init__()
        if kernel_size - ratio < 2:
            raise ValueError("K - S must > 2")
        padding = int((kernel_size - ratio) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, ratio, padding)
        if normalization:
            self.bn = nn.BatchNorm1d(out_channels)
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


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio,
        activation=nn.ReLU,
        normalization=True,
    ):
        super(Upsample, self).__init__()
        if kernel_size - ratio < 2:
            raise ValueError("K - S must > 2")
        padding = int((kernel_size - ratio) / 2)

        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, ratio, padding
        )
        if normalization:
            self.bn = nn.BatchNorm1d(out_channels)
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


class LSTM1DCAE(nn.Module):
    def __init__(self, in_channels=2):
        super(LSTM1DCAE, self).__init__()

        self.encoder = nn.Sequential(
            Downsample(in_channels, 32, 256, 16, nn.ReLU, True),
            Downsample(32, 64, 32, 16, nn.ReLU, True),
            Downsample(64, 64, 16, 2, nn.ReLU, True),
            Downsample(64, 64, 8, 2, nn.ReLU, True),
        )

        self.lstm1 = torch.nn.LSTM(64, 64, batch_first=True)
        self.lstm2 = torch.nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = torch.nn.LSTM(64, 64, batch_first=True)

        self.decoder = nn.Sequential(
            Upsample(64, 64, 8, 2, nn.ReLU, True),
            Upsample(64, 64, 16, 2, nn.ReLU, True),
            Upsample(64, 32, 32, 16, nn.ReLU, True),
            Upsample(32, in_channels, 256, 16, False, False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x, (h, c))
        x, (h, c) = self.lstm3(x, (h, c))
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x = self.decoder(x)

        return x

    def get_latent_feature(self, x):
        x = self.encoder(x)
        x = torch.permute(x, (0, 2, 1)).contiguous()
        x, (_, _) = self.lstm1(x)

        return x


class Simple1DCAE(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32):
        super(Simple1DCAE, self).__init__()

        self.encoder = nn.Sequential(
            Downsample(in_channels, 32, 256, 16, nn.ReLU, True),
            Downsample(32, 64, 32, 16, nn.ReLU, True),
            Downsample(64, 64, 16, 2, nn.ReLU, True),
            Downsample(64, 64, 8, 2, nn.ReLU, True),
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            torch.nn.Linear(1600, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1600),
        )

        self.decoder = nn.Sequential(
            Upsample(64, 64, 8, 2, nn.ReLU, True),
            Upsample(64, 64, 16, 2, nn.ReLU, True),
            Upsample(64, 32, 32, 16, nn.ReLU, True),
            Upsample(32, in_channels, 256, 16, False, False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = x.view(-1, 64, 25)
        x = self.decoder(x)

        return x

    def get_latent_feature(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.mlp[0](x)
        x = self.mlp[1](x)
        x = self.mlp[2](x)

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
