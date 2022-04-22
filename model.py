from turtle import forward
import torch

class ModelA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 9, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 9, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 7, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 7, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, 7, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 512, 7, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 1024, 7, stride=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1024, 512, 7, stride=3),
            torch.nn.ConvTranspose1d(512, 256, 7, stride=3),
            torch.nn.ConvTranspose1d(256, 128, 7, stride=3),
            torch.nn.ConvTranspose1d(128, 64, 7, stride=3),
            torch.nn.ConvTranspose1d(64, 32, 7, stride=3),
            torch.nn.ConvTranspose1d(32, 16, 9, stride=4),
            torch.nn.ConvTranspose1d(16, 1, 9, stride=4)
        )

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)

        return output