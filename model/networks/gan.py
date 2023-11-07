import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Conv2d(num_filters * 8, num_filters * 16, kernel_size=3, stride=2, padding=1),
            # # nn.BatchNorm2d(num_filters * 16),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_filters * 8, 1, kernel_size=3, stride=1, padding=1),
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            # nn.Linear(num_filters * 16, 1),
            nn.Sigmoid()
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.style_model = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.final_layer = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d(1),
        #     # nn.Flatten(),
        #     # nn.Linear(640, 1),
        #     nn.Conv2d(1280, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x, target):
        feat = self.model(x)
        # style = self.style_model(target)
        # adp = nn.AdaptiveMaxPool2d(1)
        # feat = adp(feat)
        # style = adp(style)
        # final = torch.cat((feat, style), dim=1)
        # return self.final_layer(final)
        return feat

