import torch.nn as nn


class Discriminator(nn.Module):
    """
    This is a discriminator for image colorization task
    """

    def __init__(self, image_channels=3, nf=64, n_blocks=3, p=0.15):
        super(Discriminator, self).__init__()

        # Initial layer
        layers = [
            nn.Conv2d(image_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p / 2),
        ]

        # Intermediate layers
        for i in range(n_blocks):
            layers += [
                nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(p),
            ]
            nf *= 2

        # Final layers
        layers += [
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1, bias=False),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
