
import torch
import torch.nn as nn


class FrequencyCNN(nn.Module):

    def __init__(self, output_features=128):
        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Reduce spatial dimensions
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Convert CNN output into a feature vector
        self.classifier = nn.Linear(128, output_features)

    def forward(self, x):

        x = self.features(x)

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)

        # Return learned frequency features
        x = self.classifier(x)

        return x
