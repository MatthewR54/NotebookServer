import torch.nn as nn

class UniversalFakeDetectv2(nn.Module):
    # Alternate input dimensions for other backbones:
    # input_dim = 1024 for DINOv2 ViT-L, 1536 for ViT-g.
    def __init__(self, input_dim=768, hidden_dim=384, num_classes=1):
        super(UniversalFakeDetectv2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # If x is the output from DINOv2 then its shape is [batch_size, num_patches, embedding_dim]
        # and we have to convert shape to [batch_size, embedding_dim] so we average over patches:
        # x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)
    