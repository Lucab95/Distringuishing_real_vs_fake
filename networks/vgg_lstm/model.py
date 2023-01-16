import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchfile


class EmotionLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EmotionLSTM, self).__init__()

        self.lstm_features = 64

        self.lstm = nn.LSTM(num_features, self.lstm_features, 1, batch_first=True)

        self.fc = nn.Linear(self.lstm_features, num_classes)

    def forward(self, x):
        r_out, _ = self.lstm(x)

        x = F.dropout(r_out[:, -1, :], 0.25, self.training)

        out = self.fc(x)

        return out

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x
        #x = F.dropout(x, 0.5, self.training)
        #return self.fc7(x)

class VGGLSTM(nn.Module):
    def __init__(self, num_classes):
        super(VGGLSTM, self).__init__()


        # self.backbone = VGG_16()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True, progress=True)
        self.backbone.classifier[6] = nn.Linear(4096, 4096)
        print(self.backbone)
        self.emotion = EmotionLSTM(4096, num_classes)

    def forward(self, x):
        bs, t, c, h, w = x.shape

        c_in = x.view(bs*t, c, h, w)

        x = self.backbone(c_in)
        x = x.view(x.size(0), -1)

        r_in = x.view(bs, t, -1)

        out = self.emotion(r_in)

        return out

