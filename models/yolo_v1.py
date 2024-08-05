"""
Implementation of Yolo (v1) architecture with modification in added BatchNorm
"""
import torch
import torch.nn as nn

'''
Architecture config is adopted from paper of Yolo where 
(k, f, s, p) ---> kernel, filter, stride, padding 
M -----> Max pooling of string 2x2 and kernel of 2x2
'''
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size= kernel_size, stride= stride,
                              padding= padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))

class Yolov1(nn.Module) :
    def __init__(self, in_channels=3, *args, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture :
            if type(x) == tuple :
                layers += [CNNBlock(
                    in_channels,
                    out_channels=x[1],
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )]

                in_channels = x[1]

            elif type(x) == str :
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list :
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Int

                for _ in range(num_repeats) :
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S,B,C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S* S * (C + B * 5)) # In paper last layer (S, S, 30) where C + B*5 = 30
                                                         # Details is as per paper C : Number of classes
                                                         # B : (x, y, w, h) and probability score for class
                                                         # We will have two bounding box there fore B = 2 ,total B x 5 = 10
        )




def test(split_size = 7, num_boxes = 2, num_classes = 20) :
    model = Yolov1(split_size = split_size, num_boxes = num_boxes, num_classes = num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)


