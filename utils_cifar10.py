from cifar10models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(num):
    modelNames = ["VGG16", "ResNet18", "PreActResNet18", "GoogLeNet", "DenseNet121", "ResNeXt29_2x64d", "MobileNet", "MobileNetV2", "DPN92", "ShuffleNetG2", "ShuffleNetV2", "EfficientNetB0", "RegNetX_200MF", "SimpleDLA"]
    if num == 0:
        net = VGG("VGG16")
    elif num == 1:
        net = ResNet18()
    elif num == 2:
        net = PreActResNet18()
    elif num == 3:
        net = GoogLeNet()
    elif num == 4:
        net = DenseNet121()
    elif num == 5:
        net = ResNeXt29_2x64d()
    elif num == 6:
        net = MobileNet()
    elif num == 7:
        net = MobileNetV2()
    elif num == 8:
        net = DPN92()
    elif num == 9:
        net = ShuffleNetG2()
    elif num == 10:
        net = ShuffleNetV2(1)
    elif num == 11:
        net = EfficientNetB0()
    elif num == 12:
        net = RegNetX_200MF()
    elif num == 13:
        net = SimpleDLA()

    net = net.to(device)

    return net, modelNames[num]
