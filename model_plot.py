from sidekick.nn.conv.resnet import ResNet
from sidekick.plot.plot_model import visualize_model

model= ResNet.build(64, 64, 3, 200, (3,5,7),(64, 128, 256, 512))
visualize_model(model, 'resnet.jpg')