from nntoolbox.vision.learner import StyleTransferLearner
from nntoolbox.vision.components import FeatureExtractor, PretrainedModel, ResidualBlock, ResidualBlockPreActivation, ResNeXtBlock, ConvolutionalLayer
from nntoolbox.vision.utils import UnlabelledImageDataset, pil_to_tensor
from nntoolbox.utils import get_device
from torchvision.models import vgg16_bn
from torch import nn
from fastai.vision.models.unet import DynamicUnet
import torch
from torch.utils.data import DataLoader
from PIL import Image

torch.autograd.set_detect_anomaly(True)

data_size = 1000
n_channel = 3
img_dim = (224, 224)
h = 12
w = 12

images = UnlabelledImageDataset("data/", img_dim=img_dim)
style = pil_to_tensor(Image.open("data/trump.jpg"))
content = pil_to_tensor(Image.open("data/trump_2.jpeg").resize(img_dim))
dataloader = DataLoader(images, shuffle=True, batch_size=1)
dataloader_val = DataLoader(images, batch_size=len(images))

feature_extractor = FeatureExtractor(fine_tune=False, device=get_device())

encoder = PretrainedModel(model=vgg16_bn, fine_tune=True)
model = DynamicUnet(encoder=encoder, n_classes=3, y_range=(0, 1))

learner = StyleTransferLearner(
    dataloader, dataloader_val, style, content,
    model, feature_extractor,
    feature_layers={1, 2}, style_layers={1, 2},
    style_weight=1e9, content_weight=1.0, total_variation_weight=0.1, device=get_device(),
)
learner.learn(32, 16, draw=True)