from nntoolbox.vision.learner import StyleTransferLearner
from nntoolbox.vision.components import *
from nntoolbox.vision.utils import UnlabelledImageDataset, pil_to_tensor
from nntoolbox.utils import get_device
from torchvision.models import vgg16_bn
from torch.nn import Sequential, InstanceNorm2d
from fastai.vision.models.unet import DynamicUnet
from torch.utils.data import DataLoader
from PIL import Image

print(vgg16_bn().features)

data_size = 1000
n_channel = 3
img_dim = (128, 128)
h = 12
w = 12


images = UnlabelledImageDataset("MiniCOCO/128/", img_dim=img_dim)
train_size = int(0.8 * len(images))
val_size = len(images) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(images, [train_size, val_size])

style = pil_to_tensor(Image.open("la_muse.jpg"))
dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
dataloader_val = DataLoader(val_dataset, shuffle=True, batch_size=16)

feature_extractor = FeatureExtractor(fine_tune=False, device=get_device())

# encoder = PretrainedModel(model=vgg16_bn, fine_tune=True)
encoder = Sequential(
    ConvolutionalLayer(3, 16, padding=1, stride=2, normalization=InstanceNorm2d),
    ResidualBlockPreActivation(16, normalization=InstanceNorm2d),
    ConvolutionalLayer(16, 32, padding=1, stride=2, normalization=InstanceNorm2d),
    ResidualBlockPreActivation(32, normalization=InstanceNorm2d),
    ConvolutionalLayer(32, 64, padding=1, stride=2, normalization=InstanceNorm2d),
    ResidualBlockPreActivation(64, normalization=InstanceNorm2d),
    ConvolutionalLayer(64, 128, padding=1, stride=2, normalization=InstanceNorm2d),
    ResidualBlockPreActivation(128, normalization=InstanceNorm2d),
    ConvolutionalLayer(128, 256, padding=1, stride=2, normalization=InstanceNorm2d),
)
model = DynamicUnet(encoder=encoder, n_classes=3, y_range=(0, 1))

learner = StyleTransferLearner(
    dataloader, dataloader_val, style,
    model, feature_extractor,
    style_layers={5, 12, 22, 32, 42}, content_layers={32},
    style_weight=1e7, content_weight=1.0, total_variation_weight=0.1, device=get_device(),
)
learner.learn(100, print_every=100, eval_every=1, draw=True)