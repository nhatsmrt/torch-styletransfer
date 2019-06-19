from nntoolbox.vision.learner import StyleTransferLearner
from nntoolbox.vision.components import *
from nntoolbox.vision.utils import UnlabelledImageDataset, pil_to_tensor
from nntoolbox.utils import get_device
from torchvision.models import vgg16_bn, vgg19_bn
from torch.nn import Sequential, InstanceNorm2d
from fastai.vision.models.unet import DynamicUnet
from torch.utils.data import DataLoader
from PIL import Image
from .unet import CustomDynamicUnet


def run_test(
        encoder=None, style_weight=1e5, content_weight=1.0, total_variation_weight=1e-4,
        n_epoch=100, print_every=100, eval_every=1, batch_size=4,
        style_layers={0, 7, 14, 27, 40}, content_layers={30}, train_ratio=0.95, img_dim=(128, 128),
        style_path="mouse.png", save_path="weights/model.pt"
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = UnlabelledImageDataset("MiniCOCO/128/", img_dim=img_dim)
    train_size = int(train_ratio * len(images))
    val_size = len(images) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(images, [train_size, val_size])

    style = pil_to_tensor(Image.open(style_path).convert("RGB"))
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dataloader_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    feature_extractor = FeatureExtractor(
        model=vgg19_bn, fine_tune=False,
        mean=mean, std=std,
        device=get_device()
    )
    if encoder is None:
        encoder = Sequential(
            ConvolutionalLayer(3, 16, padding=1, stride=2, normalization=InstanceNorm2d),
            SEResidualBlockPreActivation(16, normalization=InstanceNorm2d),
            ConvolutionalLayer(16, 32, padding=1, stride=2, normalization=InstanceNorm2d),
            SEResidualBlockPreActivation(32, normalization=InstanceNorm2d),
            ConvolutionalLayer(32, 64, padding=1, stride=2, normalization=InstanceNorm2d),
            SEResidualBlockPreActivation(64, normalization=InstanceNorm2d),
            ConvolutionalLayer(64, 128, padding=1, stride=2, normalization=InstanceNorm2d),
            SEResidualBlockPreActivation(128, normalization=InstanceNorm2d),
            ConvolutionalLayer(128, 256, padding=1, stride=2, normalization=InstanceNorm2d),
        )
    model = CustomDynamicUnet(encoder=encoder, normalization=InstanceNorm2d, n_classes=3, y_range=(0, 1), blur=True)
    print(model)

    learner = StyleTransferLearner(
        dataloader, dataloader_val, style,
        model, feature_extractor,
        style_layers=style_layers, content_layers=content_layers,
        style_weight=style_weight, content_weight=content_weight,
        total_variation_weight=total_variation_weight, device=get_device()
    )
    learner.learn(n_epoch=n_epoch, print_every=print_every, eval_every=eval_every, draw=True, save_path=save_path)