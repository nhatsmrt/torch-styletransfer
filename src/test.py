from nntoolbox.vision.learner import StyleTransferLearner
from nntoolbox.vision.components import *
from nntoolbox.vision.utils import UnlabelledImageDataset, UnlabelledImageListDataset, pil_to_tensor
from nntoolbox.utils import get_device, compute_num_batch, MultiRandomSampler
from torchvision.models import vgg16_bn, vgg19_bn, vgg19
from torch.nn import Sequential, InstanceNorm2d
from fastai.vision.models.unet import DynamicUnet
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
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


def run_test_multiple(
        style_weight=10.0, content_weight=1.0, total_variation_weight=0.1,
        n_iter=80000, print_every=1000, style_path="./data/train_9/"
):
    from nntoolbox.vision.learner import MultipleStylesTransferLearner
    from nntoolbox.vision.utils import UnlabelledImageDataset, PairedDataset, UnlabelledImageListDataset
    from nntoolbox.utils import get_device
    from nntoolbox.callbacks import Tensorboard, MultipleMetricLogger,\
        ModelCheckpoint, ToDeviceCallback, ProgressBarCB, MixedPrecisionV2
    from src.models import GenericDecoder, MultipleStyleTransferNetwork, PixelShuffleDecoder, MultipleStyleUNet
    from torchvision.models import vgg19
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, RandomCrop
    from torch.optim import Adam

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print("Begin creating dataset")

    content_images = UnlabelledImageListDataset("MiniCOCO/256/")
    style_images = UnlabelledImageListDataset(style_path, transform=Compose(
        [
            Resize(512),
            RandomCrop((256, 256))
        ]
    ))

    # img_dim = (128, 128)
    # # content_images = UnlabelledImageDataset("MiniCOCO/128/", img_dim=img_dim)
    # # style_images = UnlabelledImageDataset(style_path, img_dim=img_dim)
    #
    #
    # content_images = UnlabelledImageListDataset("data/", img_dim=img_dim)
    # style_images = UnlabelledImageListDataset("data/train_9/", img_dim=img_dim)

    print("Begin splitting data")
    train_size = int(0.8 * len(content_images))
    val_size = len(content_images) - train_size
    train_content, val_content = torch.utils.data.random_split(content_images, [train_size, val_size])

    train_size = int(0.8 * len(style_images))
    val_size = len(style_images) - train_size
    train_style, val_style = torch.utils.data.random_split(style_images, [train_size, val_size])

    train_dataset = PairedDataset(train_content, train_style)
    val_dataset = PairedDataset(val_content, val_style)

    # train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=8, drop_last=True)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=8)
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=8)


    print("Begin creating data dataloaders")
    dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
    dataloader_val = DataLoader(val_dataset, sampler=val_sampler, batch_size=8)
    # print(len(dataloader))

    print("Creating models")
    feature_extractor = FeatureExtractor(
        model=vgg19(True), fine_tune=False,
        mean=mean, std=std,
        device=get_device()
    )
    print("Finish creating feature extractor")
    # decoder = GenericDecoder()
    # decoder = PixelShuffleDecoder()
    print("Finish creating decoder")
    # model = MultipleStyleTransferNetwork(
    #     encoder=feature_extractor,
    #     decoder=decoder,
    #     extracted_feature=20
    # )
    # model = MultipleStyleTransferNetwork(
    #     encoder=FeatureExtractor(
    #         model=vgg19(True), fine_tune=False,
    #         mean=mean, std=std,
    #         device=get_device()
    #     ),
    #     decoder=decoder,
    #     extracted_feature=20
    # )
    model = MultipleStyleUNet(
        encoder=FeatureExtractorSequential(
            model=vgg19(True), fine_tune=False,
            mean=mean, std=std, last_layer=20
        ),
        extracted_feature=20
    )
    optimizer = Adam(model.parameters())
    learner = MultipleStylesTransferLearner(
        dataloader, dataloader_val,
        model, feature_extractor, optimizer=optimizer,
        style_layers={1, 6, 11, 20}, total_variation_weight=total_variation_weight,
        style_weight=style_weight, content_weight=content_weight, device=get_device()
    )
    callbacks = [
        ToDeviceCallback(),
        # MixedPrecisionV2(),
        Tensorboard(every_iter=1000, every_epoch=1),
        MultipleMetricLogger(
            iter_metrics=["content_loss", "style_loss", "total_variation_loss", "loss"], print_every=print_every
        ),
        ModelCheckpoint(learner=learner, save_best_only=False, filepath='weights/model.pt'),
        # ProgressBarCB(range(print_every))
    ]
    learner.learn(n_iter=n_iter, callbacks=callbacks, eval_every=print_every)
