from nntoolbox.vision.learner import StyleTransferLearner
from nntoolbox.vision.components import *
from nntoolbox.vision.utils import UnlabelledImageDataset, UnlabelledImageListDataset, pil_to_tensor
from nntoolbox.utils import get_device, compute_num_batch, MultiRandomSampler
from torchvision.models import vgg16_bn, vgg19_bn, vgg19
from torch.nn import Sequential, InstanceNorm2d
from fastai.vision.models.unet import DynamicUnet
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from PIL import Image
from nntoolbox.vision.components.unet import DynamicUnetV2

from nntoolbox.vision.learner import MultipleStylesTransferLearner
from nntoolbox.vision.utils import UnlabelledImageDataset, PairedDataset, UnlabelledImageListDataset
from nntoolbox.utils import get_device
from src.models import GenericDecoder, MultipleStyleTransferNetwork, PixelShuffleDecoder
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop
from torch.optim import Adam

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



print("Creating models")
feature_extractor = FeatureExtractorSequential(
    model=vgg19(True), fine_tune=False,
    last_layer=20, mean=mean, std=std
)
# print("Finish creating feature extractor")
# # decoder = GenericDecoder()
# decoder = PixelShuffleDecoder()
# print("Finish creating decoder")
# model = MultipleStyleTransferNetwork(
#     encoder=FeatureExtractor(
#         model=vgg19(True), fine_tune=False,
#         mean=mean, std=std,
#         device=get_device()
#     ),
#     decoder=decoder,
#     extracted_feature=20
# )

input = torch.rand(3, 3, 128, 128)

unet = DynamicUnetV2(feature_extractor, n_classes=3, y_range=(0, 1))

from fastai.callbacks.hooks import model_sizes
# print(model_sizes(unet, (128, 128)))
# unet(input)
# print(unet.sfs.hooks[0].stored.shape)
# print(unet(input).shape)


decoder = unet.get_decoder()
encoded = feature_extractor(input)
print(unet.sfs.hooks[2].stored.shape)
decoded = decoder(encoded)

from src.models import MultipleStyleUNet
from nntoolbox.utils import load_model
model = MultipleStyleUNet(
    encoder=FeatureExtractorSequential(
        model=vgg19(True), fine_tune=False,
        mean=mean, std=std, last_layer=20
    ),
    extracted_feature=20
)
load_model(model, "weights/model.pt")
