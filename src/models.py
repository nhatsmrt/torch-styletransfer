from nntoolbox.vision import *
from .layers import *


__all__ = ['MultipleStyleTransferNetwork', 'GenericDecoder', 'PixelShuffleDecoder', 'SimpleDecoder']


class MultipleStyleTransferNetwork(nn.Module):
    def __init__(self, encoder: FeatureExtractor, decoder:nn.Module, extracted_feature: int):
        super(MultipleStyleTransferNetwork, self).__init__()
        self.adain = AdaIN()
        self.encoder = encoder
        self.decoder = decoder
        self._extracted_feature = extracted_feature

    def forward(self, input: Tensor) -> Tensor:
        return self.decoder(self.style_encode(input))

    def style_encode(self, input: Tensor) -> Tensor:
        return self.adain(self.encode(input)) # t = AdaIn(f(c), f(s))

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input, layers=[self._extracted_feature])

    def decode(self, input: Tensor) -> Tensor:
        return self.decoder(input)

    def set_style(self, style_img: Tensor):
        '''
        :param style_img:
        :return:
        '''
        self.adain.set_style(self.encode(style_img))


class GenericDecoder(nn.Module):
    def __init__(self):
        super(GenericDecoder, self).__init__()
        self.upsample1 = ResizeConvolutionalLayer(in_channels=512, out_channels=128, normalization=nn.Identity)
        self.mid1 = ResidualBlockPreActivation(in_channels=128, normalization=nn.Identity)
        self.upsample2 = ResizeConvolutionalLayer(in_channels=128, out_channels=32, normalization=nn.Identity)
        self.mid2 = ResidualBlockPreActivation(in_channels=32, normalization=nn.Identity)
        self.upsample3 = ResizeConvolutionalLayer(in_channels=32, out_channels=8, normalization=nn.Identity)
        self.mid3 = ResidualBlockPreActivation(in_channels=8, normalization=nn.Identity)
        self.upsample4 = ResizeConvolutionalLayer(
            in_channels=8, out_channels=3,
            normalization=nn.Identity, activation=nn.Identity
        )
        self.op = nn.Sigmoid()

    def forward(self, input, out_h: int=128, out_w: int=128):
        assert out_h >= 16 and out_w >= 16
        upsampled = self.upsample1(input, out_h=out_h // 8, out_w=out_w // 8)
        upsampled = self.mid1(upsampled)
        upsampled = self.upsample2(upsampled, out_h=out_h // 4, out_w=out_w // 4)
        upsampled = self.mid2(upsampled)
        upsampled = self.upsample3(upsampled, out_h=out_h // 2, out_w=out_w // 2)
        upsampled = self.mid3(upsampled)
        upsampled = self.upsample4(upsampled, out_h=out_h, out_w=out_w)
        op = self.op(upsampled)
        return op


class PixelShuffleDecoder(nn.Module):
    def __init__(self):
        super(PixelShuffleDecoder, self).__init__()
        self.upsample1 = PixelShuffleConvolutionLayer(
            in_channels=512, out_channels=128,
            normalization=nn.Identity, upscale_factor=2
        )
        self.mid1 = CustomResidualBlockPreActivation(in_channels=128, normalization=nn.Identity)
        self.upsample2 = PixelShuffleConvolutionLayer(
            in_channels=128, out_channels=32,
            normalization=nn.Identity, upscale_factor=2
        )
        self.mid2 = CustomResidualBlockPreActivation(in_channels=32, normalization=nn.Identity)
        self.upsample3 = PixelShuffleConvolutionLayer(
            in_channels=32, out_channels=3, activation=nn.Identity,
            normalization=nn.Identity, upscale_factor=2
        )
        # self.pad = nn.ReplicationPad2d(1)
        # self.conv = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=0)
        self.op = nn.Sigmoid()


    def forward(self, input):
        upsampled = self.upsample1(input)
        upsampled = self.mid1(upsampled)
        upsampled = self.upsample2(upsampled)
        upsampled = self.mid2(upsampled)
        upsampled = self.upsample3(upsampled)
        op = self.op(upsampled)
        return op


class PixelShuffleDecoderV2(nn.Sequential):
    def __init__(self):
        super(PixelShuffleDecoderV2, self).__init__(
            PixelShuffleConvolutionLayer(
                in_channels=512, out_channels=128,
                normalization=nn.Identity, upscale_factor=2
            ),
            CustomResidualBlockPreActivation(in_channels=128, normalization=nn.Identity),
            PixelShuffleConvolutionLayer(
                in_channels=128, out_channels=32,
                normalization=nn.Identity, upscale_factor=2
            ),
            CustomResidualBlockPreActivation(in_channels=32, normalization=nn.Identity),
            PixelShuffleConvolutionLayer(
                in_channels=32, out_channels=3, activation=nn.Identity,
                normalization=nn.Sigmoid, upscale_factor=2
            )
        )


class SimpleDecoder(nn.Sequential):
    def __init__(self):
        super(SimpleDecoder, self).__init__(
            ReflectionPaddedConv(512, 256),
            nn.Upsample(scale_factor=2),

            ReflectionPaddedConv(256, 256),
            ReflectionPaddedConv(256, 256),
            ReflectionPaddedConv(256, 256),

            ReflectionPaddedConv(256, 128),
            nn.Upsample(scale_factor=2),
            ReflectionPaddedConv(128, 128),

            ReflectionPaddedConv(128, 64),
            nn.Upsample(scale_factor=2),
            ReflectionPaddedConv(64, 64),

            ReflectionPaddedConv(64, 3, activation=nn.Sigmoid)
        )


class MultipleStyleUNet(nn.Module):
    def __init__(self, encoder: FeatureExtractorSequential, extracted_feature: int):
        super(MultipleStyleUNet, self).__init__()
        self.adain = AdaIN()
        self.encoder = encoder
        self.unet = DynamicUnetV2(encoder, n_classes=3, y_range=(0, 1), normalization=nn.Identity)
        self.encoder.default_extracted_feature = extracted_feature

    def forward(self, input: Tensor) -> Tensor:
        return self.decode(self.style_encode(input))

    def style_encode(self, input: Tensor) -> Tensor:
        return self.adain(self.encode(input)) # t = AdaIn(f(c), f(s))

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        return self.unet.get_decoder()(input)

    def set_style(self, style_img: Tensor):
        '''
        :param style_img:
        :return:
        '''
        self.adain.set_style(self.encode(style_img))
