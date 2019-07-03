from .unet import *
from nntoolbox.vision import *


__all__ = ['MultipleStyleTransferNetwork', 'GenericDecoder']


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
        self.adain.set_style(self.encoder(style_img))


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