# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
from pathlib import Path
import typing as tp
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

import quantization as qt
import modules as m
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url
from modules.classifier import ReversalClassifier

ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class MILoss(nn.Module):
    def __init__(self, dim_e_f, dim_e_p):
        super(MILoss, self).__init__()
        self.mu_layer = nn.Linear(dim_e_f, dim_e_p)
        self.logvar_layer = nn.Linear(dim_e_f, dim_e_p)

    def forward(self, e_f, e_p):
        e_f = e_f.permute(0, 2, 1)
        e_p = e_p.permute(0, 2, 1)
        mu = self.mu_layer(e_f)  
        logvar = self.logvar_layer(e_f)  

        log_p_pos = -0.5 * (logvar + (e_p - mu) ** 2 / torch.exp(logvar))  # positive
        mi_upper_bound = log_p_pos.mean()

        e_p_shuffled = e_p[torch.randperm(e_p.size(0))]  # negative
        log_p_neg = -0.5 * (logvar + (e_p_shuffled - mu) ** 2 / torch.exp(logvar))
        mi_upper_bound -= log_p_neg.mean()  

        return mi_upper_bound

class LMModel(nn.Module):
    """Language Model to estimate probabilities of each codebook entry.
    We predict all codebooks in parallel for a given time step.

    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
    """
    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args:
            indices (torch.Tensor): indices from the previous time step. Indices
                should be 1 + actual index in the codebook. The value 0 is reserved for
                when the index is missing (i.e. first time step). Shape should be
                `[B, n_q, T]`.
            states: state for the streaming decoding.
            offset: offset of the current time step.

        Returns a 3-tuple `(probabilities, new_states, new_offset)` with probabilities
        with a shape `[B, card, n_q, T]`.

        """
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self,
                 encoder: m.SEANetEncoder_scale,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels: int,
                 checkpoint_path: str = "/data2/junchuan/EnCodec_Finetune/news_LibriTTS/batch5_cut50000_epoch90.pth",
                 normalize: bool = False,
                 segment: tp.Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'unset',
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 train_quantization: bool = False):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.f0_embedding = nn.Embedding(num_embeddings=256, embedding_dim=256)
        self.uv_embedding = nn.Embedding(num_embeddings=2, embedding_dim=256)
        self.prosody = segment / 40
        self.target = segment / 320
        self.overlap = overlap
        self.f0_classifier = ReversalClassifier(input_dim=256, hidden_dim=384, output_dim=256)
        self.uv_classifier = ReversalClassifier(input_dim=256, hidden_dim=384, output_dim=2)
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.mi_loss = MILoss(dim_e_f=256, dim_e_p=256)
        
        for param in self.quantizer.parameters():
            param.requires_grad = False
        # self.checkpoint_path = checkpoint_path
        # self.checkpoint = torch.load(self.checkpoint_path, map_location=device)
        # encoder_state_dict = {k.replace("encoder.", ""): v for k, v in self.checkpoint.items() if k.startswith("encoder.")}
        # decoder_state_dict = {k.replace("decoder.", ""): v for k, v in self.checkpoint.items() if k.startswith("decoder.")}
        # quantizer_state_dict = {k.replace("quantizer.", ""): v for k, v in self.checkpoint.items() if k.startswith("quantizer.")}
        # self.encoder.load_state_dict(encoder_state_dict)
        # self.decoder.load_state_dict(decoder_state_dict)
        # self.quantizer.load_state_dict(quantizer_state_dict)
        
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        self.train_quantization = train_quantization
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))
    
    @property
    def prosody_length(self) -> tp.Optional[int]:
        if self.prosody is None:
            return None
        return int(self.prosody * self.sample_rate)

    @property
    def prosody_stride(self) -> tp.Optional[int]:
        prosody_length = self.prosody_length
        if prosody_length is None:
            return None
        return max(1, int((1 - self.overlap) * prosody_length))
    
    @property
    def target_length(self) -> tp.Optional[int]:
        if self.target is None:
            return None
        return int(self.target * self.sample_rate)

    @property
    def target_stride(self) -> tp.Optional[int]:
        target_length = self.target_length
        if target_length is None:
            return None
        return max(1, int((1 - self.overlap) * target_length))

    def encode(self, x: torch.Tensor, f0: torch.Tensor, uv: torch.Tensor, tgt, train_stage) -> tp.List[EncodedFrame]:
        """Given a tuple of tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """

        # assert audio input
        assert x.dim() == 3
        _, channels, x_length = x.shape
        assert 0 < channels <= 2

        # assert f0 and uv input
        assert f0.dim() == 3, uv.dim() == 3
        _, channels, pro_length = f0.shape
        assert 0 < channels <= 2
        _, channels, uv_length = uv.shape
        assert 0 < channels <= 2
        _, channels, tgt_length = tgt.shape
        assert 0 < channels <= 128

        segment_length = self.segment_length
        if segment_length is None:
            segment_length = x_length
            stride = x_length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None
        
        prosody_length = self.prosody_length
        if prosody_length is None:
            prosody_length = pro_length
            prosody_stride = pro_length
        else:
            prosody_stride = self.prosody_stride  # type: ignore
            assert prosody_stride is not None
        
        target_length = self.target_length
        if target_length is None:
            target_length = tgt_length
            target_stride = tgt_length
        else:
            target_stride = self.target_stride  # type: ignore
            assert target_stride is not None
        
        encoded_f0: tp.List[EncodedFrame] = []
        # print("length:", length, "stride:", stride)
        for offset in range(0, pro_length, prosody_stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = f0[:, :, offset: offset + prosody_length]
            encoded_f0.append(self._encode_prosody(frame, self.f0_embedding))
        encoded_uv: tp.List[EncodedFrame] = []
        for offset in range(0, uv_length, prosody_stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = uv[:, :, offset: offset + prosody_length].to(torch.int32)
            encoded_uv.append(self._encode_prosody(frame, self.uv_embedding))
        
        encoded_tgt: tp.List[EncodedFrame] = []
        for offset in range(0, tgt_length, target_stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = tgt[:, :, offset: offset + target_length]
            encoded_tgt.append(frame)

        encoded_frames: tp.List[EncodedFrame] = []
        # print("length:", length, "stride:", stride)
        for idx, offset in enumerate(range(0, x_length, stride)):
            # print("start:", offset, "end:", offset + segment_length)
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame, encoded_f0[idx][1], encoded_uv[idx][1], train_stage))
        
        return encoded_frames, [f0[0] for f0 in encoded_f0], [uv[0] for uv in encoded_uv], encoded_tgt
    
    def _encode_prosody(self, x: torch.Tensor, emb_layer: nn.Embedding) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.prosody is None or duration <= 1e-5 + self.prosody
        emb_layer = emb_layer.to(x.device)
        e = emb_layer(x)
        e = e.permute(0, 3, 2, 1).squeeze(3)
        return x, e

    def _encode_frame(self, x: torch.Tensor, f0_emb: torch.Tensor, uv_emb: torch.Tensor, train_stage) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        # first several layers
        # emb = self.encoder(x)
        emb = self.encoder(x, "front") # torch.Size([5, 256, 600])

        # # classifier
        # # print(emb.shape)
        # pred_f0 = self.f0_classifier(emb.transpose(1, 2), train_stage="encoder") # torch.Size([5, 600, 256])
        # pred_uv = self.uv_classifier(emb.transpose(1, 2), train_stage="encoder") # torch.Size([5, 600, 2])

        mi_f0 = self.mi_loss(emb, f0_emb)
        mi_uv = self.mi_loss(emb, uv_emb)

        # any problem with scale?
        # print(emb.shape, f0_emb.shape, uv_emb.shape)
        emb = emb + f0_emb.to(emb.device) + uv_emb.to(emb.device)
        
        # the rest of the layers
        emb = self.encoder(emb, "back")
        # codes = self.quantizer.encode(emb, self.frame_rate, 6)

        # emb = self.encoder(x, "full")

        if self.training:# or True:
            return emb, scale, mi_f0, mi_uv
            # return emb, scale
        
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])
        
        frames = []
        for frame in encoded_frames:
            frames.append(self._decode_frame(frame))

        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if not self.training:# and False:
            codes = codes.transpose(0, 1)
            emb = self.quantizer.decode(codes)
        else:
            emb = codes
      
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor, f0: torch.Tensor, uv: torch.Tensor, tgt, train_stage: str) -> tuple[torch.Tensor, int, list[tuple[torch.Tensor, torch.Tensor]]]:
        self.quantizer.eval()
        l2Loss = torch.nn.MSELoss(reduction='mean')
        # CELoss = torch.nn.CrossEntropyLoss()
        frames, encoded_f0, encoded_uv, encoded_tgt = self.encode(x, f0, uv, tgt, train_stage)
        loss_enc = torch.tensor([0.0], device=x.device, requires_grad=True)
        codes = []

        mi_f0_sum = 0
        mi_uv_sum = 0
        loss_codes = 0
        loss_emb = 0
        if train_stage == "encoder":
            is_training = self.training
            for i, (emb, scale, mi_f0, mi_uv) in enumerate(frames):
                # loss_f0 = self.f0_classifier.loss(pred_f0, encoded_f0[i])
                # loss_uv = self.uv_classifier.loss(pred_uv, encoded_uv[i])
                mi_f0_sum += mi_f0
                mi_uv_sum += mi_uv
                # self.bandwidth = 6
                # codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
                # print(qv.min(),)
                l2_emb = l2Loss(emb, encoded_tgt[i])
                # loss_emb += l2_emb
                lambda_mi = 1
                loss_codes = loss_codes + lambda_mi * (mi_f0 + mi_uv) + l2_emb
                # print("predict: ", emb.max(), emb.min(), emb.mean())
                # print("target: ", encoded_tgt[i].max(), encoded_tgt[i].min(), encoded_tgt[i].mean())

            self.train(is_training)
            return loss_codes, mi_f0_sum, mi_uv_sum, l2_emb
        else:
            is_training = self.training
            self.train(self.train_quantization)
            for i, (emb, scale, pred_f0, pred_uv) in enumerate(frames):
                qv = self.quantizer.forward(emb, self.sample_rate, self.bandwidth)
                loss_f0 = self.f0_classifier.loss(pred_f0, encoded_f0[i])
                loss_uv = self.uv_classifier.loss(pred_uv, encoded_uv[i])

                loss_enc = loss_enc + qv.penalty + l2Loss(qv.quantized, emb) ** 2 + loss_f0 * 1e-6 + loss_uv * 1e-3 + l2Loss(emb, encoded_tgt[i])
                codes.append((qv.quantized, scale))
            self.train(is_training)
            return self.decode(codes)[:, :, :x.shape[-1]], loss_enc, frames, loss_f0, loss_uv

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        """Return the associated LM model to improve the compression rate.
        """
        torch.manual_seed(1234)  # todo remove: this
        device = next(self.parameters()).device
        lm = LMModel(self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200,
                     past_context=int(3.5 * self.frame_rate)).to(device)
        checkpoints = {
            'encodec_24khz': 'encodec_lm_24khz-1608e3c0.th',
            'encodec_48khz': 'encodec_lm_48khz-7add9fc3.th',
        }
        try:
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        state = torch.hub.load_state_dict_from_url(
            url, map_location='cpu', check_hash=True)  # type: ignore
        lm.load_state_dict(state)
        lm.eval()
        return lm

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset'):
        encoder = m.SEANetEncoder_scale(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))  # = 32
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split('-')[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type:ignore

    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained causal 24khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        checkpoint_name = 'encodec_24khz-d7cc33bc.th'
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_24khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained 48khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [3., 6., 12., 24.]
        checkpoint_name = 'encodec_48khz-7e698e3e.th'
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=False, model_norm='time_group_norm', audio_normalize=True,
            segment=1., name='encodec_48khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def my_encodec_model(checkpoint_name: str):
        """Return the trained model.
        """
        print("loading model from:", checkpoint_name)
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
                target_bandwidths, sample_rate, channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=1., name='my_encodec_24khz')
        pre_dic = torch.load(checkpoint_name)
        model.load_state_dict(pre_dic)
        model.eval()
        return model


def test():
    from itertools import product
    import torchaudio
    bandwidths = [3, 6, 12, 24]
    models = {
        'encodec_24khz': EncodecModel.encodec_model_24khz,
        'encodec_48khz': EncodecModel.encodec_model_48khz
    }
    for model_name, bw in product(models.keys(), bandwidths):
        model = models[model_name]()
        model.set_target_bandwidth(bw)
        audio_suffix = model_name.split('_')[1][:3]
        wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")
        wav = wav[:, :model.sample_rate * 2]
        wav_in = wav.unsqueeze(0)
        wav_dec = model(wav_in)[0]
        assert wav.shape == wav_dec.shape, (wav.shape, wav_dec.shape)


# if __name__ == '__main__':
#     test()
