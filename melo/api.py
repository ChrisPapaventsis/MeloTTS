import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import logging

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='cpu',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        if language in ['ZH', 'JP', 'KR']:
            # For Asian languages, split on punctuation and preserve it
            pattern = r'([。！？；])'
            sentences = re.split(pattern, text)
            # Combine punctuation with previous sentence
            result = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    result.append(sentences[i] + sentences[i+1])
                else:
                    result.append(sentences[i])
            return result
        else:
            # For other languages, split on standard sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Further split long sentences at commas or natural pauses
            result = []
            for sentence in sentences:
                if len(sentence.split()) > 15:  # If sentence is long
                    parts = re.split(r'(?<=[,;])\s+', sentence)
                    result.extend(parts)
                else:
                    result.append(sentence)
            return result

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)

    def tts_streaming(self, text, speaker_id, callback):
        """Stream TTS output sentence by sentence"""
        texts = self.split_sentences_into_pieces(text, self.language)
        sr = self.hps.data.sampling_rate
        
        for i, sentence in enumerate(texts):
            if not sentence.strip():
                continue
                
            # Preprocess input
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                sentence, self.language, self.hps, self.device, self.symbol_to_id
            )
            
            with torch.no_grad():
                x_tst = phones.unsqueeze(0).to(self.device)
                tones = tones.unsqueeze(0).to(self.device)
                lang_ids = lang_ids.unsqueeze(0).to(self.device)
                bert = bert.unsqueeze(0).to(self.device)
                ja_bert = ja_bert.unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                speakers = torch.LongTensor([speaker_id]).to(self.device)
                
                # Generate audio
                audio = self.model.infer(
                    x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert,
                    sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8
                )[0][0, 0].data.cpu().float().numpy()
                
                # Add small silence between sentences
                silence = np.zeros(int(sr * 0.1))
                audio_chunk = np.concatenate([audio, silence])

                # Stream out the audio chunk with progress info
                is_last = i == len(texts) - 1
                callback(audio_chunk, sr, is_last)

            # Clear GPU memory
            torch.cuda.empty_cache()
