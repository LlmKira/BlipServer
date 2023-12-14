# -*- coding: utf-8 -*-
# @Time    : 2/5/23 12:34 PM
# @FileName: Blip.py
# @Software: PyCharm
# @Github    ：sudoskys
import atexit
import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from loguru import logger
from transformers import BlipForConditionalGeneration, Blip2ForConditionalGeneration, AutoModelForCausalLM, \
    AutoProcessor

# from utils.Base import Tool
if not torch.cuda.is_available():
    logger.warning("GPU Unavailable,If You Enable The Media Service,May Cause CPU OverLoaded")


@dataclass
class Config:
    # models can optionally be passed in directly
    caption_model = None
    caption_processor = None
    clip_model = None
    clip_preprocess = None

    # blip settings
    caption_max_length: int = 32
    caption_model_name: Optional[str] = 'blip-large'  # use a key from CAPTION_MODELS or None
    caption_offload: bool = False

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: Optional[str] = None
    clip_offload: bool = False

    # interrogator settings
    cache_path: str = 'cache'  # path to store cached text embeddings
    download_cache: bool = True  # when true, cached embeds are downloaded from huggingface
    chunk_size: int = 2048  # batch size for CLIP, use smaller for lower VRAM
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    flavor_intermediate_count: int = 2048
    quiet: bool = False  # when quiet progress bars are not shown

    def apply_low_vram_defaults(self):
        self.caption_model_name = 'blip-base'
        self.caption_offload = True
        self.clip_offload = True
        self.chunk_size = 1024
        self.flavor_intermediate_count = 1024


CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',  # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',  # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',  # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',  # 1.58GB
}


class Interrogator(object):
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model_name = config.caption_model_name
        self.caption_offloaded = True
        self.clip_offloaded = True

        self.caption_model = None
        self.caption_processor = None
        self.load_caption_model()

    def load_caption_model(self):
        if self.config.caption_model is None and self.config.caption_model_name:
            if not self.config.quiet:
                print(f"Loading caption model {self.config.caption_model_name}...")

            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if self.config.caption_model_name.startswith('git-'):
                caption_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            elif self.config.caption_model_name.startswith('blip2-'):
                caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
            self.caption_processor = AutoProcessor.from_pretrained(model_path)

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    def _prepare_caption(self):
        if self.config.clip_offload and not self.clip_offloaded:
            self.clip_model = self.clip_model.to('cpu')
            self.clip_offloaded = True
        if self.caption_offloaded:
            self.caption_model = self.caption_model.to(self.device)
            self.caption_offloaded = False

    def generate_caption(self, pil_image: Image) -> str:
        assert self.caption_model is not None, "No caption model loaded."
        self._prepare_caption()
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
        if not self.config.caption_model_name.startswith('git-'):
            inputs = inputs.to(self.dtype)
        tokens = self.caption_model.generate(**inputs, max_new_tokens=self.config.caption_max_length)
        return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


@atexit.register
def torch_done():
    torch.cuda.empty_cache()
