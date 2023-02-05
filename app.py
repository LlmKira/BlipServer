# -*- coding: utf-8 -*-
# @Time    : 2/5/23 12:26 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import rtoml
from pydantic import BaseModel
import tempfile
from typing import Union, Optional
import shutil

import uvicorn

from utils import Blip
from fastapi import FastAPI, File, UploadFile
from PIL import Image

CONF = rtoml.load(open("config.toml", 'r'))

BlipModel = CONF["blip"].get("model")
if BlipModel not in ['large', 'base']:
    BlipModel = 'large'
BlipConfig = Blip.Config(device=CONF["blip"].get("device"))
BlipConfig.model = BlipModel
BlipInterrogator = Blip.Interrogator(BlipConfig)

app = FastAPI()


@app.post("/upload/")
def create_upload_file(file: Optional[UploadFile] = None):
    if not file.file:
        return {"code": 0, "message": "No upload file sent"}
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as buffer:
            shutil.copyfileobj(file.file, buffer)
            image_pil = Image.open(buffer.name).convert('RGB')
        if image_pil:
            BlipInterrogatorText = BlipInterrogator.generate_caption(
                pil_image=image_pil)
            return {"code": 1, "message": BlipInterrogatorText}


if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=10885, reload=False, log_level="debug", workers=1)
