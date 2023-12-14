# -*- coding: utf-8 -*-
# @Time    : 2/5/23 12:26 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import shutil
import tempfile
from typing import Optional

import rtoml
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
from loguru import logger

from utils import Blip

CONF = rtoml.load(open("config.toml", 'r'))
ServerConf = CONF.get("server") if CONF.get("server") else {}
BlipConf = CONF.get("blip") if CONF.get("blip") else {}

if not BlipConf:
    logger.warning("No BlipConf")
if not ServerConf:
    logger.warning("No ServerConf")

AutoReload = ServerConf.get("reload") if ServerConf.get("reload") else False
ServerHost = ServerConf.get("host") if ServerConf.get("host") else "127.0.0.1"
ServerPort = ServerConf.get("port") if ServerConf.get("port") else 10885

LowVram = BlipConf.get("low_vram_model") if BlipConf.get("low_vram_model") else False

BlipConfig = Blip.Config(device=BlipConf.get("device"))

if LowVram:
    BlipConfig.apply_low_vram_defaults()

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
    uvicorn.run('app:app', host=ServerHost, port=ServerPort,
                reload_delay=5,
                reload=False,
                log_level="debug",
                workers=1
                )
