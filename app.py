# -*- coding: utf-8 -*-
# @Time    : 2/5/23 12:26 PM
# @FileName: main.py
# @Software: PyCharm
# @Github    ：sudoskys
import io

import rtoml
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from loguru import logger
from starlette.responses import JSONResponse

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

LowVram = BlipConf.get("low_vram") if BlipConf.get("low_vram") else False

BlipConfig = Blip.Config(device=BlipConf.get("device"))

if LowVram:
    BlipConfig.apply_low_vram_defaults()

BlipInterrogator = Blip.Interrogator(BlipConfig)

app = FastAPI()


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    if image_pil:
        blip_interrogator_text = BlipInterrogator.generate_caption(pil_image=image_pil)
        return JSONResponse(
            content={"text": blip_interrogator_text}
        )


if __name__ == '__main__':
    uvicorn.run('app:app', host=ServerHost, port=ServerPort,
                reload_delay=5,
                reload=False,
                log_level="debug",
                workers=1
                )
