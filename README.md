# BlipServer

## App

```
pip install pdm
pdm install
pdm run python app.py
```

## Config

`nano config.toml`

```toml
[blip]
low_vram = true
device = "cuda" #or cpu

[server]
host = '127.0.0.1'
port = 10885
```