# BlipServer

## App

`pip install -r requirements.txt`

`python3 app.py`

## Config

`nano config.toml`

```toml
[blip]
model = "large" #or base
device = "cuda" #or cpu

[server]
host = '127.0.0.1'
port = 10885
```