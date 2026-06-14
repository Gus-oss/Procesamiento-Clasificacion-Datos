# Poner mis credenciales aqui.
# No subir a Github 

# config/credentials.py
import os
from dotenv import load_dotenv

# Carga el archivo .env
load_dotenv()

TWITTER_CONFIG = {
    "bearer_token"        : os.getenv("TWITTER_BEARER_TOKEN"),
    "api_key"             : os.getenv("TWITTER_API_KEY"),
    "api_key_secret"      : os.getenv("TWITTER_API_KEY_SECRET"),
    "access_token"        : "",
    "access_token_secret" : ""
}

YOUTUBE_CONFIG = {
    "api_key" : os.getenv("YOUTUBE_API_KEY")
}

APIFY_CONFIG = {
    "api_token" : os.getenv("APIFY_API_TOKEN")
}

TARGET = {
    "twitter_handle" : "AdriandelaGarza",
    "nombre"         : "Samuel García"
}
