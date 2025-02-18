import openai

from config import config
from models.config import EmbeddingConfig
from utils.logger import get_logger
from utils.authenticate import get_token
from customs.embeddings.custom_embeddings import CustomEmbeddings

logger = get_logger(name="embeder", log_dir="logs")
logger.info(f"config['embedding']: {config['embedding']}")

def create_embedding(model_config: EmbeddingConfig):    
    if model_config["api_key"] in [None, ""]:
        api_key = get_token(
            server_url=model_config["auth"]["server_url"],
            realm_name=model_config["auth"]["realm_name"],
            client_id=model_config["auth"]["client_id"],
            client_secret_key=model_config["auth"]["client_secret_key"],
            username=model_config["auth"]["username"],
            password=model_config["auth"]["password"]
        )
    else:
        api_key = model_config["api_key"]

    logger.info(f"api_key: {api_key}")
    
    return CustomEmbeddings(
        openai_api_base=model_config["model_url"],
        model=model_config["name"],
        openai_api_key=api_key
    )
