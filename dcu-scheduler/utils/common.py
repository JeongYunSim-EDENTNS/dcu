import random, string, json, pytz, os

from datetime import datetime
from bs4 import BeautifulSoup

from models.config import EmbeddingConfig

from customs.embeddings.custom_embeddings import CustomEmbeddings

def get_timezone():
    return pytz.timezone("Asia/Seoul")

def get_now():
    """현재 시간 가져오기"""
    return datetime.now(get_timezone())

def extract_text_from_html(html_content):
    """
    HTML에서 순수 텍스트를 추출합니다.
    :param html_content: HTML 형식의 문자열
    :return: 순수 텍스트
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        raise ValueError(f"HTML 텍스트 추출 중 오류 발생: {e}")

