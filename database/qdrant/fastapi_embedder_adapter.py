from functools import wraps
from langchain.embeddings.base import Embeddings
import requests
import time
from typing import List

class FastAPIEmbedderAdapter(Embeddings):
    """FastAPI 임베딩 서버와 통신하는 어댑터"""
    
    def __init__(self, base_url="http://localhost:8000", retry_attempts=3, retry_delay=2, timeout=60):
        self.base_url = base_url.rstrip('/')
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._verify_connection()

    def _verify_connection(self):
        """서버 연결 확인"""
        print(f"[🔗] Connecting to FastAPI server at {self.base_url}...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            print("[✅] FastAPI connection successful.")
        except Exception as e:
            print(f"[❌] Failed to connect to FastAPI server: {e}")
            raise

    @staticmethod
    def _retry(func):
        """재시도 데코레이터"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            attempts = 0
            last_exception = None
            
            while attempts < self.retry_attempts:
                try:
                    return func(self, *args, **kwargs)
                except requests.exceptions.RequestException as e:
                    attempts += 1
                    last_exception = e
                    print(f"[⚡] FastAPI request failed ({e}). Retrying {attempts}/{self.retry_attempts}...")
                    
                    if attempts < self.retry_attempts:
                        time.sleep(self.retry_delay * attempts)
                    else:
                        print(f"[❌] FastAPI server request failed after {self.retry_attempts} attempts.")
                        raise Exception(f"FastAPI server operation failed: {last_exception}")
                except Exception as e:
                    print(f"[❌] Unexpected error during FastAPI call: {e}")
                    raise
                    
            raise Exception(f"Failed after {self.retry_attempts} retries: {last_exception}")
        return wrapper

    @_retry
    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]

    @_retry
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 배치 임베딩"""
        response = requests.post(
            f"{self.base_url}/embed_documents",
            json={"texts": texts},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["embeddings"]
