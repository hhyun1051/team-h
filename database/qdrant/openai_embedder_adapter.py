"""
OpenAI Embedding API와 통신하는 어댑터 (text-embedding-3-large 전용)
"""

from langchain.embeddings.base import Embeddings
from typing import List
import os
from openai import OpenAI


class OpenAIEmbedderAdapter(Embeddings):
    """OpenAI Embedding API를 사용하는 어댑터 (text-embedding-3-large)"""

    # text-embedding-3-large의 기본 차원
    DEFAULT_DIMENSIONS = 3072

    def __init__(
        self,
        api_key: str = None,
        dimensions: int = None,
    ):
        """
        OpenAI Embedder 초기화

        Args:
            api_key: OpenAI API 키 (기본값: 환경변수에서 로드)
            dimensions: 임베딩 차원 (기본값: 3072)
                       - 1 ~ 3072 사이의 값으로 설정 가능
                       - 작은 값을 사용하면 비용 절감 및 성능 향상
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-large"
        self.dimensions = dimensions or self.DEFAULT_DIMENSIONS

        print(f"[🔗] Initializing OpenAI Embedder: {self.model}")
        print(f"    - Dimensions: {self.dimensions}")

    def get_embedding_dimensions(self) -> int:
        """
        현재 임베딩 모델의 차원 반환

        Returns:
            임베딩 차원
        """
        return self.dimensions

    def embed_query(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[❌] OpenAI embedding failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트 배치 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[❌] OpenAI batch embedding failed: {e}")
            raise
