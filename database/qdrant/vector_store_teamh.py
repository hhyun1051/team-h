# vector_store_teamh.py
from functools import wraps
from langchain.embeddings.base import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, Filter, FieldCondition, MatchValue, MatchAny
from langchain_core.documents import Document
from collections import defaultdict
import traceback
import time
from typing import Optional, Dict, Any, List, Set
from database.qdrant.fastapi_embedder_adapter import FastAPIEmbedderAdapter
from database.qdrant.openai_embedder_adapter import OpenAIEmbedderAdapter
from database.qdrant.config import MemoryConfig

def timing_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n[⏱️] Starting '{func.__name__}'...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[✅] Finished '{func.__name__}' in {elapsed:.2f} seconds\n")
        return result
    return wrapper

def create_qdrant_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
    """`None`, `"None"`, 빈 리스트는 무시하고 조건을 만든다."""
    if not metadata_filter:
        return None

    conditions = []
    for field, value in metadata_filter.items():
        if value is None or \
            value == "None" or \
            (isinstance(value, list) and not value):
            continue

        key = f"metadata.{field}"
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=conditions) if conditions else None


class VectorStore:
    @timing_step
    def __init__(
        self,
        url="http://localhost:6333",
        api_key=None,
        collection_name=None,
        dense_size=None,
        recreate_collection=True,
        embedding_type=None,
        embedder_url=None,
        openai_api_key=None,
    ):
        """
        VectorStore 초기화

        Args:
            url: Qdrant 서버 URL
            api_key: Qdrant API 키
            collection_name: 컬렉션 이름
            dense_size: 임베딩 차원 (기본값: embedding_type에 따라 자동 설정)
            recreate_collection: 컬렉션 재생성 여부
            embedding_type: 임베딩 타입 ("fastapi" 또는 "openai", 기본값: config에서 로드)
            embedder_url: FastAPI 임베딩 서버 URL (embedding_type="fastapi"일 때 사용)
            openai_api_key: OpenAI API 키 (embedding_type="openai"일 때 사용)
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.recreate_collection = recreate_collection

        # embedding_type 결정 (파라미터 > config > 기본값)
        embedding_type = embedding_type or MemoryConfig.EMBEDDING_TYPE

        # Qdrant 클라이언트 초기화
        client_args = {'url': url, 'timeout': 60}
        if api_key:
            client_args['api_key'] = api_key
        self.client = QdrantClient(**client_args)

        self.sparse_embedder = self._create_sparse_embedder()

        # 임베딩 타입에 따라 embedder 초기화
        if embedding_type == "fastapi":
            embedder_url = embedder_url or MemoryConfig.EMBEDDER_URL
            dense_size = dense_size or MemoryConfig.FASTAPI_EMBEDDING_DIMS

            print(f"[🔗] Initializing FastAPI Embedder: {embedder_url}")
            self.embedder = FastAPIEmbedderAdapter(
                base_url=embedder_url,
                retry_attempts=3,
                retry_delay=2,
                timeout=60
            )
            self.dense_size = dense_size

        elif embedding_type == "openai":
            openai_api_key = openai_api_key or MemoryConfig.OPENAI_API_KEY
            dense_size = dense_size or MemoryConfig.OPENAI_EMBEDDING_DIMS

            print(f"[🔗] Initializing OpenAI Embedder: text-embedding-3-large")
            self.embedder = OpenAIEmbedderAdapter(
                api_key=openai_api_key,
                dimensions=dense_size,
            )
            self.dense_size = dense_size

        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be 'fastapi' or 'openai'.")

        self.embedding_type = embedding_type

        self._ensure_collection_exists()
        self.vector_store = self._create_vector_store()

    @staticmethod
    def _filter_internal_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """메타데이터에서 '_'로 시작하는 내부 키를 필터링합니다."""
        if not metadata:
            return {}
        return {k: v for k, v in metadata.items() if not k.startswith('_')}

    @timing_step
    def delete_collection(self):
        """컬렉션을 삭제하는 메서드"""
        try:
            print(f"Attempting to delete collection '{self.collection_name}'...")
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting collection '{self.collection_name}': {e}")
            return False

    @timing_step
    def _ensure_collection_exists(self):
        """콜렉션이 존재하는지 확인하고 없으면 생성"""
        try:
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            print(f"Existing collections: {collection_names}")
            collection_exists = self.collection_name in collection_names

            if self.recreate_collection and collection_exists:
                print(f"Recreate option is enabled. Deleting existing collection '{self.collection_name}'...")
                deleted = self.delete_collection()
                if deleted:
                    collection_exists = False
                else:
                    print(f"[⚠️] Failed to delete collection '{self.collection_name}'.")

            if not collection_exists:
                print(f"Collection '{self.collection_name}' does not exist or was deleted. Creating...")
                self.create_collection()
            else:
                print(f"Collection '{self.collection_name}' already exists and recreate=False.")

        except Exception as e:
            print(f"[❌] Error ensuring collection exists: {e}")
            raise

    @timing_step
    def create_collection(self):
        """Create a collection with both dense and sparse vectors"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.dense_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            )
            print(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            print(f"[❌] Error creating collection '{self.collection_name}': {e}")
            pass

    @timing_step
    def _create_sparse_embedder(self):
        return FastEmbedSparse(model_name="Qdrant/bm25")

    @timing_step
    def _create_vector_store(self):
        """Langchain QdrantVectorStore 객체를 생성합니다."""
        if not self.embedder:
             raise RuntimeError("[❌] Dense embedder is not available.")
        if not self.sparse_embedder:
             print("[⚠️] Sparse embedder is not available. Hybrid search might not work as expected.")

        try:
            _ = self.embedder.embed_query("connection test")
            print("[✅] Embedder connection verified before creating VectorStore.")
        except Exception as e:
            print(f"[❌] Failed to verify embedder connection: {e}")

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder,
            sparse_embedding=self.sparse_embedder,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    @timing_step
    def add_documents(self, documents, batch_size=64):
        """Langchain Document 리스트를 Qdrant에 추가합니다."""
        total = len(documents)
        if not total:
            print("[⚠️] No documents provided to add.")
            return

        print(f"Starting to add {total} documents using Langchain's add_documents...")
        try:
            ids_returned = self.vector_store.add_documents(documents=documents)
            print(f"[✅] Successfully added {total} documents. Returned IDs count: {len(ids_returned)}")

        except Exception as e:
            print(f"[❌] Error adding documents via Langchain: {e}")
            print(traceback.format_exc())

    @timing_step
    def search(self, query: str, k: int = 4, metadata_filter: dict = None):
        """Langchain VectorStore를 이용한 기본 검색 (메타데이터 필터링 적용)"""
        qdrant_filter = create_qdrant_filter(metadata_filter)
        if qdrant_filter:
            print(f"[🔍] 생성된 필터를 적용합니다: {metadata_filter}")

        try:
             if not self.vector_store: raise RuntimeError("[❌] Vector store is not initialized.")
             print(f"[🔍] Performing similarity search for query: '{query[:50]}...' with k={k}")
             results = self.vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)
             for doc in results:
                 doc.metadata = self._filter_internal_metadata(doc.metadata)
             print(f"[✅] Found {len(results)} results.")
             return results
        except Exception as e:
             print(f"[❌] Error during similarity search: {e}")
             return []

    @timing_step
    def search_with_score(self, query: str, k: int = 4, metadata_filter: dict = None):
        """Langchain VectorStore를 이용한 점수 포함 검색"""
        qdrant_filter = create_qdrant_filter(metadata_filter)
        if qdrant_filter:
             print(f"[🔍] 생성된 필터를 적용합니다: {metadata_filter}")

        try:
             if not self.vector_store: raise RuntimeError("[❌] Vector store is not initialized.")
             print(f"[🔍] Performing similarity search with score for query: '{query[:50]}...' with k={k}")
             results = self.vector_store.similarity_search_with_score(query=query, k=k, filter=qdrant_filter)
             for doc, score in results:
                 doc.metadata = self._filter_internal_metadata(doc.metadata)
             print(f"[✅] Found {len(results)} results with scores.")
             return results
        except Exception as e:
             print(f"[❌] Error during similarity search with score: {e}")
             return []
