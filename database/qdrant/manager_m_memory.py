"""
Manager M 메모리 관리 모듈 (Qdrant + 커스텀 임베더 직접 구현)

Manager M은 목표 외의 모든 일반적인 기억을 관리합니다.
- 사용자 선호도
- 대화 컨텍스트
- 일상적인 상호작용
- 사용자 습관 및 패턴

기존 FastAPIEmbedderAdapter를 재사용합니다.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from database.qdrant.fastapi_embedder_adapter import FastAPIEmbedderAdapter
from database.qdrant.openai_embedder_adapter import OpenAIEmbedderAdapter
from database.qdrant.config import MemoryConfig


class ManagerMMemory:
    """Manager M을 위한 메모리 관리 클래스"""

    def __init__(
        self,
        embedding_type: Optional[str] = None,
        embedder_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dims: Optional[int] = None,
    ):
        """
        Manager M 메모리 초기화

        Args:
            embedding_type: 임베딩 타입 ("fastapi" 또는 "openai", 기본값: config에서 로드)
            embedder_url: FastAPI 임베딩 서버 URL (EMBEDDING_TYPE="fastapi"일 때 사용)
            openai_api_key: OpenAI API 키 (EMBEDDING_TYPE="openai"일 때 사용)
            qdrant_url: Qdrant 서버 URL (기본값: config에서 로드)
            qdrant_api_key: Qdrant API 키 (기본값: config에서 로드)
            collection_name: Qdrant 컬렉션 이름 (기본값: config에서 로드)
            embedding_dims: 임베딩 차원 (기본값: config에서 로드)
        """
        # config에서 기본값 로드
        embedding_type = embedding_type or MemoryConfig.EMBEDDING_TYPE
        qdrant_url = qdrant_url or MemoryConfig.QDRANT_URL
        qdrant_api_key = qdrant_api_key or MemoryConfig.QDRANT_PASSWORD
        collection_name = collection_name or MemoryConfig.MANAGER_M_COLLECTION

        # 설정 검증
        if not MemoryConfig.validate():
            raise ValueError("Invalid memory configuration. Check .env file.")

        self.collection_name = collection_name
        self.embedding_type = embedding_type

        # 임베딩 타입에 따라 embedder 초기화
        if embedding_type == "fastapi":
            embedder_url = embedder_url or MemoryConfig.EMBEDDER_URL
            embedding_dims = embedding_dims or MemoryConfig.FASTAPI_EMBEDDING_DIMS

            print(f"[🔗] Initializing FastAPI Embedder: {embedder_url}")
            self.embedder = FastAPIEmbedderAdapter(
                base_url=embedder_url,
                retry_attempts=3,
                retry_delay=2,
                timeout=60
            )
            self.embedding_dims = embedding_dims

        elif embedding_type == "openai":
            openai_api_key = openai_api_key or MemoryConfig.OPENAI_API_KEY
            embedding_dims = embedding_dims or MemoryConfig.OPENAI_EMBEDDING_DIMS

            print(f"[🔗] Initializing OpenAI Embedder: text-embedding-3-large")
            self.embedder = OpenAIEmbedderAdapter(
                api_key=openai_api_key,
                dimensions=embedding_dims,
            )
            self.embedding_dims = embedding_dims

        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be 'fastapi' or 'openai'.")

        # Qdrant 클라이언트 초기화
        print(f"[🔗] Connecting to Qdrant: {qdrant_url}")
        client_args = {'url': qdrant_url, 'timeout': 60}
        if qdrant_api_key:
            client_args['api_key'] = qdrant_api_key
        self.client = QdrantClient(**client_args)

        # 컬렉션 생성 또는 확인
        self._ensure_collection()

        print(f"[✅] Manager M Memory initialized")
        print(f"    - Collection: {self.collection_name}")
        print(f"    - Qdrant: {qdrant_url}")
        print(f"    - Embedding Type: {self.embedding_type}")
        print(f"    - Embedding Dims: {self.embedding_dims}")

    def _ensure_collection(self):
        """컬렉션이 존재하지 않으면 생성"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                print(f"[📦] Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dims,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"[✅] Collection created: {self.collection_name}")
            else:
                print(f"[✅] Collection already exists: {self.collection_name}")
        except Exception as e:
            print(f"[❌] Failed to ensure collection: {e}")
            raise

    def add_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        새로운 기억 추가

        Args:
            content: 저장할 기억 내용
            user_id: 사용자 ID
            memory_type: 기억 유형 (general, preference, habit, interaction 등)
            metadata: 추가 메타데이터

        Returns:
            추가된 기억 정보 (memory_id 포함)
        """
        try:
            # 임베딩 생성
            embedding = self.embedder.embed_query(content)

            # 메모리 ID 생성
            memory_id = str(uuid.uuid4())

            # 메타데이터 구성
            full_metadata = {
                "content": content,
                "user_id": user_id,
                "memory_type": memory_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                **(metadata or {})
            }

            # Qdrant에 저장
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=full_metadata,
                    )
                ],
            )

            print(f"[✅] Memory added for user '{user_id}': {content[:50]}...")
            return {
                "id": memory_id,
                "content": content,
                "user_id": user_id,
                "memory_type": memory_type,
                "metadata": full_metadata,
            }
        except Exception as e:
            print(f"[❌] Failed to add memory: {e}")
            raise

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        관련 기억 검색

        Args:
            query: 검색 쿼리
            user_id: 사용자 ID
            limit: 최대 결과 개수
            memory_type: 기억 유형 필터 (옵션)

        Returns:
            관련성 높은 기억 리스트 (score 포함)
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedder.embed_query(query)

            # 필터 구성
            must_conditions = [
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]

            if memory_type:
                must_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type),
                    )
                )

            query_filter = Filter(must=must_conditions) if must_conditions else None

            # Qdrant 검색
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
            )

            # 결과 포맷팅
            memories = []
            for result in search_results:
                memories.append({
                    "id": result.id,
                    "content": result.payload.get("content", ""),  # 'content' 키로 변경
                    "type": result.payload.get("memory_type", "unknown"),  # 'type' 키 추가
                    "score": result.score,
                    "metadata": result.payload,
                })

            print(f"[🔍] Found {len(memories)} memories for query: '{query[:50]}...'")
            return memories
        except Exception as e:
            print(f"[❌] Failed to search memories: {e}")
            raise

    def get_all_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        사용자의 모든 기억 조회

        Args:
            user_id: 사용자 ID
            memory_type: 기억 유형 필터 (옵션)
            limit: 최대 반환 개수 (기본값: 100)

        Returns:
            모든 기억 리스트
        """
        try:
            # 필터 구성
            must_conditions = [
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]

            if memory_type:
                must_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type),
                    )
                )

            query_filter = Filter(must=must_conditions)

            # Qdrant scroll (전체 조회)
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,  # 파라미터로 받은 limit 사용
                with_payload=True,
                with_vectors=False,
            )

            # 결과 포맷팅
            memories = []
            for point in scroll_results[0]:
                memories.append({
                    "id": point.id,
                    "content": point.payload.get("content", ""),  # 'content' 키로 변경
                    "type": point.payload.get("memory_type", "unknown"),  # 'type' 키 추가
                    "metadata": point.payload,
                })

            print(f"[📋] Retrieved {len(memories)} memories for user '{user_id}'")
            return memories
        except Exception as e:
            print(f"[❌] Failed to get all memories: {e}")
            raise

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        ID로 특정 기억 조회

        Args:
            memory_id: 조회할 기억의 ID

        Returns:
            기억 정보 (id, content, type 등), 없으면 None
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
            )

            if result and len(result) > 0:
                point = result[0]
                return {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "type": point.payload.get("memory_type", "unknown"),
                    "metadata": point.payload,
                }
            else:
                return None

        except Exception as e:
            print(f"[❌] Failed to retrieve memory {memory_id}: {e}")
            return None

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        특정 기억 삭제

        Args:
            memory_id: 삭제할 기억의 ID

        Returns:
            삭제 결과
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id],
            )
            print(f"[🗑️] Memory deleted: {memory_id}")
            return {"status": "deleted", "memory_id": memory_id}
        except Exception as e:
            print(f"[❌] Failed to delete memory: {e}")
            raise

    def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        """
        사용자의 모든 기억 삭제

        Args:
            user_id: 사용자 ID

        Returns:
            삭제 결과
        """
        try:
            # 필터로 해당 사용자의 모든 포인트 삭제
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
            )
            print(f"[🗑️] All memories deleted for user '{user_id}'")
            return {"status": "deleted", "user_id": user_id}
        except Exception as e:
            print(f"[❌] Failed to delete all memories: {e}")
            raise

    def update_memory(
        self,
        memory_id: str,
        content: str
    ) -> Dict[str, Any]:
        """
        기존 기억 업데이트

        Args:
            memory_id: 업데이트할 기억의 ID
            content: 새로운 내용

        Returns:
            업데이트 결과
        """
        try:
            # 기존 메모리 가져오기
            existing = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )

            if not existing:
                raise ValueError(f"Memory not found: {memory_id}")

            # 새 임베딩 생성
            embedding = self.embedder.embed_query(content)

            # 기존 메타데이터 유지하고 업데이트
            payload = existing[0].payload
            payload["content"] = content
            payload["updated_at"] = datetime.now().isoformat()

            # 업데이트 (upsert 사용)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            print(f"[♻️] Memory updated: {memory_id}")
            return {
                "id": memory_id,
                "content": content,
                "metadata": payload,
            }
        except Exception as e:
            print(f"[❌] Failed to update memory: {e}")
            raise

    def get_memory_history(
        self,
        memory_id: str
    ) -> List[Dict[str, Any]]:
        """
        특정 기억 조회 (히스토리는 현재 미지원)

        Note: 현재 구현에서는 히스토리를 별도로 추적하지 않습니다.
              향후 버전 관리가 필요하면 별도 컬렉션에 히스토리를 저장해야 합니다.

        Args:
            memory_id: 기억 ID

        Returns:
            현재 상태만 반환
        """
        try:
            memory = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )

            if not memory:
                return []

            print(f"[📜] Retrieved memory: {memory_id}")
            return [{
                "id": memory[0].id,
                "content": memory[0].payload.get("content", ""),
                "metadata": memory[0].payload,
            }]
        except Exception as e:
            print(f"[❌] Failed to get memory: {e}")
            raise

    def get_user_context_summary(
        self,
        user_id: str,
        max_memories: int = 10,
        memory_type: Optional[str] = None
    ) -> str:
        """
        사용자의 최근 기억을 요약하여 컨텍스트로 반환

        Args:
            user_id: 사용자 ID
            max_memories: 최대 기억 개수
            memory_type: 기억 유형 필터

        Returns:
            포맷된 컨텍스트 문자열
        """
        memories = self.get_all_memories(user_id, memory_type)

        if not memories:
            return f"No previous context available for user '{user_id}'."

        # 최근 기억만 선택 (created_at 기준 정렬)
        sorted_memories = sorted(
            memories,
            key=lambda x: x.get("metadata", {}).get("created_at", ""),
            reverse=True
        )
        recent_memories = sorted_memories[:max_memories]

        # 포맷팅
        context = f"=== User Context for '{user_id}' ===\n"
        context += f"Total memories: {len(memories)} (showing recent {len(recent_memories)})\n\n"

        for idx, mem in enumerate(recent_memories, 1):
            memory_text = mem.get("content", "")  # 'content' 키로 변경
            mem_type = mem.get("type", "unknown")  # 'type' 키 사용
            created = mem.get("metadata", {}).get("created_at", "unknown")
            context += f"{idx}. [{mem_type}] {memory_text}\n"
            context += f"   Created: {created}\n\n"

        return context


# 싱글톤 인스턴스 (선택적 사용)
_manager_m_memory_instance = None


def get_manager_m_memory(
    embedder_url: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> ManagerMMemory:
    """
    Manager M 메모리 싱글톤 인스턴스 반환

    Args:
        embedder_url: FastAPI 임베딩 서버 URL
        qdrant_url: Qdrant 서버 URL
        qdrant_api_key: Qdrant API 키

    Returns:
        ManagerMMemory 인스턴스
    """
    global _manager_m_memory_instance
    if _manager_m_memory_instance is None:
        _manager_m_memory_instance = ManagerMMemory(
            embedder_url=embedder_url,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    return _manager_m_memory_instance