import yaml
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

from src.rag.chain import RagChain
from src.rag.prompt import build_medical_prompt
from src.rag.memory import ConversationMemory

# Phase-5
from langsmith import traceable
from langsmith.run_helpers import get_run_tree_context



class RagService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        with open("configs/retrieval.yaml") as f:
            self.retrieval_cfg = yaml.safe_load(f)

        with open("configs/guardrails.yaml") as f:
            self.guardrail_cfg = yaml.safe_load(f)

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(
            path=self.retrieval_cfg["dense"]["persist_directory"],
            settings=Settings(anonymized_telemetry=False),
        )

        collection = client.get_collection(
            name=self.retrieval_cfg["dense"]["collection_name"]
        )

        self.dense = DenseRetriever(collection)
        self.sparse = SparseRetriever()

        self.hybrid = HybridRetriever(
            self.dense,
            self.sparse,
            alpha=self.retrieval_cfg["hybrid"]["alpha"],
        )

        self.reranker = Reranker(
            self.retrieval_cfg["reranker"]["model_name"]
        )

        self.memory = ConversationMemory()

        self.chain = RagChain(
            model=self.retrieval_cfg["llm"]["model"],
            temperature=self.retrieval_cfg["llm"]["temperature"],
            guardrail_cfg=self.guardrail_cfg,
        )

    # def ask(self, query):
    #     start_total = time.time()
    #     start_retrieval = time.time()

    #     query_embedding = self.embedder.encode(
    #         query,
    #         normalize_embeddings=True,
    #     )

    #     docs = self.hybrid.retrieve(
    #         query,
    #         query_embedding,
    #         self.retrieval_cfg["dense"]["top_k"],
    #         self.retrieval_cfg["sparse"]["top_k"],
    #     )

    #     docs = self.reranker.rerank(
    #         query,
    #         docs,
    #         self.retrieval_cfg["reranker"]["top_k"],
    #     )

    #     retrieval_time = time.time() - start_retrieval

    #     context = "\n\n".join(
    #         [f"[{i+1}] {d['text']}" for i, d in enumerate(docs)]
    #     )

    #     history = self.memory.get_history()

    #     prompt = build_medical_prompt(query, context, history)

    #     response, llm_time = self.chain.generate(query, docs, prompt)

    #     total_time = time.time() - start_total

    #     return {
    #         "response": response.dict(),
    #         "timing": {
    #             "retrieval_time": retrieval_time,
    #             "llm_time": llm_time,
    #             "total_time": total_time,
    #         },
    #     }

    @traceable(name="RAG_Request")
    def ask(self, query: str):
        start_total = time.time()

        with get_run_tree_context().trace("retrieval"):
            start_retrieval = time.time()

            query_embedding = self.embedder.encode(
                query,
                normalize_embeddings=True,
            )

            docs = self.hybrid.retrieve(
                query,
                query_embedding,
                self.retrieval_cfg["dense"]["top_k"],
                self.retrieval_cfg["sparse"]["top_k"],
            )

            docs = self.reranker.rerank(
                query,
                docs,
                self.retrieval_cfg["reranker"]["top_k"],
            )

            retrieval_time = time.time() - start_retrieval

        with get_run_tree_context().trace("prompt_building"):
            context = "\n\n".join(
                [f"[{i+1}] {d['text']}" for i, d in enumerate(docs)]
            )

            history = self.memory.get_history()
            prompt = build_medical_prompt(query, context, history)

        with get_run_tree_context().trace("llm_generation"):
            response, llm_time = self.chain.generate(query, docs, prompt)

        total_time = time.time() - start_total

        return {
            "response": response.dict(),
            "timing": {
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": total_time,
            },
        }
