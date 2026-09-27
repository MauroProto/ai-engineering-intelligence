import asyncio
import hashlib
import re
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from .config import settings
from .observability import tracer, span_input, span_output
from .state import Evidence


def terms(text):
    return re.findall(r"\w+", text.lower())


class HybridKnowledgeBase:
    """Chroma persistente + BM25, IDs SHA256 y RRF sin duplicados."""
    def __init__(self, gateway):
        self.gateway = gateway
        self.chunks = []
        self.collection = None
        self.bm25 = None

    async def setup(self):
        # setup repetido no duplica el indice lexical ni los fragmentos.
        self.chunks = []
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",chunk_size=500,chunk_overlap=50)
        paths = sorted((Path(__file__).resolve().parents[1]/"corpus").glob("*.md"))
        for path in paths:
            text = await asyncio.to_thread(path.read_text,encoding="utf-8")
            for index,chunk in enumerate(splitter.split_text(text)):
                chunk_id = hashlib.sha256(f"{path.name}:{index}:{chunk}".encode()).hexdigest()[:24]
                self.chunks.append({"id":chunk_id,"text":chunk,"source":path.name,
                                    "page":1,"category":path.stem,"score":0.0})
        def open_collection():
            client = chromadb.PersistentClient(path=str(settings.data_dir/"chroma"),
                settings=ChromaSettings(anonymized_telemetry=False))
            # Modelo forma parte del nombre: no mezclar espacios vectoriales.
            return client.get_or_create_collection(
                "course_"+hashlib.sha256(("onnx_local:"+settings.embedding_model).encode()).hexdigest()[:10],
                metadata={"hnsw:space":"cosine"},embedding_function=None)
        self.collection = await asyncio.to_thread(open_collection)
        existing = await asyncio.to_thread(self.collection.get,ids=[x["id"] for x in self.chunks])
        missing = [x for x in self.chunks if x["id"] not in set(existing["ids"])]
        if missing:
            vectors,usage = await self.gateway.embeddings([x["text"] for x in missing])
            await asyncio.to_thread(self.collection.upsert,
                ids=[x["id"] for x in missing],documents=[x["text"] for x in missing],
                metadatas=[{k:x[k] for k in ("source","page","category")} for x in missing],
                embeddings=vectors)
            self.ingestion_usage = usage
        else:
            self.ingestion_usage = None
        # Eliminar fragmentos obsoletos de este corpus al modificar un archivo.
        stored = await asyncio.to_thread(self.collection.get)
        stale = set(stored["ids"])-{x["id"] for x in self.chunks}
        if stale:
            await asyncio.to_thread(self.collection.delete,ids=sorted(stale))
        self.bm25 = BM25Okapi([terms(x["text"]) for x in self.chunks])

    async def search(self,query,limit=4):
        with tracer.start_as_current_span("tool.search_knowledge_base") as span:
            span_input(span,{"query":query,"limit":limit},"RETRIEVER")
            vectors,usage = await self.gateway.embeddings([query])
            result = await asyncio.to_thread(self.collection.query,
                query_embeddings=vectors,n_results=min(limit,len(self.chunks)),include=["distances"])
            lexical = self.bm25.get_scores(terms(query))
            lex_ids = [self.chunks[i]["id"] for i in sorted(range(len(lexical)),key=lambda i:lexical[i],reverse=True) if lexical[i]>0][:limit]
            scores = {}
            for ranking in (result["ids"][0],lex_ids):
                for rank,doc_id in enumerate(ranking,1):
                    scores[doc_id] = scores.get(doc_id,0)+1/(60+rank)
            by_id = {x["id"]:x for x in self.chunks}
            rows = [Evidence.model_validate({**by_id[key],"score":value}).model_dump()
                    for key,value in sorted(scores.items(),key=lambda item:item[1],reverse=True)[:limit]]
            span_output(span,rows)
            return rows,usage
