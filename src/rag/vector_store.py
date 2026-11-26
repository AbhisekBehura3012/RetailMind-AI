import os
import re
import json
import math
import numpy as np

# 1. Product Catalog + Store Policy & FAQ Knowledge Corpus
RAW_KNOWLEDGE_DOCS = [
    {
        "doc_id": "POL-RET-001",
        "doc_type": "policy",
        "sku": "ALL",
        "title": "Return, Exchange & Refund Policy",
        "text": (
            "RetailMind Store offers a 30-day hassle-free return window from the date of delivery for all unworn, "
            "unwashed, and undamaged items with original tags attached. Refunds are issued to the original payment "
            "method within 5 to 7 business days after warehouse inspection. Opened hygiene items and in-ear earbuds "
            "cannot be returned for buyer remorse, but qualify for a free 1-to-1 replacement if a hardware defect occurs "
            "within the warranty period. Final-sale clearance items marked at 50% off or more are non-refundable."
        )
    },
    {
        "doc_id": "POL-SHP-002",
        "doc_type": "policy",
        "sku": "ALL",
        "title": "Shipping Tiers, Transit Protection & International Delivery",
        "text": (
            "Standard Ground Shipping takes 3 to 5 business days and is complimentary on orders over $50 ($5.99 flat fee "
            "for orders under $50). Express Air Shipping delivers in 1 to 2 business days for $12.99. All shipments "
            "include live tracking and transit protection: if a package shows no carrier scan for more than 7 business "
            "days, RetailMind dispatches a free replacement order immediately. International shipping is available to "
            "28 countries in 6 to 10 business days with duties calculated at checkout."
        )
    },
    {
        "doc_id": "POL-LOY-003",
        "doc_type": "faq",
        "sku": "ALL",
        "title": "RetailMind Rewards & Price Adjustment FAQ",
        "text": (
            "RetailMind Rewards members earn 5 points per $1 spent, redeemable at 500 points for a $10 voucher. "
            "If an item you purchased goes on sale on RetailMind Store within 14 days of your order date, contact "
            "support with your order number for a one-time price adjustment credit to your account."
        )
    },
    {
        "doc_id": "CAT-AUD-001",
        "doc_type": "product",
        "sku": "RM-AUD-001",
        "title": "RetailMind Pulse ANC Earbuds ($79.99)",
        "text": (
            "Product SKU RM-AUD-001: RetailMind Pulse ANC Earbuds priced at $79.99 in Matte Black. "
            "Features hybrid active noise cancellation (-38dB), 32-hour total battery life (8 hours per charge plus "
            "24 hours in the USB-C wireless charging case), IPX5 sweat and rain resistance, dual beamforming mics "
            "for clear calls, and Bluetooth 5.3 multipoint pairing. Backed by a 1-year limited hardware warranty."
        )
    },
    {
        "doc_id": "CAT-APP-002",
        "doc_type": "product",
        "sku": "RM-APP-002",
        "title": "AeroFlex All-Weather Jacket ($119.00)",
        "text": (
            "Product SKU RM-APP-002: AeroFlex All-Weather Jacket priced at $119.00 in Deep Navy and Slate Grey. "
            "Engineered with a 3-layer 15,000mm waterproof and breathable membrane, fully seam-sealed construction, "
            "YKK AquaGuard waterproof zippers, and articulated underarm vents. Machine wash cold on gentle cycle "
            "with technical wash; do not use fabric softener or bleach. Covered by a 2-year seam and zipper warranty."
        )
    },
    {
        "doc_id": "CAT-KIT-003",
        "doc_type": "product",
        "sku": "RM-KIT-003",
        "title": "SmartBrew Pro Blender ($149.50)",
        "text": (
            "Product SKU RM-KIT-003: SmartBrew Pro Blender priced at $149.50 in Brushed Stainless Steel. "
            "Equipped with a 1200-watt high-torque motor, 6-blade hardened titanium assembly for crushing ice and "
            "frozen fruit, 64-ounce BPA-free Tritan pitcher, and automatic thermal shutoff protection. Dishwasher-safe "
            "pitcher and lid; covered by a 2-year motor warranty."
        )
    },
    {
        "doc_id": "CAT-APP-004",
        "doc_type": "product",
        "sku": "RM-APP-004",
        "title": "TerraStride Trail Running Shoes ($134.00)",
        "text": (
            "Product SKU RM-APP-004: TerraStride Trail Running Shoes priced at $134.00 in Terracotta Amber. "
            "Built with dual-density nitrogen-infused EVA midsole foam, 4.5mm Vibram-grade multidirectional lugs "
            "for wet rock traction, breathable ripstop mesh upper, and a 6mm heel-to-toe drop. True-to-size athletic fit."
        )
    }
]


def recursive_chunk_text(text: str, chunk_size: int = 260, chunk_overlap: int = 50) -> list[str]:
    """Splits text into overlapping semantic passages using LangChain if available or pure Python."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", ", ", " ", ""]
        )
        return splitter.split_text(text)
    except Exception:
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                period_idx = text.rfind(". ", start, end)
                if period_idx > start + int(chunk_size * 0.5):
                    end = period_idx + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= n:
                break
            start = max(end - chunk_overlap, start + 1)
        return chunks


class RetailVectorStore:
    """Chunks, embeds, persists, and queries product + policy documents for RetailMind RAG."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "models/vector_db",
        chunk_size: int = 260,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.makedirs(self.persist_dir, exist_ok=True)

        self.st_model = None
        self.faiss_mod = None
        self.vocab = {}
        self.idf = None
        self.embeddings = None
        self.faiss_index = None
        self.chunks_meta = []

        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"[WARN] SentenceTransformers runtime unavailable ({e}). Using L2-normalized TF-IDF dense projection.")

        try:
            import faiss
            self.faiss_mod = faiss
        except Exception:
            self.faiss_mod = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _fit_fallback_vectorizer(self, texts: list[str]):
        doc_freq = {}
        tokenized_docs = [self._tokenize(t) for t in texts]
        for tokens in tokenized_docs:
            for tok in set(tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        sorted_terms = sorted(doc_freq.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}
        n_docs = len(texts)
        self.idf = np.array(
            [math.log((n_docs + 1) / (doc_freq[t] + 1)) + 1.0 for t in sorted_terms],
            dtype=np.float32
        )

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.st_model is not None:
            vecs = self.st_model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)

        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            for tok in tokens:
                col = self.vocab.get(tok)
                if col is not None:
                    mat[row_idx, col] += 1.0
        mat = mat * self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms

    def build_and_persist(self, docs: list[dict] = None) -> dict:
        docs = docs or RAW_KNOWLEDGE_DOCS
        self.chunks_meta = []
        chunk_texts = []

        for doc in docs:
            passages = recursive_chunk_text(
                doc["text"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            for idx, passage in enumerate(passages):
                enriched_text = f"[{doc['title']}] {passage}"
                self.chunks_meta.append({
                    "chunk_id": f"{doc['doc_id']}_C{idx+1}",
                    "doc_id": doc["doc_id"],
                    "doc_type": doc["doc_type"],
                    "sku": doc["sku"],
                    "title": doc["title"],
                    "text": passage
                })
                chunk_texts.append(enriched_text)

        if self.st_model is None:
            self._fit_fallback_vectorizer(chunk_texts)

        self.embeddings = self._encode_texts(chunk_texts)
        dim = self.embeddings.shape[1]

        # Persist metadata + embeddings + optional FAISS binary index
        meta_path = os.path.join(self.persist_dir, "chunks_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "backend": "SentenceTransformers+FAISS" if (self.st_model and self.faiss_mod) else "DenseNormalizedNumPy",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "vocab": self.vocab,
                "chunks": self.chunks_meta
            }, f, indent=2)

        npz_path = os.path.join(self.persist_dir, "vector_index.npz")
        np.savez_compressed(
            npz_path,
            embeddings=self.embeddings,
            idf=self.idf if self.idf is not None else np.array([], dtype=np.float32)
        )

        if self.faiss_mod is not None:
            self.faiss_index = self.faiss_mod.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            self.faiss_mod.write_index(self.faiss_index, os.path.join(self.persist_dir, "index.faiss"))

        return {
            "documents_indexed": len(docs),
            "chunks_created": len(self.chunks_meta),
            "embedding_dim": int(dim),
            "persist_dir": self.persist_dir
        }

    def load(self) -> bool:
        meta_path = os.path.join(self.persist_dir, "chunks_metadata.json")
        npz_path = os.path.join(self.persist_dir, "vector_index.npz")
        if not (os.path.exists(meta_path) and os.path.exists(npz_path)):
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.chunks_meta = payload["chunks"]
        self.vocab = payload.get("vocab", {})

        arrays = np.load(npz_path)
        self.embeddings = arrays["embeddings"]
        if arrays["idf"].size > 0:
            self.idf = arrays["idf"]

        faiss_path = os.path.join(self.persist_dir, "index.faiss")
        if self.faiss_mod is not None and os.path.exists(faiss_path):
            self.faiss_index = self.faiss_mod.read_index(faiss_path)

        return True

    def similarity_search(self, query: str, top_k: int = 3, doc_type_filter: str = None) -> list[dict]:
        if self.embeddings is None:
            if not self.load():
                self.build_and_persist()

        q_vec = self._encode_texts([query])
        if self.faiss_index is not None and doc_type_filter is None:
            scores, indices = self.faiss_index.search(q_vec, top_k)
            ranked = [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]
        else:
            all_scores = (self.embeddings @ q_vec[0]).tolist()
            scored_pairs = []
            for idx, sc in enumerate(all_scores):
                if doc_type_filter and self.chunks_meta[idx]["doc_type"] != doc_type_filter:
                    continue
                scored_pairs.append((idx, float(sc)))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            ranked = scored_pairs[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            meta = self.chunks_meta[idx]
            results.append({
                "rank": rank,
                "chunk_id": meta["chunk_id"],
                "doc_type": meta["doc_type"],
                "sku": meta["sku"],
                "title": meta["title"],
                "score": round(score, 4),
                "passage": meta["text"]
            })
        return results


if __name__ == "__main__":
    store = RetailVectorStore()
    stats = store.build_and_persist()
    print("=== DAY 18 RAG VECTOR STORE BUILT & PERSISTED ===")
    print(json.dumps(stats, indent=2))

    sample_queries = [
        "Can I return opened earbuds if they are defective?",
        "How long does express shipping take and what if a package is lost in transit?",
        "Which jacket has waterproof zippers and how should I wash it?"
    ]
    for q in sample_queries:
        print(f"\nQuery: '{q}'")
        for hit in store.similarity_search(q, top_k=2):
            print(f"  #{hit['rank']} [{hit['chunk_id']} | {hit['doc_type']}] (score={hit['score']}): {hit['passage'][:110]}...")
