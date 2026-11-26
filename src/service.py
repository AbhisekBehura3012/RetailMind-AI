from src.assistant.llm_assistant import ReviewThemeSummarizer, RetailSupportChatbot
from src.creative.marketing_generator import MarketingImageGenerator
from src.creative.variation_video_studio import VariationAndVideoStudio
from src.multimodal.voice_and_clip import TTSVoiceEngine, CLIPMultimodalSearch
from src.rag.vector_store import RetailVectorStore


class RetailMindPlatformService:
    """Unified Enterprise Platform Service (Days 14-18: Generation Track + Persisted RAG Vector DB)."""

    def __init__(self):
        self.summarizer = ReviewThemeSummarizer()
        self.chatbot = RetailSupportChatbot()
        self.creative_studio = MarketingImageGenerator()
        self.variation_studio = VariationAndVideoStudio()
        self.voice_engine = TTSVoiceEngine()
        self.multimodal_search = CLIPMultimodalSearch()
        self.vector_store = RetailVectorStore()
        if not self.vector_store.load():
            self.vector_store.build_and_persist()

    def ask_support_with_voice(self, message: str) -> dict:
        text_reply = self.chatbot.chat(message)
        audio_meta = self.voice_engine.speak_reply(text_reply, tag="support_voice")
        return {
            "text_reply": text_reply,
            "audio_path": audio_meta["audio_path"],
            "tts_engine": audio_meta["engine"]
        }

    def search_catalog_text(self, query: str, top_k: int = 2) -> list[dict]:
        return self.multimodal_search.search_by_text(query, top_k=top_k)

    def retrieve_knowledge(self, query: str, top_k: int = 3, doc_type: str = None) -> list[dict]:
        return self.vector_store.similarity_search(query, top_k=top_k, doc_type_filter=doc_type)


if __name__ == "__main__":
    service = RetailMindPlatformService()
    print("[ACTIVE SERVICE] RetailMind AI — Day 18 RAG Vector Knowledge Base Loaded\n")

    hits = service.retrieve_knowledge("What is the warranty on the SmartBrew blender?", top_k=2)
    for h in hits:
        print(f"Rank #{h['rank']} [{h['chunk_id']}] (score={h['score']}): {h['passage']}")
