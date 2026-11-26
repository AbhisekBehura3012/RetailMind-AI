from src.assistant.llm_assistant import ReviewThemeSummarizer, RetailSupportChatbot
from src.creative.marketing_generator import MarketingImageGenerator
from src.creative.variation_video_studio import VariationAndVideoStudio
from src.multimodal.voice_and_clip import TTSVoiceEngine, CLIPMultimodalSearch


class RetailMindPlatformService:
    """Unified Generation-Track Platform Service (Days 14-17: LLM, Diffusion, Video, TTS Voice & CLIP Search)."""

    def __init__(self):
        self.summarizer = ReviewThemeSummarizer()
        self.chatbot = RetailSupportChatbot()
        self.creative_studio = MarketingImageGenerator()
        self.variation_studio = VariationAndVideoStudio()
        self.voice_engine = TTSVoiceEngine()
        self.multimodal_search = CLIPMultimodalSearch()

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

    def search_catalog_image(self, image_path: str, top_k: int = 2) -> list[dict]:
        return self.multimodal_search.search_by_image(image_path, top_k=top_k)


if __name__ == "__main__":
    service = RetailMindPlatformService()
    print("[ACTIVE SERVICE] RetailMind AI — Generation Track Complete (Days 14–17)\n")

    voice_resp = service.ask_support_with_voice("What is the battery life and warranty on the Pulse ANC Earbuds?")
    print("Voice Support Reply :", voice_resp["text_reply"])
    print("Synthesized WAV     :", voice_resp["audio_path"], "\n")

    print("CLIP Text Search ('wireless black gym audio'):")
    for r in service.search_catalog_text("wireless black gym audio", top_k=2):
        print(f"  -> {r['sku']}: {r['name']} (score={r['similarity']})")
