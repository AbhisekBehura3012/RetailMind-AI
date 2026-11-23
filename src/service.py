from src.assistant.llm_assistant import ReviewThemeSummarizer, RetailSupportChatbot
from src.creative.marketing_generator import MarketingImageGenerator
from src.creative.variation_video_studio import VariationAndVideoStudio


class RetailMindPlatformService:
    """Unified GenAI Platform Service (Days 14-16: LLM Assistant + Diffusion Creative & Video Studio)."""

    def __init__(self):
        self.summarizer = ReviewThemeSummarizer()
        self.chatbot = RetailSupportChatbot()
        self.creative_studio = MarketingImageGenerator()
        self.variation_studio = VariationAndVideoStudio()

    def summarize_product_reviews(self, product_name: str, reviews: list[str]) -> dict:
        return self.summarizer.summarize_reviews(product_name, reviews)

    def ask_support_bot(self, message: str) -> str:
        return self.chatbot.chat(message)

    def generate_marketing_visual(self, product_key: str, custom_setting: str = "") -> dict:
        return self.creative_studio.generate_asset(product_key, custom_setting=custom_setting)

    def create_campaign_pack(self, product_key: str) -> dict:
        base_asset = self.creative_studio.generate_asset(product_key)
        variations = self.variation_studio.generate_variations(product_key, init_image_path=base_asset["output_path"])
        promo_clip = self.variation_studio.generate_promo_clip(product_key)
        return {
            "base_hero": base_asset["output_path"],
            "variations": [v["output_path"] for v in variations],
            "promo_clip": promo_clip["output_path"]
        }


if __name__ == "__main__":
    service = RetailMindPlatformService()
    print("[ACTIVE SERVICE] RetailMind GenAI Studio (Day 14 Assistant + Days 15-16 Diffusion & Video)\n")

    pack = service.create_campaign_pack("earbuds")
    print("Hero Asset :", pack["base_hero"])
    for var_path in pack["variations"]:
        print("Variation  :", var_path)
    print("Promo Clip :", pack["promo_clip"])
