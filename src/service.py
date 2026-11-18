from src.assistant.llm_assistant import ReviewThemeSummarizer, RetailSupportChatbot

class RetailMindAssistantService:
    def __init__(self):
        self.summarizer = ReviewThemeSummarizer()
        self.chatbot = RetailSupportChatbot()

    def summarize_product_reviews(self, product_name: str, reviews: list[str]) -> dict:
        return self.summarizer.summarize_reviews(product_name, reviews)

    def ask_support_bot(self, message: str) -> str:
        return self.chatbot.chat(message)

if __name__ == "__main__":
    service = RetailMindAssistantService()
    print("[ACTIVE SERVICE] Day 14 LLM Assistant\n")
    print(service.summarize_product_reviews("AeroFlex Jacket", [
        "Kept me dry during a heavy downpour.",
        "Windproof shell and waterproof zippers fit true to size."
    ]))
    print("\nCustomer : How long does standard shipping take?")
    print("Assistant:", service.ask_support_bot("How long does standard shipping take?"))
