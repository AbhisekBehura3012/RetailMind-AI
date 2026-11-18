import os
import json
import urllib.request

RETAIL_KNOWLEDGE_BASE = """
BRAND: RetailMind Store
RETURN & REFUND POLICY:
- Customers may return unworn, undamaged items within 30 days of delivery for a full refund.
- Refunds are processed back to the original payment method within 5-7 business days after warehouse inspection.
- Final-sale items and opened hygiene/earbud products are eligible for exchange only if defective.

SHIPPING & DELIVERY POLICY:
- Standard shipping takes 3-5 business days (free on orders over ).
- Express shipping takes 1-2 business days (.99 flat rate).
- Orders stuck in transit for more than 7 business days qualify for a free replacement shipment.

PRODUCT CATALOG HIGHLIGHTS:
1. RetailMind Pulse ANC Earbuds (.99): 32-hour battery life, IPX5 water resistance, Bluetooth 5.3, 1-year warranty.
2. AeroFlex All-Weather Jacket (.00): Windproof, breathable membrane, YKK waterproof zippers, machine washable cold.
3. SmartBrew Pro Blender (.50): 1200W motor with auto-shutoff thermal protection, 2-year motor warranty.
"""

SYSTEM_PROMPT_TEMPLATE = """You are the official RetailMind Store Support Assistant.
Follow these strict rules:
1. Tone: Warm, concise, professional, and on-brand.
2. Grounding: Answer ONLY using the verified Product & Policy Knowledge Base provided below.
3. Guardrail: If a customer asks about something not covered in the knowledge base, politely state that you do not have that information and offer to escalate to a human specialist. Never invent policies, prices, or promo codes.

=== VERIFIED KNOWLEDGE BASE ===
{knowledge_base}
===============================
"""

class ReviewThemeSummarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.summarizer = None
        try:
            from transformers import pipeline
            self.summarizer = pipeline("summarization", model=self.model_name)
        except Exception as e:
            print(f"[WARN] HF Transformers pipeline unavailable ({e}). Using extractive fallback.")

    def summarize_reviews(self, product_name: str, reviews: list[str]) -> dict:
        combined_text = " ".join(r.strip() for r in reviews)
        if self.summarizer is not None:
            result = self.summarizer(combined_text[:2500], max_length=75, min_length=25, do_sample=False)
            summary_text = result[0]["summary_text"].strip()
            engine = f"HuggingFace ({self.model_name})"
        else:
            sentences = [s.strip() for s in combined_text.split(".") if len(s.strip()) > 15]
            summary_text = ". ".join(sentences[:3]) + "."
            engine = "Extractive Fallback (Pure Python)"

        return {
            "product": product_name,
            "review_count": len(reviews),
            "engine": engine,
            "theme_summary": summary_text
        }

class RetailSupportChatbot:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(knowledge_base=RETAIL_KNOWLEDGE_BASE.strip())
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _call_llm_api(self) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": self.history, "temperature": 0.2, "max_tokens": 250}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def _offline_grounded_reply(self, user_message: str) -> str:
        msg = user_message.lower()
        if "return" in msg or "refund" in msg:
            return "Unworn, undamaged items can be returned within 30 days for a full refund (5-7 business days). Opened earbuds qualify for exchange only if defective."
        if "ship" in msg or "delivery" in msg or "transit" in msg:
            return "Standard shipping takes 3-5 business days (free over ); Express takes 1-2 business days (.99). Orders stuck over 7 business days get a free replacement."
        if "earbud" in msg or "battery" in msg or "warranty" in msg:
            return "RetailMind Pulse ANC Earbuds (.99) offer 32-hour battery life, IPX5 water resistance, Bluetooth 5.3, and a 1-year warranty."
        return "That detail isn't in our current policy guide. Would you like me to connect you with a RetailMind support specialist?"

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        if self.api_key:
            try:
                reply = self._call_llm_api()
            except Exception as e:
                reply = f"[API Fallback] {self._offline_grounded_reply(user_message)} ({e})"
        else:
            reply = self._offline_grounded_reply(user_message)
        self.history.append({"role": "assistant", "content": reply})
        return reply
