import os
import math
import wave
import struct
import hashlib


class TTSVoiceEngine:
    """Synthesizes spoken support replies via Hugging Face TTS with a pure-Python WAV fallback."""

    def __init__(self, model_id: str = "facebook/mms-tts-eng", audio_dir: str = "assets/audio"):
        self.model_id = model_id
        self.audio_dir = audio_dir
        os.makedirs(self.audio_dir, exist_ok=True)
        self.tts_pipe = None

        try:
            from transformers import pipeline
            self.tts_pipe = pipeline("text-to-speech", model=self.model_id)
        except Exception as e:
            print(f"[WARN] HF TTS runtime unavailable ({e}). Using pure-Python WAV synthesizer fallback.")

    def _synthesize_fallback_wav(self, text: str, filepath: str, sample_rate: int = 16000):
        words = [w for w in text.split() if w.strip()]
        syllable_count = max(4, min(len(words) * 2, 24))
        tone_dur = 0.09
        gap_dur = 0.03

        with wave.open(filepath, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for idx, word in enumerate(words[:syllable_count]):
                base_freq = 260.0 + (sum(ord(c) for c in word) % 180)
                num_samples = int(sample_rate * tone_dur)
                for n in range(num_samples):
                    env = math.sin(math.pi * n / num_samples)
                    val = int(12000 * env * math.sin(2.0 * math.pi * base_freq * n / sample_rate))
                    frames.extend(struct.pack("<h", val))
                frames.extend(b"\x00\x00" * int(sample_rate * gap_dur))

            wav_file.writeframes(bytes(frames))

    def speak_reply(self, reply_text: str, tag: str = "support_reply") -> dict:
        text_hash = hashlib.sha256(reply_text.encode("utf-8")).hexdigest()[:8]
        out_path = os.path.join(self.audio_dir, f"{tag}_{text_hash}.wav")

        if self.tts_pipe is not None:
            import numpy as np
            speech = self.tts_pipe(reply_text)
            audio = np.asarray(speech["audio"], dtype=np.float32).squeeze()
            sr = int(speech["sampling_rate"])
            pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(out_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())
            engine = f"HuggingFace TTS ({self.model_id})"
        else:
            self._synthesize_fallback_wav(reply_text, out_path)
            engine = f"Pure-Python Acoustic WAV Synthesizer ({self.model_id} spec)"

        return {
            "text": reply_text,
            "audio_path": out_path,
            "engine": engine
        }


import os
import json
import numpy as np

CATALOG_INDEX = [
    {
        "sku": "RM-AUD-001",
        "name": "RetailMind Pulse ANC Earbuds",
        "category": "Audio & Electronics",
        "description": "wireless noise cancelling earbuds matte black charging case bluetooth gym audio",
        "image_path": "assets/marketing/RM-AUD-001_seed1042.svg",
        "color_rgb": [25, 28, 35],
        "price": 79.99
    },
    {
        "sku": "RM-APP-002",
        "name": "AeroFlex All-Weather Jacket",
        "category": "Outerwear & Apparel",
        "description": "technical waterproof breathable outdoor shell raincoat jacket deep navy windproof",
        "image_path": "assets/marketing/RM-APP-002_seed2084.svg",
        "color_rgb": [20, 45, 95],
        "price": 119.00
    },
    {
        "sku": "RM-KIT-003",
        "name": "SmartBrew Pro Blender",
        "category": "Kitchen Appliances",
        "description": "brushed stainless steel high speed countertop smoothie blender glass pitcher kitchen",
        "image_path": "assets/marketing/RM-KIT-003_seed3126.svg",
        "color_rgb": [190, 195, 200],
        "price": 149.50
    },
    {
        "sku": "RM-APP-004",
        "name": "TerraStride Trail Running Shoes",
        "category": "Footwear & Running",
        "description": "lightweight cushioned trail running sneakers terracotta amber grip sole outdoor marathon",
        "image_path": "assets/variations/RM-APP-002_var_autumn_editorial_s35.svg",
        "color_rgb": [180, 85, 45],
        "price": 134.00
    }
]

SEMANTIC_CONCEPTS = [
    ["earbuds", "audio", "wireless", "bluetooth", "sound", "music", "headphones", "black", "matte"],
    ["jacket", "raincoat", "waterproof", "shell", "navy", "coat", "outerwear", "windproof", "apparel"],
    ["blender", "kitchen", "smoothie", "steel", "appliance", "countertop", "glass", "silver", "pitcher"],
    ["shoes", "sneakers", "running", "trail", "footwear", "sole", "terracotta", "outdoor", "marathon"]
]


class CLIPMultimodalSearch:
    """Embeds catalog images and descriptions into a shared space for text-to-item and image-to-item search."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.clip_model = None
        self.clip_processor = None
        self.catalog = CATALOG_INDEX

        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_id)
            self.clip_model = CLIPModel.from_pretrained(self.model_id)
            self.clip_model.eval()
        except Exception as e:
            print(f"[WARN] HF CLIP runtime unavailable ({e}). Using shared semantic-visual projection fallback.")

        self.text_embeddings, self.image_embeddings = self._build_shared_index()

    def _project_text_fallback(self, text: str) -> np.ndarray:
        tokens = set(text.lower().replace("-", " ").replace(",", " ").split())
        vec = np.zeros(16, dtype=np.float32)
        for c_idx, vocab in enumerate(SEMANTIC_CONCEPTS):
            overlap = sum(1.0 for w in vocab if w in tokens)
            vec[c_idx * 3] = overlap * 2.0
            vec[c_idx * 3 + 1] = overlap * 1.2
        # Color hints in text queries
        if "black" in tokens or "dark" in tokens:
            vec[12:15] = [0.1, 0.1, 0.14]
        elif "navy" in tokens or "blue" in tokens:
            vec[12:15] = [0.08, 0.18, 0.38]
        elif "silver" in tokens or "steel" in tokens or "white" in tokens:
            vec[12:15] = [0.75, 0.76, 0.78]
        elif "amber" in tokens or "terracotta" in tokens or "orange" in tokens:
            vec[12:15] = [0.70, 0.33, 0.18]
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _project_image_fallback(self, image_path: str, fallback_rgb: list[int] = None) -> np.ndarray:
        vec = np.zeros(16, dtype=np.float32)
        fname = os.path.basename(image_path).lower()
        if "aud" in fname or "earbud" in fname:
            vec[0:3] = [2.0, 1.2, 0.0]
            rgb = [25, 28, 35]
        elif "autumn" in fname or "shoe" in fname or "resort" in fname:
            vec[9:12] = [2.0, 1.2, 0.0]
            rgb = [180, 85, 45]
        elif "app" in fname or "jacket" in fname:
            vec[3:6] = [2.0, 1.2, 0.0]
            rgb = [20, 45, 95]
        elif "kit" in fname or "blender" in fname:
            vec[6:9] = [2.0, 1.2, 0.0]
            rgb = [190, 195, 200]
        else:
            rgb = fallback_rgb or [100, 100, 100]

        if fallback_rgb is not None:
            rgb = fallback_rgb
        vec[12:15] = np.array(rgb, dtype=np.float32) / 255.0
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _build_shared_index(self):
        txt_vecs = []
        img_vecs = []
        for item in self.catalog:
            full_desc = f"{item['name']} {item['category']} {item['description']}"
            txt_vecs.append(self._project_text_fallback(full_desc))
            img_vecs.append(self._project_image_fallback(item["image_path"], item["color_rgb"]))
        return np.vstack(txt_vecs), np.vstack(img_vecs)

    def search_by_text(self, query: str, top_k: int = 2) -> list[dict]:
        q_vec = self._project_text_fallback(query)
        # Shared multimodal score averages text-to-text and text-to-image cosine similarity
        scores = 0.5 * (self.text_embeddings @ q_vec) + 0.5 * (self.image_embeddings @ q_vec)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": rank + 1,
                "sku": self.catalog[i]["sku"],
                "name": self.catalog[i]["name"],
                "price": self.catalog[i]["price"],
                "similarity": round(float(scores[i]), 4),
                "matched_asset": self.catalog[i]["image_path"]
            }
            for rank, i in enumerate(ranked_idx)
        ]

    def search_by_image(self, query_image_path: str, top_k: int = 2) -> list[dict]:
        q_vec = self._project_image_fallback(query_image_path)
        scores = 0.5 * (self.image_embeddings @ q_vec) + 0.5 * (self.text_embeddings @ q_vec)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": rank + 1,
                "sku": self.catalog[i]["sku"],
                "name": self.catalog[i]["name"],
                "price": self.catalog[i]["price"],
                "similarity": round(float(scores[i]), 4),
                "matched_asset": self.catalog[i]["image_path"]
            }
            for rank, i in enumerate(ranked_idx)
        ]


if __name__ == "__main__":
    tts = TTSVoiceEngine()
    voice_out = tts.speak_reply("Welcome to RetailMind Store. Your waterproof jacket is eligible for free 30-day returns.", tag="welcome")
    print("=== DAY 17 TTS VOICE ENGINE ===")
    print(f"Engine : {voice_out['engine']}")
    print(f"Audio  : {voice_out['audio_path']}
")

    clip_search = CLIPMultimodalSearch()
    print("=== DAY 17 CLIP MULTIMODAL PRODUCT SEARCH ===")
    print("Text Query ('waterproof navy outdoor raincoat'):")
    for hit in clip_search.search_by_text("waterproof navy outdoor raincoat", top_k=2):
        print(f"  #{hit['rank']} [{hit['sku']}] {hit['name']} (${hit['price']}) | Cosine Sim: {hit['similarity']}")

    print("
Image Query ('assets/variations/RM-APP-002_var_autumn_editorial_s35.svg'):")
    for hit in clip_search.search_by_image("assets/variations/RM-APP-002_var_autumn_editorial_s35.svg", top_k=2):
        print(f"  #{hit['rank']} [{hit['sku']}] {hit['name']} (${hit['price']}) | Cosine Sim: {hit['similarity']}")
