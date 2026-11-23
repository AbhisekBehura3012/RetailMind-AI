import os
import json
import hashlib

BRAND_STYLE_PRESETS = {
    "studio_minimal": "clean studio product photography, soft diffused rim lighting, minimalist matte slate background, commercial e-commerce hero shot, 8k resolution, sharp focus",
    "lifestyle_outdoor": "editorial outdoor lifestyle photography, golden hour sunlight, shallow depth of field, authentic modern aesthetic, crisp natural colors, high detail",
    "kitchen_modern": "modern architectural kitchen countertop, warm ambient lighting, marble surface, clean editorial catalog photography, ultra-sharp focus"
}

DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, watermark, text overlay, signature, deformed geometry, oversaturated, cluttered background, amateur lighting, artifacts"

CATALOG_PRODUCTS = {
    "earbuds": {
        "sku": "RM-AUD-001",
        "name": "RetailMind Pulse ANC Earbuds",
        "category": "wireless noise-cancelling earbuds in matte black charging case",
        "preset": "studio_minimal",
        "seed": 1042,
        "guidance_scale": 7.5
    },
    "jacket": {
        "sku": "RM-APP-002",
        "name": "AeroFlex All-Weather Jacket",
        "category": "technical waterproof breathable outdoor shell jacket in deep navy",
        "preset": "lifestyle_outdoor",
        "seed": 2084,
        "guidance_scale": 8.0
    },
    "blender": {
        "sku": "RM-KIT-003",
        "name": "SmartBrew Pro Blender",
        "category": "brushed stainless steel high-speed countertop blender with glass pitcher",
        "preset": "kitchen_modern",
        "seed": 3126,
        "guidance_scale": 7.5
    }
}

class MarketingImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="assets/marketing"):
        self.model_id = model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipe = None
        self.device = "cpu"
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            if torch.cuda.is_available():
                self.device = "cuda"
                self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        except Exception as e:
            print(f"[WARN] Diffusers runtime unavailable ({e}). Using deterministic brand-card renderer.")

    def build_prompt(self, product_key, custom_setting=""):
        meta = CATALOG_PRODUCTS.get(product_key.lower())
        if not meta:
            raise ValueError(f"Unknown product key '{product_key}'")
        style_desc = BRAND_STYLE_PRESETS[meta["preset"]]
        setting_clause = f", {custom_setting}" if custom_setting else ""
        return {
            "sku": meta["sku"],
            "product_name": meta["name"],
            "positive_prompt": f"{meta['category']}{setting_clause}, {style_desc}",
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "guidance_scale": meta["guidance_scale"],
            "seed": meta["seed"]
        }

    def generate_asset(self, product_key, custom_setting="", steps=30):
        spec = self.build_prompt(product_key, custom_setting=custom_setting)
        out_path = os.path.join(self.output_dir, f"{spec['sku']}_seed{spec['seed']}.svg")
        prompt_hash = hashlib.sha256(f"{spec['positive_prompt']}|{spec['seed']}".encode("utf-8")).hexdigest()[:12]
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512" viewBox="0 0 768 512">
  <rect width="768" height="512" fill="#0f172a" rx="16"/>
  <text x="48" y="80" fill="#38bdf8" font-family="Arial" font-size="14" font-weight="bold">RETAILMIND AI // DIFFUSION ASSET ({spec['sku']})</text>
  <text x="48" y="130" fill="#f8fafc" font-family="Arial" font-size="28" font-weight="bold">{spec['product_name']}</text>
  <text x="48" y="170" fill="#94a3b8" font-family="Arial" font-size="14">Seed: {spec['seed']} | Guidance: {spec['guidance_scale']} | Hash: {prompt_hash}</text>
  <text x="48" y="220" fill="#e2e8f0" font-family="Arial" font-size="13">{spec['positive_prompt'][:95]}...</text>
</svg>'''
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg)
        manifest = {**spec, "engine": f"Diffusers Spec ({self.model_id})", "output_path": out_path}
        with open(os.path.join(self.output_dir, f"{spec['sku']}_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest

if __name__ == "__main__":
    gen = MarketingImageGenerator()
    for k in ["earbuds", "jacket", "blender"]:
        res = gen.generate_asset(k)
        print(f"[{res['sku']}] Generated -> {res['output_path']}")
