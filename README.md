# Feedback-Driven Image Pipeline

Local Streamlit UI for the pipeline in the reference slide:

`User prompt → Qwen3-4B enhance → Bonsai 4B generate → vision critic score → Qwen3-4B fix`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull qwen3:4b
streamlit run app.py
```

The critic needs a vision-capable Ollama model. The Bonsai field expects a local HTTP image-generation service that accepts `{"prompt":"..."}` and returns an image, `{"image":"<base64>"}`, or `{"url":"..."}`. If no endpoint is configured, upload a source image to exercise the enhancement and scoring loop.

The sidebar's **Pull slide models** button downloads the exact models represented in the slide:

- `qwen3:4b` — via Ollama
- `prism-ml/Bonsai-4B-mlx-1bit` — via Hugging Face MLX
- `mlx-community/VisualQuality-R1-7B-bf16` — via Hugging Face MLX

**Note:** MLX weights and Ollama model files are separate formats; the downloaded MLX folders are not automatically importable into Ollama. Bonsai and VisualQuality-R1 need an MLX-compatible serving/inference adapter to replace the current Bonsai HTTP endpoint and Ollama vision critic at runtime.

## Latency Optimizations

- **Prepare pipeline models** checks `/api/tags`, skips models already installed, and pulls missing models concurrently
- Ollama requests reuse a cached HTTP session, warm both models in parallel, and set `keep_alive: -1`, so Qwen and the vision critic remain loaded between pipeline runs
- Vision inputs are resized to a maximum 1280px edge and JPEG-compressed before upload
- Prompt-only calls use small output limits and low temperature
- The input form prevents a full Streamlit rerun on every keystroke or file selection

**Tip for low-memory systems:** Change `keep_alive` in `app.py` from `-1` to a duration such as `"10m"` so Ollama can unload idle models.
