import base64
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
IMAGE_MAX_EDGE = 1280


@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@st.cache_data(ttl=30, max_entries=4)
def installed_ollama_models() -> tuple[str, ...]:
    response = get_http_session().get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return tuple(model["name"] for model in response.json().get("models", []))


def ollama_generate(model: str, prompt: str, image: Optional[Image.Image] = None,
                    json_mode: bool = False, max_tokens: int = 256) -> str:
    payload: Dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False, "keep_alive": -1,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    if json_mode:
        payload["format"] = "json"
    if image is not None:
        image = resize_for_vision(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=82, optimize=True)
        payload["images"] = [base64.b64encode(buf.getvalue()).decode("utf-8")]
    response = get_http_session().post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def resize_for_vision(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    longest = max(image.size)
    if longest <= IMAGE_MAX_EDGE:
        return image
    scale = IMAGE_MAX_EDGE / longest
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        return {"score": 0, "feedback": text}


def generate_with_bonsai(prompt: str, endpoint: str, uploaded: Optional[Image.Image]) -> Optional[Image.Image]:
    """Call a local image endpoint, or return the uploaded image for review-only mode.

    The endpoint may return an image directly, {"image": base64}, or {"url": ...}.
    """
    if not endpoint:
        return uploaded
    response = get_http_session().post(endpoint, json={"prompt": prompt}, timeout=600)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    data = response.json()
    if data.get("image"):
        return Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    if data.get("url"):
        return Image.open(io.BytesIO(get_http_session().get(data["url"], timeout=120).content)).convert("RGB")
    raise ValueError("Bonsai endpoint did not return an image, base64 image, or image URL.")


def pull_model(model: str, installed: tuple[str, ...] = ()) -> str:
    """Pull only when the exact model tag is absent; Ollama reuses cached layers."""
    if model in installed:
        return f"{model} is already installed; skipped pull."
    result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800)
    return result.stdout or result.stderr


def prepare_models(models: list[str]) -> Dict[str, str]:
    unique_models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    installed = installed_ollama_models()
    output: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(2, len(unique_models))) as executor:
        futures = {executor.submit(pull_model, model, installed): model for model in unique_models}
        for future in as_completed(futures):
            model = futures[future]
            try:
                output[model] = future.result()
            except Exception as exc:
                output[model] = f"Failed: {exc}"
    installed_ollama_models.clear()
    return output


def warm_model(model: str) -> str:
    """Load a model into Ollama memory with a one-token no-op request."""
    try:
        ollama_generate(model, "Reply with OK.", max_tokens=1)
        return f"{model} warmed and kept alive."
    except Exception as exc:
        return f"{model} pull succeeded, but warm-up failed: {exc}"


def warm_models(models: list[str]) -> Dict[str, str]:
    unique_models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    with ThreadPoolExecutor(max_workers=min(2, len(unique_models))) as executor:
        futures = {executor.submit(warm_model, model): model for model in unique_models}
        return {futures[future]: future.result() for future in as_completed(futures)}


def download_mlx(repo: str, directory: str) -> str:
    Path(directory).expanduser().mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["hf", "download", repo, "--local-dir", str(Path(directory).expanduser())],
        capture_output=True, text=True, timeout=1800,
    )
    return result.stdout or result.stderr


def download_mlx_models(models: list[tuple[str, str]]) -> Dict[str, str]:
    with ThreadPoolExecutor(max_workers=min(2, len(models))) as executor:
        futures = {executor.submit(download_mlx, repo, directory): repo for repo, directory in models}
        output: Dict[str, str] = {}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                output[repo] = future.result()
            except Exception as exc:
                output[repo] = f"Failed: {exc}"
        return output


st.set_page_config(page_title="Feedback-Driven Image Pipeline", page_icon="🎨", layout="wide")
st.markdown("""
<style>
.pipeline {display:flex; align-items:center; gap:8px; margin:14px 0 24px; overflow-x:auto}
.node {padding:12px 14px; border-radius:10px; border:1px solid #d8dee9; min-width:125px; text-align:center; background:#f6f8fb}
.node b {display:block; font-size:14px}.node small {color:#5b6472}.arrow {font-size:24px; color:#8b95a5}
.hero {padding:18px 22px; border-radius:16px; background:linear-gradient(135deg,#eef7ff,#f8f2ff); margin-bottom:10px}
</style>
<div class="hero"><h1>Feedback-Driven Image Pipeline</h1>
<p>Enhance → Generate → Score → Fix defects</p></div>
<div class="pipeline">
<div class="node"><b>User Prompt</b><small>input</small></div><div class="arrow">→</div>
<div class="node"><b>Qwen3-4B</b><small>enhance</small></div><div class="arrow">→</div>
<div class="node"><b>Bonsai 4B</b><small>generate</small></div><div class="arrow">→</div>
<div class="node"><b>VisualQuality-R1</b><small>score + think</small></div><div class="arrow">→</div>
<div class="node"><b>Qwen3-4B</b><small>fix defects</small></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Local models")
    enhancer = st.text_input("Qwen3-4B enhancer / fixer", os.getenv("QWEN_MODEL", "qwen3:4b"))
    critic = st.text_input("Visual-quality critic", os.getenv("CRITIC_MODEL", "qwen2.5vl:7b"))
    bonsai_endpoint = st.text_input(
        "Image-generation endpoint",
        os.getenv("BONSAI_ENDPOINT", "http://127.0.0.1:8765/generate"),
        help="The local MFLUX bridge is prefilled. Leave this only when using upload-only review mode.",
    )
    max_rounds = st.slider("Fix rounds", 1, 3, 1)
    st.divider()
    st.subheader("Model setup")
    bonsai_repo = st.text_input("Bonsai 4B MLX repo", "prism-ml/Bonsai-4B-mlx-1bit")
    visualquality_repo = st.text_input("VisualQuality-R1 MLX repo", "mlx-community/VisualQuality-R1-7B-bf16")
    mlx_root = st.text_input("MLX models folder", "models/slide-models")
    if st.button("Pull slide models", width="stretch", help="Pull Qwen3-4B with Ollama and download Bonsai 4B + VisualQuality-R1 with Hugging Face MLX."):
        with st.spinner("Pulling slide models…"):
            try:
                ollama_results = prepare_models([enhancer])
                for model, output in ollama_results.items():
                    st.write(f"**{model}**")
                    st.code(output[-2000:])
                mlx_results = download_mlx_models([
                    (bonsai_repo, str(Path(mlx_root).expanduser() / "bonsai-4b")),
                    (visualquality_repo, str(Path(mlx_root).expanduser() / "visualquality-r1")),
                ])
                for model, output in mlx_results.items():
                    st.write(f"**{model}**")
                    st.code(output[-2000:])
                st.write("Warming Qwen3-4B…")
                for model, output in warm_models([enhancer]).items():
                    st.caption(output)
            except Exception as exc:
                st.error(str(exc))
    st.caption("Qwen3-4B uses Ollama. Bonsai 4B and VisualQuality-R1 use Apple MLX downloads because they are not standard Ollama registry tags.")

with st.form("pipeline_inputs", border=False):
    prompt = st.text_area("User prompt", "A cinematic editorial photo of a red fox in a snowy pine forest at sunrise, natural light, detailed fur", height=100)
    image_source = st.radio(
        "How should the image be provided?",
        ["Upload an existing image", "Generate a new image"],
        horizontal=True,
        help="Choose generation to send the enhanced prompt to your Bonsai image endpoint, or upload an image to review it with the feedback loop.",
    )
    uploaded_file = None
    if image_source == "Upload an existing image":
        uploaded_file = st.file_uploader("Add an image", type=["png", "jpg", "jpeg"])
    run = st.form_submit_button("Run pipeline", type="primary", width="stretch")
source_image = resize_for_vision(Image.open(uploaded_file)) if uploaded_file else None

if "result" not in st.session_state:
    st.session_state.result = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None

if run:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()
    if image_source == "Generate a new image" and not bonsai_endpoint:
        st.warning("Enter a Bonsai image endpoint in the sidebar to generate a new image, or choose ‘Upload an existing image’.")
        st.stop()
    if image_source == "Upload an existing image" and source_image is None:
        st.warning("Add an image to review, or choose ‘Generate a new image’.")
        st.stop()
    try:
        with st.status("Running feedback loop…", expanded=True) as status:
            st.write("Enhancing prompt with Qwen3-4B…")
            enhanced = ollama_generate(enhancer, f"Rewrite this image prompt for a high-quality generator. Return only the improved prompt.\n\n{prompt}", max_tokens=180)
            st.write("Generating with Bonsai 4B…" if image_source == "Generate a new image" else "Using uploaded image…")
            result = generate_with_bonsai(enhanced, bonsai_endpoint, source_image)
            if result is None:
                raise ValueError("Add a Bonsai endpoint or upload a source image.")
            for round_no in range(max_rounds):
                st.write(f"Scoring image with VisualQuality-R1 (round {round_no + 1})…")
                critique = parse_json(ollama_generate(
                    critic,
                    "Evaluate this image for prompt alignment, anatomy, composition, lighting, artifacts, and text errors. Return JSON with integer score 0-100, concise feedback, and a list of defects.",
                    image=result, json_mode=True, max_tokens=220,
                ))
                st.session_state.feedback = critique
                if int(critique.get("score", 0)) >= 90 or not bonsai_endpoint:
                    break
                st.write("Fixing defects with Qwen3-4B and regenerating…")
                enhanced = ollama_generate(enhancer, f"Improve this image prompt using the defects below. Return only the revised prompt.\nPrompt: {enhanced}\nDefects: {critique.get('defects', critique.get('feedback', ''))}", max_tokens=180)
                result = generate_with_bonsai(enhanced, bonsai_endpoint, result)
            st.session_state.result = result
            status.update(label="Pipeline complete", state="complete")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")

left, right = st.columns(2)
with left:
    st.subheader("Generated image")
    if st.session_state.result:
        st.image(st.session_state.result, width="stretch")
    else:
        st.info("Your image will appear here.")
with right:
    st.subheader("VisualQuality-R1 feedback")
    if st.session_state.feedback:
        feedback = st.session_state.feedback
        st.metric("Quality score", f"{feedback.get('score', '—')}/100")
        st.write(feedback.get("feedback", "No feedback returned."))
        defects = feedback.get("defects", [])
        if defects:
            st.write("Defects")
            for defect in defects:
                st.write(f"• {defect}")
    else:
        st.info("Run the pipeline to receive structured feedback.")
