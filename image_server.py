"""Small local HTTP bridge for MFLUX text-to-image generation."""

import base64
import io
import json
import os
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

from mflux.models.z_image import ZImageTurbo


HOST = os.getenv("MFLUX_HOST", "127.0.0.1")
PORT = int(os.getenv("MFLUX_PORT", "8765"))
WIDTH = int(os.getenv("MFLUX_WIDTH", "768"))
HEIGHT = int(os.getenv("MFLUX_HEIGHT", "768"))
STEPS = int(os.getenv("MFLUX_STEPS", "4"))
QUANTIZE = int(os.getenv("MFLUX_QUANTIZE", "8"))
model = None


def get_model():
    global model
    if model is None:
        print("Loading MFLUX Z-Image Turbo; the first run downloads model weights…", flush=True)
        model = ZImageTurbo(quantize=QUANTIZE)
    return model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model_loaded": model is not None})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "Use POST /generate"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(request.get("prompt", "")).strip()
            if not prompt:
                self._json(400, {"error": "prompt is required"})
                return
            generated = get_model().generate_image(
                seed=random.randint(0, 1_000_000_000),
                prompt=prompt,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
            )
            image = getattr(generated, "image", generated)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            self._json(200, {"image": base64.b64encode(buffer.getvalue()).decode("ascii")})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


if __name__ == "__main__":
    print(f"MFLUX image endpoint listening at http://{HOST}:{PORT}/generate", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()
