"""Name the foods on the plate with a VLM, from one or two frames of the scan.

    names = name_foods([frame_a_rgb, frame_b_rgb], backend="gemini")
    # -> [{"name": "white rice", "ingredients": ["rice"], "container": "bowl"}, ...]

Backends
- "gemini": google-genai SDK, model from $NAMING_MODEL (default gemini-3-flash-preview... set it),
  key in $GEMINI_API_KEY. Best food naming in the benchmarks we found (DiningBench 81.8 %).
- "openai": any OpenAI-compatible chat endpoint ($OPENAI_BASE_URL, $OPENAI_API_KEY, $NAMING_MODEL),
  which is how a vLLM-served Qwen3-VL on the cluster is reached.

Design choices from the literature (docs/model_survey.md §5): two views, not more (the
1->2 view gain is where multi-view helps; small models degrade beyond 3); no chain of
thought for naming; strict JSON schema; short plain names that map onto FNDDS.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Sequence

import numpy as np
from PIL import Image

PROMPT = (
    "You are labelling a photo of a meal for a nutrition study. List every distinct food item on "
    "the plate or in the bowl. For each item give a short common English name as a diner would say it "
    "(e.g. 'white rice', 'grilled chicken breast', 'steamed broccoli'), its main ingredients, and the "
    "container it sits in ('plate', 'bowl', 'cup', 'other'). Do not list utensils, the table, drinks, "
    "or the container itself as items. Do not merge distinct items. Do not explain. "
    "Answer with JSON only: {\"items\": [{\"name\": str, \"ingredients\": [str], \"container\": str}]}"
)


def _jpeg_b64(rgb: np.ndarray, max_side: int = 1024, quality: int = 90) -> str:
    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _parse(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    items = data.get("items", data if isinstance(data, list) else [])
    out = []
    for it in items:
        name = str(it.get("name", "")).strip().lower()
        if name:
            out.append({"name": name,
                        "ingredients": [str(x).lower() for x in it.get("ingredients", [])],
                        "container": str(it.get("container", "plate")).lower()})
    return out


def name_foods(frames: Sequence[np.ndarray], backend: str = "gemini", model: str | None = None) -> list[dict]:
    model = model or os.environ.get("NAMING_MODEL")
    if backend == "gemini":
        return _gemini(frames, model or "gemini-3-flash-preview")
    if backend == "openai":
        return _openai(frames, model or "Qwen/Qwen3-VL-8B-Instruct")
    raise ValueError(backend)


def _gemini(frames, model):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    parts = [types.Part.from_bytes(data=base64.b64decode(_jpeg_b64(f)), mime_type="image/jpeg") for f in frames]
    parts.append(PROMPT)
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
    return _parse(resp.text)


def _openai(frames, model):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(f)}"}} for f in frames]
    content.append({"type": "text", "text": PROMPT})
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], temperature=0.0,
        response_format={"type": "json_object"})
    return _parse(resp.choices[0].message.content)


def pick_views(scan, n: int = 2) -> list[np.ndarray]:
    """One near-overhead frame and one oblique frame: the camera whose optical axis is
    most aligned with gravity (ARKit world +Y, i.e. -Y in OpenCV world is 'down') and
    the frame a quarter of the scan later."""
    frames = scan.good_frames()
    if not frames:
        raise RuntimeError("no frames")
    down = np.array([0.0, -1.0, 0.0])                       # ARKit world +Y up -> OpenCV-converted poses keep world axes
    axes = np.array([f.T_cw[:3, 2] for f in frames])       # camera +Z (forward) in world
    i0 = int(np.argmax(axes @ down))
    i1 = (i0 + len(frames) // 4) % len(frames)
    return [frames[i].rgb() for i in ([i0, i1][:n])]
