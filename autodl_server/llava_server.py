#!/usr/bin/env python3
"""HTTP service that locates a drone with LLaVA on an AutoDL server."""

import io
import json
import math
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH", "")
ADAPTER_PATH = os.environ.get("LLAVA_ADAPTER_PATH", "")
MOCK_MODE = os.environ.get("LLAVA_MOCK", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRUST_REMOTE_CODE = os.environ.get("LLAVA_TRUST_REMOTE_CODE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_NEW_TOKENS = int(os.environ.get("LLAVA_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("LLAVA_TEMPERATURE", "0.95"))
TOP_P = float(os.environ.get("LLAVA_TOP_P", "0.7"))
REQUIRE_CUDA = os.environ.get("LLAVA_REQUIRE_CUDA", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

BBOX_PROMPT = """Locate the quadcopter drone in this image.
Return exactly one JSON object and no other text:
{"found":true,"bbox":[x1,y1,x2,y2]}
The bounding-box coordinates must be decimal values from 0.0 to 1.0 relative
to the full image. The top-left corner is (0.0, 0.0) and the bottom-right is
(1.0, 1.0).
If no quadcopter drone is visible, return:
{"found":false,"bbox":null}
"""

CENTER_PROMPT = """You are a counter-drone system operator. When a drone
appears in an image, you need to analyze the surrounding environment and select
the most appropriate strike plan. Available strike methods include: radio
jamming, laser strikes, and high-energy microwave strikes. Please design a
suitable strike plan."""


class LlavaBackend:
    def __init__(self, model_path, adapter_path="", trust_remote_code=False):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoLlavaModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoLlavaModel

        self.torch = torch
        if REQUIRE_CUDA and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for LLaVA inference. Start the AutoDL "
                "instance in GPU mode, or explicitly set LLAVA_REQUIRE_CUDA=false."
            )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoLlavaModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.prompt = CENTER_PROMPT
        else:
            self.prompt = BBOX_PROMPT
        self.model.eval()

    def locate(
        self,
        image,
        prompt=None,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        prompt = prompt or self.prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {
            name: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for name, value in inputs.items()
        }

        with self.torch.inference_mode():
            generation_options = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_options.update(
                    temperature=temperature,
                    top_p=top_p,
                )
            output = self.model.generate(**inputs, **generation_options)

        input_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()


def validated_pixel_center(center, width, height):
    """Return a validated pixel center, accepting 0..1 normalized values."""
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError(f"invalid center: {center!r}")

    x, y = [float(value) for value in center]
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= width
        y *= height
    if not 0.0 <= x < width or not 0.0 <= y < height:
        raise ValueError(
            f"center {center!r} is outside {width}x{height} image"
        )
    return [x, y]


def parse_model_answer(answer, width, height):
    """Parse bbox JSON or the center-coordinate format used for LoRA training."""
    match = re.search(r"\{.*?\}", answer, flags=re.DOTALL)
    if match:
        result = json.loads(match.group(0))
        if result.get("found") is False:
            return {"found": False, "bbox": None, "center": None}
        if result.get("center") is not None:
            center = validated_pixel_center(result["center"], width, height)
            return {"found": True, "bbox": None, "center": center}

        bbox = result.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid bbox: {bbox!r}")

        x1, y1, x2, y2 = [float(value) for value in bbox]
        coordinates = (x1, y1, x2, y2)
        if not all(0.0 <= value <= 1000.0 for value in coordinates):
            raise ValueError(
                f"bbox coordinates must be within 0..1000: {bbox!r}"
            )
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox must have positive area: {bbox!r}")

        coordinate_scale = 1.0 if max(coordinates) <= 1.0 else 1000.0
        pixel_bbox = [
            x1 * width / coordinate_scale,
            y1 * height / coordinate_scale,
            x2 * width / coordinate_scale,
            y2 * height / coordinate_scale,
        ]
        center = [
            (pixel_bbox[0] + pixel_bbox[2]) / 2.0,
            (pixel_bbox[1] + pixel_bbox[3]) / 2.0,
        ]
        return {"found": True, "bbox": pixel_bbox, "center": center}

    center_match = re.search(
        r"\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]",
        answer,
    )
    if center_match:
        center = validated_pixel_center(
            [center_match.group(1), center_match.group(2)],
            width,
            height,
        )
        return {"found": True, "bbox": None, "center": center}

    raise ValueError(f"model did not return a supported coordinate: {answer!r}")


def validated_generation_options(
    prompt,
    do_sample,
    temperature,
    top_p,
    max_new_tokens,
):
    """Validate request-controlled generation settings."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 8192:
        raise ValueError("prompt must contain at most 8192 characters")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be within (0, 1]")
    if not 1 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be within 1..1024")
    return {
        "prompt": prompt,
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
    }


@asynccontextmanager
async def lifespan(app):
    app.state.backend = None
    if not MOCK_MODE:
        if not MODEL_PATH:
            raise RuntimeError(
                "Set LLAVA_MODEL_PATH or start with LLAVA_MOCK=true"
            )
        app.state.backend = LlavaBackend(
            MODEL_PATH,
            ADAPTER_PATH,
            TRUST_REMOTE_CODE,
        )
    yield


app = FastAPI(title="Drone LLaVA Locator", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "mock" if MOCK_MODE else "llava",
        "model_path": MODEL_PATH or None,
        "adapter_path": ADAPTER_PATH or None,
        "generation": {
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    }


@app.post("/locate")
async def locate(
    image: UploadFile = File(...),
    request_id: int = Form(0),
    prompt: str = Form(""),
    do_sample: bool = Form(True),
    temperature: float = Form(TEMPERATURE),
    top_p: float = Form(TOP_P),
    max_new_tokens: int = Form(MAX_NEW_TOKENS),
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    default_prompt = (
        app.state.backend.prompt if app.state.backend is not None else CENTER_PROMPT
    )
    try:
        generation = validated_generation_options(
            prompt or default_prompt,
            do_sample,
            temperature,
            top_p,
            max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    width, height = pil_image.size
    if MOCK_MODE:
        box_width = width * 0.2
        box_height = height * 0.2
        result = {
            "found": True,
            "bbox": [
                width / 2.0 - box_width / 2.0,
                height / 2.0 - box_height / 2.0,
                width / 2.0 + box_width / 2.0,
                height / 2.0 + box_height / 2.0,
            ],
            "center": [width / 2.0, height / 2.0],
            "raw_answer": None,
        }
    else:
        raw_answer = app.state.backend.locate(pil_image, **generation)
        try:
            result = parse_model_answer(raw_answer, width, height)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": str(exc), "raw_answer": raw_answer},
            ) from exc
        result["raw_answer"] = raw_answer

    result.update(
        {
            "request_id": request_id,
            "image_width": width,
            "image_height": height,
            "generation": {
                name: value
                for name, value in generation.items()
                if name != "prompt"
            },
        }
    )
    return result
