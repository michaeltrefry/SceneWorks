"""Image captioning adapters for training datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .image_adapters import require_inference_backend_for_gpu_worker, select_torch_device, select_torch_dtype


JOY_CAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JOY_NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."
JOY_CAPTION_RESAMPLE = Image.Resampling.BICUBIC

JOY_CAPTION_TYPE_MAP: dict[str, tuple[str, str, str]] = {
    "Descriptive": (
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ),
    "Descriptive (Casual)": (
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ),
    "Straightforward": (
        'Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
        'Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
        'Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
    ),
    "Stable Diffusion Prompt": (
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ),
    "MidJourney": (
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ),
    "Danbooru tag list": (
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ),
    "e621 tag list": (
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
    ),
    "Rule34 tag list": (
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
    ),
    "Booru-like tag list": (
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ),
    "Art Critic": (
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ),
    "Product Listing": (
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ),
    "Social Media Post": (
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ),
}


@dataclass
class JoyCaptionOptions:
    caption_type: str = "Descriptive"
    caption_length: str = "long"
    extra_options: list[str] = field(default_factory=list)
    name_input: str = ""
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    caption_prompt: str = ""
    low_vram: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "JoyCaptionOptions":
        payload = payload or {}
        return cls(
            caption_type=str(payload.get("captionType") or "Descriptive"),
            caption_length=str(payload.get("captionLength") or "long"),
            extra_options=[str(option) for option in payload.get("extraOptions") or []],
            name_input=str(payload.get("nameInput") or ""),
            temperature=float(payload.get("temperature", 0.6)),
            top_p=float(payload.get("topP", 0.9)),
            max_new_tokens=int(payload.get("maxNewTokens", 256)),
            caption_prompt=str(payload.get("captionPrompt") or ""),
            low_vram=bool(payload.get("lowVram", False)),
        )


def build_joy_caption_prompt(options: JoyCaptionOptions) -> str:
    caption_length = options.caption_length
    map_index = 2
    if caption_length == "any":
        map_index = 0
    elif caption_length.isdigit():
        map_index = 1
    templates = JOY_CAPTION_TYPE_MAP.get(options.caption_type, JOY_CAPTION_TYPE_MAP["Descriptive"])
    prompt = templates[map_index]
    if options.extra_options:
        prompt = f"{prompt} {' '.join(options.extra_options)}"
    return (
        prompt.replace("{name}", options.name_input or "{NAME}")
        .replace("{length}", caption_length)
        .replace("{word_count}", caption_length)
    )


def caption_with_trigger_words(caption: str, trigger_words: list[str]) -> str:
    cleaned = " ".join(str(caption or "").split()).strip()
    lower_caption = cleaned.lower()
    missing = [word for word in trigger_words if word and word.lower() not in lower_caption]
    return ", ".join([*missing, cleaned]).strip(", ")


def normalize_processor_resample(processor: Any) -> None:
    """Avoid torch resize modes such as Lanczos that Transformers may load from config."""
    targets = [processor, getattr(processor, "image_processor", None)]
    for target in targets:
        if target is None:
            continue
        if hasattr(target, "resample"):
            setattr(target, "resample", JOY_CAPTION_RESAMPLE)


class JoyCaptioner:
    def __init__(self, *, model_name_or_path: str, options: JoyCaptionOptions, gpu_id: str) -> None:
        self.model_name_or_path = model_name_or_path or JOY_CAPTION_MODEL
        self.options = options
        self.gpu_id = gpu_id
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self.torch_dtype = None

    def loaded_models(self) -> list[str]:
        return [self.model_name_or_path] if self.model is not None else []

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        require_inference_backend_for_gpu_worker(torch, self.gpu_id)
        self.torch = torch
        self.device = select_torch_device(torch, self.gpu_id)
        self.torch_dtype = select_torch_dtype(torch, self.device, None)
        # JoyCaption's tokenizer expects the fast processor path; the slow
        # conversion can route tokenizer config booleans into text encoders.
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, use_fast=True)
        normalize_processor_resample(self.processor)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="cpu" if self.options.low_vram else "auto",
        )
        if self.options.low_vram:
            self.model.to(self.device)
        self.model.eval()

    def caption_image(self, image_path: str) -> str:
        if self.model is None or self.processor is None or self.tokenizer is None:
            self.load()
        prompt = self.options.caption_prompt.strip() or build_joy_caption_prompt(self.options)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[convo_string], images=[rgb_image], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        generate_kwargs = {
            "max_new_tokens": self.options.max_new_tokens,
            "do_sample": True,
            "temperature": self.options.temperature,
            "top_p": self.options.top_p,
            "use_cache": True,
            "suppress_tokens": None,
        }
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)[0]
        generated = generated[inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


def run_training_caption_job(
    *,
    api: Any,
    settings: Any,
    job: dict[str, Any],
    progress: Callable[[str, str, float, str], None],
    cancel_requested: Callable[[], bool],
) -> dict[str, Any]:
    payload = job.get("payload") or {}
    if payload.get("captioner") != "joy_caption":
        raise ValueError(f"Unsupported training captioner: {payload.get('captioner')!r}")
    items = payload.get("items") or []
    if not items:
        raise ValueError("Training caption job has no items to caption.")
    options = JoyCaptionOptions.from_payload(payload.get("options"))
    captioner = JoyCaptioner(
        model_name_or_path=str(payload.get("modelNameOrPath") or JOY_CAPTION_MODEL),
        options=options,
        gpu_id=getattr(settings, "gpu_id", "auto"),
    )
    progress("loading_model", "loading_model", 0.08, "Loading Joy Caption model.")
    captioner.load()
    captions = []
    total = len(items)
    for index, item in enumerate(items):
        if cancel_requested():
            raise InterruptedError("Training captioning canceled by user.")
        image_path = str(item.get("imagePath") or "")
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Training caption image not found: {image_path}")
        progress(
            "running",
            "running",
            0.12 + (0.76 * index / max(total, 1)),
            f"Captioning image {index + 1} of {total}.",
        )
        trigger_words = [str(word).strip() for word in item.get("triggerWords") or [] if str(word).strip()]
        caption = caption_with_trigger_words(captioner.caption_image(image_path), trigger_words)
        captions.append(
            {
                "itemId": item.get("itemId"),
                "caption": {
                    "text": caption,
                    "source": "auto",
                    "triggerWords": trigger_words,
                },
            }
        )
    progress("saving", "saving", 0.94, "Saving generated captions.")
    project_id = payload.get("projectId")
    dataset_id = payload.get("datasetId")
    result = api.post(
        f"/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars",
        {"items": captions},
    )
    return {
        "captioner": "joy_caption",
        "modelNameOrPath": captioner.model_name_or_path,
        "datasetId": dataset_id,
        "datasetVersion": result.get("dataset", {}).get("version"),
        "captionedItemCount": len(captions),
        "sidecars": result.get("sidecars", []),
    }
