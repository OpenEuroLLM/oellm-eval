"""LMMS_MODEL_ADAPTERS must name adapters the pinned lmms-eval registers."""

import pytest

from oellm.constants import LMMS_MODEL_ADAPTERS, detect_lmms_model_type

# Registry of lmms-eval 45c766f60b6f8c153e4c72d06ca636e2db0ebcdb, the commit
# docs/VENV.md installs. Regenerate from that venv with:
#   python -c "import json, lmms_eval.models as m; print(json.dumps(sorted(
#       set(m.MODEL_REGISTRY_V2.list_model_names()) | set(m.AVAILABLE_SIMPLE_MODELS))))"
PINNED_LMMS_EVAL_ADAPTERS = frozenset(
    [
        "aero",
        "aria",
        "async_hf",
        "async_hf_model",
        "async_openai",
        "async_openai_compatible",
        "async_openai_compatible_chat",
        "audio_flamingo_3",
        "auroracap",
        "bagel",
        "bagel_lmms_engine",
        "bagel_umm",
        "bagel_unig2u",
        "baichuan_omni",
        "batch_gpt4",
        "cambrians",
        "cambrians_vsc",
        "cambrians_vsc_streaming",
        "cambrians_vsr",
        "claude",
        "cogvlm2",
        "dummy",
        "dummy_video_reader",
        "egogpt",
        "fastvideo",
        "from_log",
        "fuyu",
        "gemini_api",
        "gemma3",
        "glm4v",
        "gpt4o_audio",
        "gpt4v",
        "huggingface",
        "idefics2",
        "illume_plus",
        "instructblip",
        "internvideo2",
        "internvideo2_5",
        "internvl",
        "internvl2",
        "internvl3",
        "internvl3_5",
        "internvl_hf",
        "kimi_audio",
        "litellm",
        "litellm_chat",
        "litellm_compatible",
        "llama4_scout",
        "llama_vid",
        "llama_vision",
        "llava",
        "llava_hf",
        "llava_onevision",
        "llava_onevision1_5",
        "llava_onevision2",
        "llava_onevision_moviechat",
        "llava_sglang",
        "llava_vid",
        "longva",
        "longvila",
        "mantis",
        "minicpm_o",
        "minicpm_v",
        "minimonkey",
        "mmada",
        "moviechat",
        "mplug_owl_video",
        "nanovlm",
        "ola",
        "omnivinci",
        "openai",
        "openai_compatible",
        "openai_compatible_chat",
        "oryx",
        "ovis_u1",
        "penguinvl",
        "phi3v",
        "phi4_multimodal",
        "plm",
        "qwen2_5_omni",
        "qwen2_5_vl",
        "qwen2_audio",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_omni",
        "qwen3_vl",
        "qwen_image_edit",
        "qwen_vl",
        "qwen_vl_api",
        "reka",
        "ross",
        "sam3",
        "sglang",
        "slime",
        "srt_api",
        "thyme",
        "tinyllava",
        "uni_moe_2_omni",
        "videoChatGPT",
        "video_llava",
        "video_salmonn_2",
        "videochat2",
        "videochat_flash",
        "videollama3",
        "vila",
        "vita",
        "vllm",
        "vllm_generate",
        "vora",
        "whisper",
        "whisper_tt",
        "whisper_vllm",
        "xcomposer2_4KHD",
        "xcomposer2d5",
    ]
)


def test_every_mapped_adapter_exists_in_the_pinned_registry():
    missing = sorted(
        {adapter for _, adapter in LMMS_MODEL_ADAPTERS} - PINNED_LMMS_EVAL_ADAPTERS
    )
    assert not missing, (
        f"LMMS_MODEL_ADAPTERS routes to adapters the pinned lmms-eval does not "
        f"register: {missing}"
    )


@pytest.mark.parametrize(
    "model, adapter",
    [
        ("HuggingFaceM4/idefics2-8b", "idefics2"),
        ("llava-hf/llava-interleave-qwen-0.5b-hf", "llava_hf"),
        ("Qwen/Qwen2.5-Omni-7B", "qwen2_5_omni"),
        ("openai/whisper-large-v3", "whisper"),
        ("moonshotai/Kimi-Audio-7B-Instruct", "kimi_audio"),
    ],
)
def test_model_families_route_to_registered_adapters(model, adapter):
    assert detect_lmms_model_type(model) == adapter


@pytest.mark.parametrize(
    "model", ["HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceM4/Idefics3-8B-Llama3"]
)
def test_idefics3_family_is_refused_at_schedule_time(model):
    """No adapter in the pinned lmms-eval can run Idefics3 checkpoints with
    transformers<4.50; refusing here beats a headless model on the node."""
    with pytest.raises(ValueError):
        detect_lmms_model_type(model)
