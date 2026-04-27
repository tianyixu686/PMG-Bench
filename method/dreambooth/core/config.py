from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_json: str
    pretrained_model_name_or_path: str
    output_dir: str

    target_token: str = "[V]"
    instance_token: str = "sks"
    initializer_token: str = "photo"

    resolution: int = 512
    train_batch_size: int = 1
    max_train_steps: int = 800

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    rank: int = 4
    lora_dropout: float = 0.0

    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # no|fp16|bf16
    seed: int = 42
    dataloader_num_workers: int = 0

    enable_xformers_memory_efficient_attention: bool = False
    gradient_checkpointing: bool = False

    train_text_encoder_lora: bool = False
    train_instance_token_embedding: bool = True

    checkpointing_steps: int = 0
    log_interval: int = 50


@dataclass
class InferConfig:
    test_json: str
    sd15_path: str
    lora_root: str
    output_dir: str

    num_images_per_sample: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    seed: int = 42
    negative_prompt: str = "lowres, text, error, cropped, worst quality, low quality"

    instance_token: str = "sks"
    initializer_token: str = "photo"

    no_style_mask: bool = False
    style_mask_terms: str = ""
    user_ids: str = ""


@dataclass
class EvalConfig:
    test_json: str
    infer_output_dir: str
    output_json: str

    image_size: int = 512
    num_history: int = 10
    user_ids: str = ""
