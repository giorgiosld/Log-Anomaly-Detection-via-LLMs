from transformers import LlamaForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

from accelerate import Accelerator

def get_model():
    
    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
   
    accelerate = Accelerator(mixed_precision="fp16")
    
    device_index=accelerate.process_index

    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B',
            token="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",
            quantization_config=quantization_config,
            device_map = {"": device_index},
            # device_map={"": torch.cuda.current_device()}, 
            # device_map='auto',
            torch_dtype=torch.float16,
    )

    # Prepare for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    return model, accelerate



