from transformers import LlamaForCausalLM

def get_model():
    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B',
            token="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")
    return model

