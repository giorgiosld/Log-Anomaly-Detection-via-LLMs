from transformers import AutoModelForCausalLM, AutoTokenizer 

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B", hf_token=None):    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",  
        trust_remote_code=True,
    )


    
    return model, tokenizer
