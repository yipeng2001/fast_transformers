import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Paths to the tokenizer and model files
tokenizer_path = '/Volumes/KINGSTON/gtp2data/tokenizer'  # Directory containing merges.txt, tokenizer_config.json, vocab.json, and config.json
model_path = '/Volumes/BOOTCAMP/workgtp/deepblue/08GPT/08_GPT/99_TrainedModel/WikiGPT.pth'  # Path to your trained model file

# Load the tokenizer files directly
tokenizer = GPT2Tokenizer(
    vocab_file=f"{tokenizer_path}/vocab.json",
    merges_file=f"{tokenizer_path}/merges.txt",
    tokenizer_file=f"{tokenizer_path}/tokenizer_config.json"
)

# Set the pad token id to eos token id
tokenizer.pad_token = tokenizer.eos_token

# Load the configuration file directly
config = GPT2Config.from_json_file(f"{tokenizer_path}/config.json")

# Initialize the model architecture with the configuration
model = GPT2LMHeadModel(config)

# Load the trained weights
trained_state_dict = torch.load(model_path)

# Load the trained weights with strict=False
model.load_state_dict(torch.load(model_path), strict=False)

# Get the model's current state dictionary
model_state_dict = model.state_dict()

# Update the model's state dictionary to ensure all necessary keys are present
model_state_dict.update(trained_state_dict)

# Load the updated state dictionary into the model with strict=False
model.load_state_dict(model_state_dict, strict=False)

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    print("Tokenized Input IDs:", inputs.input_ids)
    print("Attention Mask:", inputs.attention_mask)
    
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("Generated Output IDs:", outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__": 
    test_prompt = "The history of artificial intelligence"
    generated_text = generate_text(test_prompt)
    print("Generated Text:")
    print(generated_text)
