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

# Load the configuration file directly
config = GPT2Config.from_json_file(f"{tokenizer_path}/config.json")

# Initialize the model architecture with the configuration
model = GPT2LMHeadModel(config)

# Load the trained weights
trained_state_dict = torch.load(model_path)

# Load the trained weights
# Load the trained weights with strict=False
model.load_state_dict(torch.load(model_path), strict=False)

# 获取模型的当前状态字典
model_state_dict = model.state_dict()
# 更新模型的状态字典，确保所有必要的键都存在
model_state_dict.update(trained_state_dict)
# 使用更新后的状态字典加载模型权重
# model.load_state_dict(model_state_dict)
model.load_state_dict(model_state_dict, strict=False)
# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    test_prompt = "The history of artificial intelligence"
    generated_text = generate_text(test_prompt)
    print("Generated Text:")
    print(generated_text)
