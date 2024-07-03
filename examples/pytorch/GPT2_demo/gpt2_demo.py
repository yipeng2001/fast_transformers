from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 设置模型和tokenizer路径
model_path = "/Volumes/KINGSTON/gtp2data/"  # 替换为你的模型文件所在路径

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 加载模型
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

# 生成文本的函数
def generate_text(prompt, max_length=50):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")

    # 使用模型生成输出
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 测试生成文本
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
