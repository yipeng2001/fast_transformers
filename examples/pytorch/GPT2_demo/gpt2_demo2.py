from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 设置模型和tokenizer路径
model_path = "/Volumes/KINGSTON/gtp2data/"  # 替换为你的模型文件所在路径

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 加载模型
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

# 生成文本的函数
def ask_question(question, max_length=100):
    # 编码输入
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # 使用模型生成输出
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_beams=5,  # 使用beam search以提高生成质量
        pad_token_id=tokenizer.eos_token_id  # 设置pad_token_id为eos_token_id以避免警告
    )

    # 解码输出
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 测试提问并生成回答
question = "夜里睡不着咋办?"
answer = ask_question(question)
print("Q:", question)
print("A:", answer)
