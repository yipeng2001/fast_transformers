from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 定义模型名称 
# # 定义模型名称，可以是 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model_name = "gpt2"

# 下载并保存分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("/Volumes/KINGSTON/gtp2data")

# 下载并保存模型
model = GPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained("/Volumes/KINGSTON/gtp2data")