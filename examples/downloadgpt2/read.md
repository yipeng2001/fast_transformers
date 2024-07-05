是的，你可以自己下载 GPT-2 模型，然后将其存储在指定的位置，并让程序从指定的位置读取模型和分词器。以下是详细步骤：

### 下载 GPT-2 模型和分词器

首先，你需要从 Hugging Face 下载 GPT-2 模型和分词器的权重和配置文件。你可以使用以下命令来下载并保存这些文件：

```bash
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 定义模型名称
model_name = "gpt2"

# 下载并保存分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("/path/to/save/directory")

# 下载并保存模型
model = GPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained("/path/to/save/directory")
```

将 `/path/to/save/directory` 替换为你希望保存模型和分词器的目录。

### 从指定位置加载 GPT-2 模型和分词器

下载并保存模型和分词器后，你可以从指定的位置加载它们：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 定义保存模型和分词器的目录
model_directory = "/path/to/save/directory"

# 从指定位置加载分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

# 从指定位置加载模型
model = GPT2LMHeadModel.from_pretrained(model_directory)

# 输入提示文本
input_text = "Once upon a time"

# 将输入文本编码为模型输入所需的格式
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 使用模型生成文本
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的文本
print(generated_text)
```

将 `/path/to/save/directory` 替换为你保存模型和分词器的实际目录路径。

### 步骤总结

1. **下载并保存模型和分词器**：

    ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 定义模型名称
    model_name = "gpt2"

    # 下载并保存分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("/path/to/save/directory")

    # 下载并保存模型
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained("/path/to/save/directory")
    ```

2. **从指定位置加载模型和分词器**：

    ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 定义保存模型和分词器的目录
    model_directory = "/path/to/save/directory"

    # 从指定位置加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

    # 从指定位置加载模型
    model = GPT2LMHeadModel.from_pretrained(model_directory)

    # 输入提示文本
    input_text = "Once upon a time"

    # 将输入文本编码为模型输入所需的格式
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 使用模型生成文本
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 打印生成的文本
    print(generated_text)
    ```
 

GPT-2 (124M parameters)：

大约 500MB
GPT-2 Medium (355M parameters)：

大约 1.5GB