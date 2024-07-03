要运行这个示例代码，请按照以下步骤操作：

1. **准备环境**：
   - 确保你已经安装了必要的库：`transformers` 和 `safetensors`。
   - 将你提供的模型文件放在同一个目录下。

2. **编写脚本**：
   - 创建一个新的Python脚本文件，例如`gpt2_demo.py`。
   - 将以下代码复制粘贴到这个文件中，并保存。

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 设置模型和tokenizer路径
model_path = "./"  # 替换为你的模型文件所在路径

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
```

3. **运行脚本**：
   - 打开终端或命令提示符，导航到脚本所在的目录。
   - 运行以下命令来执行脚本：

```bash
python gpt2_demo.py
```

如果一切设置正确，你应该会看到脚本输出生成的文本。

### 详细步骤

1. **安装依赖**：
   打开终端或命令提示符并运行：
   ```bash
   pip install transformers safetensors
   ```

2. **放置模型文件**：
   将你之前展示的模型文件放在与`gpt2_demo.py`同一个目录下。这些文件包括：
   - `config.json`
   - `generation_config.json`
   - `merges.txt`
   - `model.safetensors`
   - `special_tokens_map.json`
   - `tokenizer_config.json`
   - `vocab.json`

3. **运行脚本**：
   在终端中导航到脚本所在的目录，然后运行：
   ```bash
   python gpt2_demo.py
   ```

这将启动脚本并生成文本输出。如果遇到任何问题，请检查模型文件路径是否正确，以及所需的库是否已正确安装。