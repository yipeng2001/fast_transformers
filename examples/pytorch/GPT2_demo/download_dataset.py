from datasets import load_dataset

# 加载OpenWebText数据集
dataset = load_dataset("openwebtext")

# 选择训练和验证集
train_dataset = dataset["train"]

# 打印一些数据集信息
print(train_dataset)
