import numpy as np
import json
# 生成代表性词汇和词向量（参考之前的生成方法）
def load_eeg_words(json_path):
        """
    从指定的JSON文件中加载词数组。

    参数:
    json_path (str): JSON文件的路径

    返回:
    list: 词数组
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data['eeg_words']
    
json_path = '../eeg_words.json'  # 确保路径正确
eeg_words = load_eeg_words(json_path)
print("Loaded words:", eeg_words)    
 

embedding_dim = 128
word_vectors = np.random.randn(100, embedding_dim)

high_energy_indices = np.random.choice(100, 30, replace=False)
low_energy_indices = np.random.choice(np.setdiff1d(np.arange(100), high_energy_indices), 30, replace=False)
word_vectors[high_energy_indices] *= 10
word_vectors[low_energy_indices] *= 0.1

# 创建词到向量的映射
word_to_vector = {word: word_vectors[i] for i, word in enumerate(eeg_words)}

# 设定能量阈值，超过该值的词被认为是重要的
energy_threshold = 5.0

def is_important(word, word_to_vector, threshold):
    vector = word_to_vector.get(word)
    if vector is None:
        return False
    energy = np.linalg.norm(vector)
    return energy > threshold

def classify_sentence(sentence, word_to_vector, threshold):
    words = sentence.split()
    important_words = []
    unimportant_words = []
    
    for word in words:
        if is_important(word, word_to_vector, threshold):
            important_words.append(word)
        else:
            unimportant_words.append(word)
    
    return important_words, unimportant_words

# 示例句子
sentence = "EEG signals show synchronization and desynchronization during different brain states"

important_words, unimportant_words = classify_sentence(sentence, word_to_vector, energy_threshold)

print("重要的词:", important_words)
print("不重要的词:", unimportant_words)
