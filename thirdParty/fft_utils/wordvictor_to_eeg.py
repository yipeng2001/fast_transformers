import numpy as np
import matplotlib.pyplot as plt
import torch
import json
# 生成代表性词汇和词向量（参考之前的生成方法）
eeg_words = [
    "alpha", "beta", "gamma", "delta", "theta", "mu", "kappa", "lambda",
    "brainwave", "neurofeedback", "EEG", "electrode", "signal", "frequency",
    "amplitude", "oscillation", "cortex", "neuron", "synapse", "spike",
    "artifact", "noise", "band", "spectral", "coherence", "power", "density",
    "activity", "baseline", "epoch", "event", "related", "potential", "ERP",
    "synchronization", "desynchronization", "theta-alpha", "beta-gamma",
    "connectivity", "network", "mapping", "topography", "analysis", "feature",
    "extraction", "classification", "detection", "monitoring", "signal", "processing",
    "filtering", "transform", "Fourier", "wavelet", "decomposition", "component",
    "ICA", "PCA", "time", "domain", "frequency", "domain", "spatial", "domain",
    "complexity", "entropy", "fractal", "dimensionality", "phase", "locking",
    "coupling", "correlation", "spectrum", "coherence", "cross-frequency",
    "modulation", "binaural", "beats", "neuroplasticity", "cognitive", "state",
    "attention", "meditation", "sleep", "wakefulness", "arousal", "relaxation",
    "stress", "anxiety", "depression", "therapy", "biofeedback", "neurotherapy",
    "brain-computer", "interface", "BCI", "neurotechnology", "mindfulness", "consciousness"
]

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
    
    
    
if __name__ == "__main__":
    json_path = '/mnt/data/eeg_words.json'  # 确保路径正确
    words = load_eeg_words(json_path)
    print("Loaded words:", words)    
    
    
eeg_words = eeg_words[:100]

embedding_dim = 128
word_vectors = np.random.randn(100, embedding_dim)

high_energy_indices = np.random.choice(100, 30, replace=False)
low_energy_indices = np.random.choice(np.setdiff1d(np.arange(100), high_energy_indices), 30, replace=False)
word_vectors[high_energy_indices] *= 10
word_vectors[low_energy_indices] *= 0.1

# 创建词到向量的映射
word_to_vector = {word: word_vectors[i] for i, word in enumerate(eeg_words)}

# 将词向量转换为脑电波信号
def word_to_eeg_signal(word, word_to_vector):
    vector = word_to_vector.get(word)
    if vector is None:
        return None
    # 模拟脑电波信号，将词向量视为时间序列
    time_series = vector
    # 对时间序列进行快速傅里叶变换（FFT）
    freq_domain = torch.fft.fft(torch.tensor(time_series), dim=0)
    return freq_domain

# 可视化脑电波信号
def plot_eeg_signal(word, signal):
    plt.figure(figsize=(10, 5))
    plt.plot(signal.real, label='Real part')
    plt.plot(signal.imag, label='Imaginary part')
    plt.title(f"EEG Signal for '{word}'")
    plt.xlabel("Frequency bin")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

 
