# 生成测试用原始语料库，里面有能量高和能量低的词向量

import numpy as np

# 生成代表性词汇
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

# 确保只有100个词
eeg_words = eeg_words[:100]

# 生成词向量（128维度）
embedding_dim = 128
word_vectors = np.random.randn(100, embedding_dim)

# 调整能量（人为设置一部分词向量的能量高，一部分低）
high_energy_indices = np.random.choice(100, 30, replace=False)
low_energy_indices = np.random.choice(np.setdiff1d(np.arange(100), high_energy_indices), 30, replace=False)

# 设置高能量词向量
word_vectors[high_energy_indices] *= 10

# 设置低能量词向量
word_vectors[low_energy_indices] *= 0.1

# 打印一些词和对应的向量能量
for i in range(10):
    word = eeg_words[i]
    vector = word_vectors[i]
    energy = np.linalg.norm(vector)
    print(f"Word: {word}, Energy: {energy}")

# 打印高能量和低能量词的示例
print("\nHigh energy words and vectors:")
for idx in high_energy_indices[:5]:
    print(f"Word: {eeg_words[idx]}, Vector Energy: {np.linalg.norm(word_vectors[idx])}")

print("\nLow energy words and vectors:")
for idx in low_energy_indices[:5]:
    print(f"Word: {eeg_words[idx]}, Vector Energy: {np.linalg.norm(word_vectors[idx])}")
