import numpy as np

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
eeg_words = eeg_words[:100]

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
