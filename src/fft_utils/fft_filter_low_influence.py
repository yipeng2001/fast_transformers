import torch
import torch.fft

# Based on the spectral theorem, filter out vectors with low influence.
#  Perform time-frequency transformation on word vectors and remove vectors with low influence.
def filter_low_influence_components(embedding_matrix, threshold=0.1):
    """
    对词向量进行时频变换，去掉影响力低的向量。
    Based on the spectral theorem, filter out vectors with low influence.
    Perform time-frequency transformation on word vectors and remove vectors with low influence.
    参数:
    embedding_matrix (torch.Tensor): 词向量矩阵，形状为 (batch_size, seq_length, embedding_dim)
    threshold (float): 影响力阈值，低于该值的频域分量将被去掉
    
    返回:
    torch.Tensor: 简化后的词向量矩阵
    """
    # 对词向量矩阵进行快速傅里叶变换
    freq_domain = torch.fft.fft(embedding_matrix, dim=-1)
    
    # 计算频域分量的幅值
    magnitude = torch.abs(freq_domain)
    
    # 创建一个掩码，将低于阈值的频域分量置零
    mask = magnitude > threshold
    filtered_freq_domain = freq_domain * mask
    
    # 进行逆快速傅里叶变换，得到简化后的词向量
    filtered_embedding_matrix = torch.fft.ifft(filtered_freq_domain, dim=-1).real
    
    return filtered_embedding_matrix

# 示例词向量矩阵
batch_size = 2
seq_length = 5
embedding_dim = 4
embedding_matrix = torch.randn(batch_size, seq_length, embedding_dim)

# 过滤低影响力的向量
threshold = 0.1
filtered_embedding_matrix = filter_low_influence_components(embedding_matrix, threshold)

print("原始词向量矩阵:\n", embedding_matrix)
print("简化后的词向量矩阵:\n", filtered_embedding_matrix)
