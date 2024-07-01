import torch
import torch.fft
# Fast Fourier Transform and Self-Attention Mechanism
def fft_attention(query, key, value):
    query_fft = torch.fft.fft(query, dim=-1)
    key_fft = torch.fft.fft(key, dim=-1)
    value_fft = torch.fft.fft(value, dim=-1)
    # Bhid is a tensor dimension naming convention used to represent the arrangement of tensors in specific dimensions. This naming convention is often used when using the einsum function to express the meaning of operations more clearly.
    
#     Specifically:

# b represents the batch size (the number of samples in a batch).
# h represents the number of heads (the number of attention heads).
# i and d are commonly used to represent the sequence length and feature dimension, respectively.
# In the context of self-attention mechanisms, the dimensions of a tensor can be interpreted as follows:

# batch_size (b): The size of the batch of input data.
# num_heads (h): The number of heads in the multi-head attention mechanism.
# seq_length (i): The length of the input sequence.
# head_dim (d): The feature dimension size of each attention head.
# Therefore, bhid typically represents a tensor with a shape of (batch_size, num_heads, seq_length, head_dim).

    attention_scores = torch.einsum("bhid,bhjd->bhij", query_fft, key_fft.transpose(-2, -1).conj())
    attention_probs = torch.fft.ifft(attention_scores, dim=-1).real
    
    context_layer = torch.einsum("bhij,bhjd->bhid", attention_probs, value_fft)
    context_layer = torch.fft.ifft(context_layer, dim=-1).real
    
    return context_layer
