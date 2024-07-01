 gpu 太贵了， 所以我想提高transformers 架构的执行效率； 从谱定理，对角矩阵，快速傅里叶变换，拉普拉斯变换，获得 了灵感，过滤掉能量小的词向量， 基于transformer架构开始编写  快速 transformer 架构。  
 
 我实在买不起gpu, 因此更想用便宜的cpu， 想让cpu也能快速实现gpu的效果。


 GPUs are too expensive, so I want to improve the execution efficiency of the transformer architecture. Inspired by the spectral theorem, diagonal matrix, fast Fourier transform, and Laplace transform, I filter out word vectors with low energy and start writing a fast transformer architecture based on the transformer architecture.

I really can't afford a GPU, so I prefer to use a cheaper CPU. I want the CPU to achieve the same effect as a GPU quickly.