 gpu 太贵了， 所以我想提高transformers 架构的执行效率； 从谱定理，对角矩阵，快速傅里叶变换，拉普拉斯变换，获得 了灵感，过滤掉能量小的词向量， 基于transformer架构开始编写  快速 transformer 架构。  
 
 我实在买不起gpu, 因此更想用便宜的cpu， 想让cpu也能快速实现gpu的效果。


 GPUs are too expensive, so I want to improve the execution efficiency of the transformer architecture. Inspired by the spectral theorem, diagonal matrix, fast Fourier transform, and Laplace transform, I filter out word vectors with low energy and start writing a fast transformer architecture based on the transformer architecture.

I really can't afford a GPU, so I prefer to use a cheaper CPU. I want the CPU to achieve the same effect as a GPU quickly.



<!-- 编译和启动 -->
要编译和启动从 [https://github.com/yipeng2001/fast_transformers/tree/sparse-attention](https://github.com/yipeng2001/fast_transformers/tree/sparse-attention) 下载的 Transformer 架构源码，请按照以下步骤操作：

### 1. 克隆仓库

首先，将仓库克隆到本地：

```bash
git clone -b sparse-attention https://github.com/yipeng2001/fast_transformers.git
cd fast_transformers
```

### 2. 创建并激活虚拟环境

为避免依赖冲突，建议使用虚拟环境：

```bash
python3 -m venv transformer_env
source transformer_env/bin/activate  # Linux/macOS
transformer_env\Scripts\activate  # Windows
```

### 3. 安装依赖项

虽然仓库中可能没有 `requirements.txt` 文件，但你可以通过查看 `setup.py` 文件来确定依赖项。安装这些依赖项：

```bash
pip install numpy torch
pip install -e .  # 这将会安装包和它的依赖项
```

### 4. 编译 C++/CUDA 代码（如果有）

如果项目包含 C++ 或 CUDA 代码，你需要确保已经安装了相应的编译工具。项目中可能有一个 `setup.py` 或者 `CMakeLists.txt` 文件用于编译这些部分。检查并执行以下命令：

```bash
python setup.py build
python setup.py install
```

如果是基于 CMake 的项目：

```bash
mkdir build
cd build
cmake ..
make
make install
```

### 5. 运行项目

确保一切安装无误后，你可以尝试运行项目中的主脚本或者示例脚本。以下是一个通用的运行方法：

```bash
python examples/run_transformer.py  # 假设有一个示例脚本
```

### 具体步骤示例

1. **克隆仓库**：

    ```bash
    git clone -b sparse-attention https://github.com/yipeng2001/fast_transformers.git
    cd fast_transformers
    ```

2. **创建并激活虚拟环境**：

    ```bash
    python3 -m venv transformer_env
    source transformer_env/bin/activate  # Linux/macOS
    transformer_env\Scripts\activate  # Windows
    ```

3. **安装依赖项**：

    ```bash
    pip install numpy torch
    pip install -e .
    ```

4. **编译源码**（如果有需要编译的部分）：

    ```bash
    python setup.py build
    python setup.py install
    ```

5. **运行项目**：

    ```bash
    python examples/run_transformer.py  # 假设有一个示例脚本
    ```

### 额外说明

- **查看 README.md 和文档**：在仓库根目录中通常会有 README.md 文件，其中可能包含项目的具体安装和运行指南。
- **检查 `setup.py` 文件**：该文件中通常包含包的依赖项，可以根据其中的信息手动安装。

如果在执行过程中遇到任何问题，请提供详细的错误信息，以便进一步协助解决。

