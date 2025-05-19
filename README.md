# 文档检索系统

一个基于Python的本地文档检索系统，能够搜索和检索各种类型的文档内容，包括Word、Excel、PPT、PDF、HTML和文本文件。

## 功能特点

- 支持多种文档格式（PDF、Word、Excel、HTML、文本）
- 提供多种检索算法（TF-IDF、向量空间模型、布尔检索）
- 基于规则的结果过滤和排序
- GPU加速支持，显著提升检索性能
- Web界面和RESTful API
- 数据抓取和资源管理
- 性能评估和可视化

## 系统要求

- Python 3.8+
- 支持GPU加速（可选）：CUDA兼容的NVIDIA GPU

## 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/Document-Retrieval-System.git
cd Document-Retrieval-System
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

对于GPU加速支持，确保安装了正确的CUDA版本和对应的cupy库。

## 配置

主要配置文件位于`config/`目录：

- `app_config.yaml`：应用程序全局配置
- `rules_config.yaml`：检索规则配置

### GPU加速配置

在`app_config.yaml`中可以配置GPU加速选项：

```yaml
# 硬件加速配置
hardware:
  use_gpu: true        # 是否使用GPU加速
  gpu_id: 0            # 使用的GPU ID
  mixed_precision: true # 是否使用混合精度
  batch_size: 32       # 批处理大小
  max_threads: 4       # CPU线程数
```

## 使用方法

### 命令行接口

#### 基本用法

```bash
# 执行查询
python main.py --query "人工智能"

# 指定算法执行查询
python main.py --query "人工智能" --algorithm tfidf

# 重建索引
python main.py --rebuild-index

# 禁用GPU加速
python main.py --query "人工智能" --gpu-disabled
```

#### 性能基准测试

```bash
# 运行性能基准测试
python main.py --benchmark

# 指定算法运行基准测试
python main.py --benchmark --algorithm tfidf
```

### Web服务

```bash
# 启动Web服务
python run.py

# 指定主机和端口
python run.py --host 0.0.0.0 --port 8080

# 启用调试模式
python run.py --debug

# 禁用GPU加速
python run.py --gpu-disabled
```

## GPU加速性能

使用GPU加速可以显著提高检索性能，特别是对于大型文档集合。性能提升取决于多种因素，包括：

- GPU型号和性能
- 文档数量和大小
- 检索算法

使用GPU加速后，检索性能通常可以获得2-10倍的提升，具体取决于上述因素。

### 性能测试

可以使用以下命令运行GPU性能基准测试：

```bash
python tests/gpu_benchmark.py
```

或者：

```bash
python main.py --benchmark
```

测试结果会以图表形式保存在`visualization/output/`目录。

## 项目结构

```
Document-Retrieval-System/
├── config/                 # 配置文件
├── core/                   # 核心功能
│   ├── datasource/         # 数据源和解析器
│   ├── retrieval/          # 检索引擎
│   ├── rules/              # 规则引擎
│   └── utils/              # 工具类
├── data/                   # 数据目录
│   ├── index_db/           # 索引数据库
│   ├── processed/          # 处理后的文档
│   └── raw_documents/      # 原始文档
├── docs/                   # 文档
├── gui/                    # Web界面
├── models/                 # 模型目录
├── tests/                  # 测试目录
└── visualization/          # 可视化工具
```

## 许可证

[MIT](LICENSE)