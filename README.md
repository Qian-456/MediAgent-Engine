# MediAgent-Engine

MediAgent-Engine 是一个基于 Qwen 系列模型的专业医疗问答推理与微调引擎。本项目旨在利用高质量的医疗对话数据（如 cMedQA2），通过 LoRA (Low-Rank Adaptation) 技术微调大模型，使其具备专业的医疗咨询能力，并使用 vLLM 实现高并发、低延迟的推理服务。

## 📚 数据集介绍

本项目主要支持 **cMedQA2** 中文医疗问答数据集。

*   **定位**: 真实世界的中文“患者提问 + 医生回答”对。
*   **规模**: 包含约 108,000 个问题和 200,000+ 个答案（原始数据规模更大，可扩展至百万级）。
*   **特点**: 
    *   **真实性**: 数据来源于在线医疗咨询平台，已进行脱敏处理。
    *   **多样性**: 覆盖内科、外科、儿科等多个科室的常见病症。
    *   **口语化**: 问题描述贴近日常口语，非常适合训练面向患者的对话机器人。

## 🏗️ 代码结构

```text
MediAgent-Engine/
├── data/                   # 数据存储
│   ├── raw/                # 原始 CSV 数据 (question.csv, answer.csv)
│   └── processed/          # 处理后的 JSONL 训练数据
├── models/                 # 模型仓库
│   ├── base/               # 基础模型 (Qwen/Qwen3-8B)
│   └── checkpoints/        # 微调后的 LoRA 适配器权重
├── scripts/                # 任务脚本
│   ├── data_processing/    # 数据处理脚本
│   └── training/           # 训练/微调脚本
├── src/                    # 核心源代码
│   ├── inference/          # 推理引擎代码 (vLLM)
│   └── tools/              # 工具脚本 (模型合并)
├── tests/                  # 测试用例
└── README.md               # 项目文档
```

## 🚀 安装指南

1.  **环境要求**:
    *   **推荐平台**: 阿里云 PAI-DSW
    *   **镜像**: `modelscope:1.34.0-pytorch2.9.1-gpu-py311-cu124-ubuntu22.04`
    *   **实例规格**: `ecs.gn7i-c8g1.2xlarge` (8 vCPU, 30 GiB, NVIDIA A10 * 1)
    *   **Python**: 3.11
    *   **CUDA**: 12.4

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ 使用说明

### 1. 数据处理

将 `question.csv` 和 `answer.csv` 放入 `data/raw/` 目录，然后运行：

```bash
python scripts/data_processing/process_csv_to_jsonl.py 
```

*   **功能**: 自动合并问题和答案，清洗空数据，并转换为 ChatML 格式。
*   **参数**: 可通过 `--limit` 参数限制处理条数用于测试。

### 2. 模型微调 (LoRA)

运行微调脚本启动训练：

```bash
python scripts/training/finetune_qwen_lora.py
```

*   **输入**: 读取 `data/processed/train_peft.jsonl`。
*   **输出**: 训练好的 LoRA 权重将保存在 `models/checkpoints/qwen_lora_<TIMESTAMP>` 下。
*   **配置**: 可在脚本中调整 `BATCH_SIZE`, `LEARNING_RATE` 等超参数。

### 3. 推理服务 (Inference)

启动 vLLM 推理引擎，支持动态加载 LoRA：

```bash
    python src/inference/vllm_lora_inference.py \
        --base_model models/base/Qwen/Qwen3-8B \
        --lora_path models/checkpoints/qwen_lora_<TIMESTAMP>/final_model  
    ```

### 4. 模型合并 (Merge LoRA)

为了方便部署或减少推理时的显存开销，你可以将 LoRA 权重永久合并到基础模型中：

```bash
python src/tools/merge_lora.py \
    --base_model models/base/Qwen/Qwen3-8B \
    --lora_model models/checkpoints/qwen_lora_<TIMESTAMP>/final_model \
    --output_dir models/merged/Qwen-Med-Merged
```

*   **功能**: 将 LoRA 权重与基础模型合并，生成一个独立的完整模型。
*   **优势**: 合并后的模型不需要加载额外的适配器，推理速度更快，且兼容所有支持标准 Transformer 模型的工具。

## 📊 数据处理流程

1.  **Ingest**: 读取 CSV 文件（自动识别 UTF-8 或 GBK 编码）。
2.  **Merge**: 根据 `question_id` 将问题与答案进行 Inner Join。
3.  **Clean**: 去除空值、去除无关字符。
4.  **Format**: 封装为 ChatML 格式：
    ```json
    {"messages": [
        {"role": "system", "content": "你是一个专业的医疗助手，能够提供准确的医疗咨询。"},
        {"role": "user", "content": "感冒了怎么办？"},
        {"role": "assistant", "content": "建议多喝水..."}
    ]}
    ```
5.  **Output**: 写入 JSONL 文件，供微调使用。