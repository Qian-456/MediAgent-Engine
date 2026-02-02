import os
import json
import torch
from datetime import datetime
from datasets import load_dataset, Dataset
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

# 确保使用GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """
    主函数：执行Qwen2.5-7B模型的PEFT微调任务。
    """
    print(f"正在使用设备: {device}")
    torch.cuda.empty_cache() # 清理显存

    # ==========================================
    # 1. 配置参数
    # ==========================================
    
    # 路径配置 (假设脚本位于 scripts/training/ 目录)
    # 使用绝对路径或相对于项目根目录的路径
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    
    # 确保目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 模型和数据路径
    # 使用 ModelScope 下载模型，解决 HuggingFace 连接问题
    print("正在从 ModelScope 下载/加载模型...")
    # 显式指定本地存储路径
    model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir=MODEL_DIR, revision='master')
    model_id = model_dir 
    
    dataset_path = os.path.join(DATA_DIR, "train_peft.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}。请先运行数据处理脚本。")
    
    # 使用时间戳作为输出目录后缀，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qwen_lora_{timestamp}"
    output_dir = os.path.join(OUTPUT_DIR, run_name)
    print(f"输出目录: {output_dir}")
    
    # 训练超参数
    MAX_LENGTH = 512
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1 # 降低 Batch Size 以减少显存占用
    GRAD_ACCUMULATION_STEPS = 4 # 增加梯度累积以保持等效 Batch Size
    NUM_EPOCHS = 3
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    WARMUP_RATIO = 0.1 # 预热比例 10%
    
    # ==========================================
    # 2. 数据准备
    # ==========================================
    
    print("正在加载数据集...")
    # 加载本地JSON数据
    try:
        data = load_dataset("json", data_files=dataset_path, split="train")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_val_split = data.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # ==========================================
    # 3. 模型与分词器加载
    # ==========================================
    
    print("正在加载分词器和模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Qwen 的 pad_token 处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置 (4-bit) 以节省显存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 为 k-bit 训练做准备
    model = prepare_model_for_kbit_training(model)
    
    # Gradient Checkpointing 开启时，use_cache 必须为 False
    model.config.use_cache = False

    # ==========================================
    # 4. PEFT (LoRA) 配置
    # ==========================================
    
    print("配置 LoRA...")
    # 找到模型中所有的线性层作为 target_modules
    # 对于 Qwen3，通常是 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # 简单起见，我们可以让 peft 自动识别，或者指定常用层
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ==========================================
    # 5. 数据预处理函数
    # ==========================================
    
    def process_func(example):
        """
        数据预处理函数：将输入和输出组合成模型训练格式。
        适配 ChatML (messages) 格式。
        
        @param {Object} example - 数据集中的单条样本
        @returns {Object} - 包含 input_ids, attention_mask, labels 的字典
        """
        messages = example.get("messages", [])
        if not messages:
             return {"input_ids": [], "attention_mask": [], "labels": []}
             
        # 使用 tokenizer 的 chat template 功能 (如果可用且正确配置)
        # 或者手动构建 ChatML 格式:
        # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        
        try:
            # 尝试使用 tokenizer 的 apply_chat_template
            # 确保 tokenizer 有 chat_template 属性，Qwen3 通常默认有
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False 
            )
        except Exception:
            # 手动 fallback 构建 (简化版 ChatML)
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text += tokenizer.eos_token
            
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        
        # 将 padding 部分的 label 设为 -100
        for i in range(len(labels)):
             if labels[i] == tokenizer.pad_token_id:
                 labels[i] = -100
                 
        tokenized["labels"] = labels
        return tokenized

    print("正在处理数据集...")
    tokenized_train = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)

    # ==========================================
    # 6. 训练参数设置
    # ==========================================
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch", # 使用 eval_strategy 替代 evaluation_strategy (新版 transformers)
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True, # 开启混合精度训练
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS, # 显存不足时可增加此值
        gradient_checkpointing=True, # 开启梯度检查点，大幅降低显存占用
        report_to=["tensorboard"], # 记录日志
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        lr_scheduler_type="cosine", # 使用余弦退火调度器
        warmup_ratio=WARMUP_RATIO, # 设置预热比例
    )

    # ==========================================
    # 7. 初始化 Trainer 并开始训练
    # ==========================================
    
    # 定义一个自定义回调，用于实时打印 Loss
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                _ = logs.pop("total_flos", None)
                if "loss" in logs:
                    print(f"Step: {state.global_step}, Loss: {logs['loss']:.4f}")
                if "eval_loss" in logs:
                    print(f"Step: {state.global_step}, Eval Loss: {logs['eval_loss']:.4f}")
                    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[PrinterCallback] # 添加回调
    )

    print("开始训练...")
    trainer.train()

    # ==========================================
    # 8. 保存模型
    # ==========================================
    
    print("保存模型...")
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"训练完成！模型已保存至 {final_save_path}")

if __name__ == "__main__":
    main()
