import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_models(base_model_path: str, lora_model_path: str, output_path: str):
    """
    加载基础模型和 LoRA 权重，合并后保存为独立模型。
    
    Args:
        base_model_path: 基础模型路径
        lora_model_path: LoRA 适配器路径
        output_path: 合并后模型的保存路径
    """
    print(f"[-] 正在加载基础模型: {base_model_path}")
    # 注意：合并时建议使用 float16 或 bfloat16，避免精度损失
    # device_map="auto" 会自动利用 GPU 加速合并过程
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    print(f"[-] 正在加载 LoRA 适配器: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    print("[-] 正在执行权重合并 (merge_and_unload)...")
    # 这一步是将 LoRA 的权重矩阵 A*B 加回到基础模型的权重 W 中
    # W_new = W + A*B
    model = model.merge_and_unload()

    print(f"[-] 正在保存合并后的模型至: {output_path}")
    model.save_pretrained(output_path)

    print("[-] 正在保存 Tokenizer...")
    # Tokenizer 通常不需要合并，直接从基础模型加载并保存一份副本即可
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"[+] 合并完成！模型已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base LLM")
    parser.add_argument("--base_model", type=str, default="./models/base/Qwen/Qwen3-8B", help="基础模型路径 (Base Model Path)")
    parser.add_argument("--lora_model", type=str, default="./models/checkpoints/qwen_lora_20260202_162449/final_model", help="LoRA 权重路径 (LoRA Adapter Path)")
    parser.add_argument("--output_dir", type=str, default="./models/merged/Qwen-Med-Merged", help="输出目录 (Output Directory)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    merge_lora_models(args.base_model, args.lora_model, args.output_dir)

if __name__ == "__main__":
    main()
