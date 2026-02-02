import os
import re
import json
from datetime import datetime
from typing import Optional, Any

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    # 允许在没有安装 vllm 的环境中导入此文件进行测试
    # Allow importing in environments without vllm for testing purposes
    LLM = Any
    SamplingParams = Any
    LoRARequest = Any

"""
@description vLLM 多 LoRA 适配器热加载与运行时切换示例

此脚本演示了如何:
1. 加载基础模型 (Qwen/Qwen3-8B)
2. 启用 LoRA 支持
3. 在推理时动态加载并使用特定的 LoRA 适配器 (如医疗微调适配器)
4. 演示如何在不同适配器之间切换 (Multi-domain Adapter Runtime Switch)
"""

class MultiLoRAInference:
    """
    vLLM 多 LoRA 推理引擎封装类
    """

    def __init__(self, base_model_path: str, lora_path: str):
        """
        初始化推理引擎配置
        @param {string} base_model_path - 基础模型路径
        @param {string} lora_path - LoRA 适配器路径
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.llm = None
        self.sampling_params = None

    def initialize_engine(self, max_model_len: int = 2048, gpu_memory_utilization: float = 0.9):
        """
        初始化 vLLM 引擎
        @param {number} [max_model_len=2048] - 最大模型上下文长度
        @param {number} [gpu_memory_utilization=0.9] - GPU 显存占用比例
        """
        print("正在初始化 vLLM 引擎 (启用 LoRA)...")
        
        # 如果是在测试环境中 mock 了 LLM，这里也可以运行
        # If LLM is mocked in test env, this can still run
        
        self.llm = LLM(
            model=self.base_model_path,
            enable_lora=True,
            max_lora_rank=16, 
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization, 
            max_model_len=max_model_len,         
            enforce_eager=True          
        )

        # 采样参数
        # 增加 max_tokens 以避免回复被截断 (default 100 -> 512)
        # Increase max_tokens to avoid response truncation
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)

    def _clean_response(self, text: str) -> str:
        """
        清理模型输出，去除思考过程标签
        Clean model output, remove thinking process tags (<think>...</think>)
        @param {string} text - 原始输出文本
        @returns {string} - 清理后的文本
        """
        # 去除 <think>...</think> 标签及其内容 (支持跨行)
        # Remove <think>...</think> tags and their content (dotall mode)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def generate_base(self, prompt: str) -> str:
        """
        使用基础模型生成回复 (不使用 LoRA)
        @param {string} prompt - 输入提示词
        @returns {string} - 生成的文本
        """
        if self.llm is None:
            raise RuntimeError("Engine not initialized")

        outputs = self.llm.generate(
            prompt, 
            self.sampling_params
        )
        raw_text = outputs[0].outputs[0].text
        return self._clean_response(raw_text)

    def generate_with_lora(self, prompt: str, lora_name: str, lora_id: int) -> str:
        """
        使用指定的 LoRA 适配器生成回复
        @param {string} prompt - 输入提示词
        @param {string} lora_name - LoRA 适配器名称标识
        @param {number} lora_id - LoRA 适配器 ID
        @returns {string} - 生成的文本
        """
        if self.llm is None:
            raise RuntimeError("Engine not initialized")

        outputs = self.llm.generate(
            prompt, 
            self.sampling_params,
            lora_request=LoRARequest(lora_name, lora_id, self.lora_path)
        )
        raw_text = outputs[0].outputs[0].text
        return self._clean_response(raw_text)

def main():
    # ==========================================
    # 1. 配置路径
    # ==========================================
    
    base_model_path = './models/base/Qwen/Qwen3-8B' 
    medical_lora_path = './models/checkpoints/qwen_lora_20260202_162449/final_model'
    
    print(f"基础模型: {base_model_path}")
    print(f"医疗适配器: {medical_lora_path}")

    # ==========================================
    # 2. 初始化引擎
    # ==========================================
    
    engine = MultiLoRAInference(base_model_path, medical_lora_path)
    engine.initialize_engine()

    # ==========================================
    # 3. 准备测试 Prompt
    # ==========================================
    
    medical_prompt = "<|im_start|>user\n感冒了应该吃什么药？<|im_end|>\n<|im_start|>assistant\n"
    
    # ==========================================
    # 4. 运行时切换演示
    # ==========================================

    base_response = ""
    lora_response = ""

    print("\n=== 场景 1: 不使用 LoRA (纯基础模型) ===")
    try:
        base_response = engine.generate_base(medical_prompt)
        print(f"基础模型回答: {base_response}")
    except Exception as e:
        print(f"基础模型推理出错: {e}")
        base_response = f"Error: {e}"

    print("\n=== 场景 2: 动态加载并使用医疗 LoRA 适配器 ===")
    try:
        lora_response = engine.generate_with_lora(medical_prompt, "medical_adapter", 1)
        print(f"医疗 LoRA 回答: {lora_response}")
    except Exception as e:
        print(f"LoRA 推理出错: {e}")
        lora_response = f"Error: {e}"

    # ==========================================
    # 5. 保存结果
    # ==========================================
    result_record = {
        "timestamp": datetime.now().isoformat(),
        "prompt": medical_prompt,
        "base_model_response": base_response,
        "lora_model_response": lora_response,
        "lora_path": medical_lora_path
    }

    output_file = "inference_results.jsonl"
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        print(f"\n[Success] 推理结果已追加保存至: {output_file}")
    except Exception as e:
        print(f"\n[Error] 保存结果失败: {e}")

if __name__ == "__main__":
    main()

