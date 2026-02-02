import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

# 在导入脚本之前 Mock vllm 模块
# Mock vllm module before importing the script
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.lora'] = MagicMock()
sys.modules['vllm.lora.request'] = MagicMock()

# 添加项目根目录到 sys.path
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import vllm_lora_inference

class TestIntegrationLocal(unittest.TestCase):
    @patch('vllm_lora_inference.LLM')
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_flow(self, mock_stdout, MockLLM):
        """
        本地集成测试：验证 main 函数在 Mock 环境下的完整流程
        Local Integration Test: Verify the full flow of main function in a Mock environment
        """
        # 设置 Mock LLM 行为
        # Setup Mock LLM behavior
        mock_instance = MockLLM.return_value
        
        # 创建模拟输出对象，模仿 vLLM 输出结构
        # Create a mock output object that mimics vLLM output structure
        # outputs[0].outputs[0].text
        mock_completion =MagicMock(outputs=[MagicMock(text="Mocked Response")])
        
        # generate 返回请求输出列表
        # generate returns a list of request outputs
        mock_instance.generate.return_value = [mock_completion]
        
        # 运行 main 函数
        # Run main function
        print("Starting integration test...")
        vllm_lora_inference.main()
        
        # 获取标准输出内容
        # Get stdout content
        output = mock_stdout.getvalue()
        
        # 验证关键日志输出
        # Verify key log outputs
        self.assertIn("基础模型: ./qwen_models/Qwen/Qwen3-8B", output)
        self.assertIn("正在初始化 vLLM 引擎", output)
        self.assertIn("基础模型回答: Mocked Response", output)
        self.assertIn("医疗 LoRA 回答: Mocked Response", output)
        
        # 验证调用次数
        # Verify call counts
        # 我们期望 2 次 generate 调用：一次基础模型，一次医疗 LoRA
        # We expect 2 generate calls: one for base, one for medical lora
        self.assertEqual(mock_instance.generate.call_count, 2)
        print("Integration test passed!")

if __name__ == "__main__":
    unittest.main()
