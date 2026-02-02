import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 模拟 vllm 模块，防止在没有安装 vllm 的环境中报错
# Mock vllm module to prevent errors in environments without vllm installed
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.lora'] = MagicMock()
sys.modules['vllm.lora.request'] = MagicMock()

# 添加项目根目录到 sys.path 以便导入模块
# Add project root to sys.path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入被测试的类
# Import the class to be tested
from src.inference.vllm_lora_inference import MultiLoRAInference

class TestMultiLoRAInference(unittest.TestCase):
    def setUp(self):
        self.base_model_path = "./mock_base_model"
        self.lora_path = "./mock_lora_path"
        self.engine = MultiLoRAInference(self.base_model_path, self.lora_path)

    @patch('src.inference.vllm_lora_inference.LLM')
    @patch('src.inference.vllm_lora_inference.SamplingParams')
    def test_initialize_engine(self, MockSamplingParams, MockLLM):
        """
        测试引擎初始化
        Test engine initialization
        """
        self.engine.initialize_engine()
        
        # 验证 LLM 初始化参数
        # Verify LLM initialization parameters
        MockLLM.assert_called_once()
        call_args = MockLLM.call_args[1]
        self.assertEqual(call_args['model'], self.base_model_path)
        self.assertTrue(call_args['enable_lora'])
        self.assertEqual(call_args['max_lora_rank'], 16)
        self.assertEqual(call_args['max_model_len'], 2048)
        
        # 验证 SamplingParams 初始化参数
        # Verify SamplingParams initialization parameters
        MockSamplingParams.assert_called_once_with(temperature=0.7, max_tokens=4096)

    @patch('src.inference.vllm_lora_inference.LLM')
    def test_generate_base(self, MockLLM):
        """
        测试基础模型生成
        Test base model generation
        """
        # 设置 mock 引擎
        # Setup mock engine
        mock_llm_instance = MockLLM.return_value
        self.engine.llm = mock_llm_instance
        self.engine.sampling_params = MagicMock()
        
        # 设置 mock 返回值
        # Setup mock return value
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Base response")]
        mock_llm_instance.generate.return_value = [mock_output]
        
        prompt = "Test prompt"
        response = self.engine.generate_base(prompt)
        
        self.assertEqual(response, "Base response")
        mock_llm_instance.generate.assert_called_once_with(prompt, self.engine.sampling_params)

    @patch('src.inference.vllm_lora_inference.LLM')
    @patch('src.inference.vllm_lora_inference.LoRARequest')
    def test_generate_with_lora(self, MockLoRARequest, MockLLM):
        """
        测试带 LoRA 的生成
        Test generation with LoRA
        """
        # 设置 mock 引擎
        # Setup mock engine
        mock_llm_instance = MockLLM.return_value
        self.engine.llm = mock_llm_instance
        self.engine.sampling_params = MagicMock()
        
        # 设置 mock 返回值
        # Setup mock return value
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="LoRA response")]
        mock_llm_instance.generate.return_value = [mock_output]
        
        prompt = "Test prompt"
        lora_name = "test_adapter"
        lora_id = 1
        
        response = self.engine.generate_with_lora(prompt, lora_name, lora_id)
        
        self.assertEqual(response, "LoRA response")
        
        # 验证 LoRARequest 创建正确
        # Verify LoRARequest was created correctly
        MockLoRARequest.assert_called_once_with(lora_name, lora_id, self.lora_path)
        
        # 验证 generate 调用包含了 lora_request
        # Verify generate was called with lora_request
        mock_llm_instance.generate.assert_called_once()
        call_args = mock_llm_instance.generate.call_args
        self.assertEqual(call_args[0][0], prompt)
        self.assertEqual(call_args[0][1], self.engine.sampling_params)
        self.assertEqual(call_args[1]['lora_request'], MockLoRARequest.return_value)

if __name__ == '__main__':
    unittest.main()