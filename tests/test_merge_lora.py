import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 将项目根目录添加到 sys.path，确保能导入 src 下的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Mock peft module before importing merge_lora
sys.modules['peft'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# 尝试导入待测试的模块 (在 TDD 初期这个文件可能还没写，所以用 try-except 包裹以便测试能运行)
try:
    from src.tools.merge_lora import merge_lora_models
except ImportError:
    merge_lora_models = None

class TestMergeLoRA(unittest.TestCase):
    def setUp(self):
        self.base_model_path = "./mock_base_model"
        self.lora_model_path = "./mock_lora_path"
        self.output_path = "./merged_model_output"

    @patch('src.tools.merge_lora.PeftModel')
    @patch('src.tools.merge_lora.AutoModelForCausalLM')
    @patch('src.tools.merge_lora.AutoTokenizer')
    def test_merge_success_flow(self, MockTokenizer, MockAutoModel, MockPeftModel):
        """
        集成测试：验证完整的模型加载、合并、保存流程
        """
        if merge_lora_models is None:
            self.skipTest("merge_lora 模块尚未实现")

        # 1. 模拟基础模型加载
        mock_base_model = MagicMock()
        MockAutoModel.from_pretrained.return_value = mock_base_model
        
        # 2. 模拟 Tokenizer 加载
        mock_tokenizer = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        
        # 3. 模拟 LoRA 模型加载
        mock_peft_model = MagicMock()
        MockPeftModel.from_pretrained.return_value = mock_peft_model
        
        # 4. 模拟合并后的模型 (merge_and_unload 返回的对象)
        mock_merged_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model

        # === 执行被测函数 ===
        merge_lora_models(self.base_model_path, self.lora_model_path, self.output_path)

        # === 验证断言 ===
        
        # 验证基础模型是否正确加载 (必须使用 CPU 加载以防爆显存，且 trust_remote_code=True)
        MockAutoModel.from_pretrained.assert_called_once_with(
            self.base_model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )

        # 验证 PeftModel 是否挂载到了基础模型上
        MockPeftModel.from_pretrained.assert_called_once_with(
            mock_base_model,
            self.lora_model_path
        )

        # 验证是否执行了合并操作
        mock_peft_model.merge_and_unload.assert_called_once()

        # 验证是否保存了合并后的模型
        mock_merged_model.save_pretrained.assert_called_once_with(self.output_path)

        # 验证是否保存了 Tokenizer (这很重要，否则部署时会缺文件)
        mock_tokenizer.save_pretrained.assert_called_once_with(self.output_path)

    @patch('src.tools.merge_lora.AutoModelForCausalLM')
    def test_error_handling(self, MockAutoModel):
        """
        边界测试：验证路径不存在时的错误处理
        """
        if merge_lora_models is None:
            self.skipTest("merge_lora 模块尚未实现")

        # 模拟加载模型时抛出 OSError
        MockAutoModel.from_pretrained.side_effect = OSError("Model not found")
        
        with self.assertRaises(OSError):
            merge_lora_models(self.base_model_path, self.lora_model_path, self.output_path)

if __name__ == '__main__':
    unittest.main()
