from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen25_caption_backend as backend


class Qwen25BackendTests(unittest.TestCase):
    def test_loads_thinker_and_attaches_adapter_to_thinker(self):
        model = MagicMock()
        processor = MagicMock()
        model_class = MagicMock()
        model_class.from_pretrained.return_value = model
        processor_class = MagicMock()
        processor_class.from_pretrained.return_value = processor
        transformers = SimpleNamespace(Qwen2_5OmniProcessor=processor_class,
                                       Qwen2_5OmniThinkerForConditionalGeneration=model_class)
        descriptor = {"path": "/adapter"}
        adapted = MagicMock()
        with patch.dict(sys.modules, {"transformers": transformers,
                                      "qwen_omni_utils": SimpleNamespace(process_mm_info=MagicMock())}), \
             patch.object(backend, "describe_adapter", return_value=descriptor), \
             patch.object(backend, "load_peft_adapter", return_value=(adapted, descriptor)) as attach:
            obj = backend.Qwen25Captioner("/model", "/adapter")
        attach.assert_called_once_with(model, descriptor)
        self.assertIs(obj.model, adapted)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertTrue(model_class.from_pretrained.call_args.kwargs["local_files_only"])
        adapted.requires_grad_.assert_called_once_with(False)

    def test_generation_preserves_batch_and_decodes_only_completion(self):
        obj = backend.Qwen25Captioner.__new__(backend.Qwen25Captioner)
        obj.torch = torch
        obj.model = MagicMock()
        obj.model.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        obj.model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[0, 10, 11, 90], [20, 21, 22, 91]]))
        obj.processor = MagicMock()
        obj.processor.tokenizer.eos_token_id = 2
        obj.processor.tokenizer.pad_token_id = 0
        obj.processor.return_value = {
            "input_ids": torch.tensor([[0, 10, 11], [20, 21, 22]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "input_features": torch.ones(2, 3),
        }
        obj.processor.batch_decode.return_value = [" A ", " B "]
        obj.process_mm_info = MagicMock(return_value=(["audio1", "audio2"], None, None))
        result = obj.generate_batch(["/a.wav", "/b.wav"], ["first", "second"], 96)
        self.assertEqual(result, ["A", "B"])
        self.assertEqual(obj.processor.batch_decode.call_args.args[0].tolist(), [[90], [91]])
        kwargs = obj.model.generate.call_args.kwargs
        self.assertEqual(kwargs["max_new_tokens"], 96)
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("return_audio", kwargs)
        self.assertNotIn("thinker_max_new_tokens", kwargs)
        self.assertEqual(kwargs["input_features"].dtype, torch.bfloat16)
        self.assertEqual(kwargs["input_ids"].dtype, torch.int64)
        self.assertEqual(obj.generate_batch([], [], 96), [])
        with self.assertRaises(ValueError):
            obj.generate_batch(["/a.wav"], [], 96)


if __name__ == "__main__":
    unittest.main()
