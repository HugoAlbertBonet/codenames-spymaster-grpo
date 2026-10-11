import tempfile
import unittest
from pathlib import Path

from grpo_reference import frozen_adapter, make_sft_reference
from train_grpo_unsloth import parse_args


class SFTReferenceTests(unittest.TestCase):
    def test_reference_remains_sft_while_policy_changes(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(12)
        model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)),
            LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        for key, parameter in model.named_parameters():
            if "lora_B" in key:
                torch.nn.init.normal_(parameter, std=0.1)
        model.eval()
        inputs = {"input_ids": torch.tensor([[1, 3, 5, 7]])}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reference = make_sft_reference(model, directory)
            before = model(**inputs).logits.detach().clone()
            self.assertTrue(torch.equal(before, reference(**inputs).logits))
            with model.disable_adapter():
                self.assertFalse(torch.allclose(before, model(**inputs).logits))
            self.assertEqual(list(reference.parameters()), [])
            flags = [p.requires_grad for p in model.parameters()]
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with frozen_adapter(model, "sft_reference"):
                    raise RuntimeError("test failure")
            self.assertEqual(model.active_adapter, "default")
            self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
            for key, parameter in model.named_parameters():
                if "lora_B.default" in key:
                    with torch.no_grad():
                        parameter.add_(0.2)
            self.assertFalse(torch.allclose(before, model(**inputs).logits))
            self.assertTrue(torch.equal(before, reference(**inputs).logits))
            reference.audit()
            model(**inputs).logits.sum().backward()
            self.assertTrue(any(p.grad is not None for k, p in model.named_parameters()
                                if ".default." in k))
            self.assertFalse(any(p.grad is not None for k, p in model.named_parameters()
                                 if ".sft_reference." in k))
            checkpoint = Path(directory) / "checkpoint"
            model.save_pretrained(checkpoint)
            self.assertTrue((checkpoint / "adapter_model.safetensors").exists())
            self.assertFalse((checkpoint / "sft_reference").exists())
            # Exercise the actual Transformers resume path with two loaded
            # adapters but a policy-only checkpoint, not a hand-written reload.
            from transformers import Trainer, TrainingArguments
            from grpo_reference import adapter_digest
            trained_digest = adapter_digest(model, "default")
            for key, parameter in model.named_parameters():
                if "lora_B.default" in key:
                    with torch.no_grad():
                        parameter.add_(0.1)
            Trainer(model=model, args=TrainingArguments(output_dir=directory, report_to="none",
                    use_cpu=True))._load_from_checkpoint(str(checkpoint))
            self.assertEqual(adapter_digest(model, "default"), trained_digest)
            reference.audit()

    def test_experiment_flags(self):
        args = parse_args(["--reference-policy", "sft", "--beta", "0.04",
                           "--game-reward-only", "--fresh-boards", "--save-total-limit", "6"])
        self.assertEqual(args.reference_policy, "sft")
        self.assertTrue(args.game_reward_only)
        self.assertEqual(args.save_total_limit, 6)
