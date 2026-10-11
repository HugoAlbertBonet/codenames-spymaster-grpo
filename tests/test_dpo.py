import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from dpo_data import PreferenceCollator, load_preferences, tokenize_preference, validate_preference
from dpo_loss import forward_response_logps, make_dpo_trainer_class, preference_metrics, response_logps, sigmoid_dpo_loss
from train_dpo import build_dpo_arguments, parse_args, validate_reference_cache


class DPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_logps_shift_mask_and_tail_match(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 7, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 3, 4, 5, -100],
                               [-100, -100, -100, -100, 6, 7, 8]])
        result = response_logps(logits, labels)
        expected = torch.stack([sum(logits[i, j - 1].log_softmax(-1)[labels[i, j]]
                                    for j in range(1, 7) if labels[i, j] != -100) for i in range(2)])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result, response_logps(logits[:, 2:], labels))
        result.sum().backward()
        self.assertEqual(logits.grad[:, :2].abs().sum(), 0)
        self.assertGreater(logits.grad[:, 2:].abs().sum(), 0)
        with self.assertRaises(ValueError):
            response_logps(logits, torch.full_like(labels, -100))

    def test_dpo_frozen_reference_and_gradient_direction(self):
        chosen = torch.tensor([-8.0], requires_grad=True)
        rejected = torch.tensor([-9.0], requires_grad=True)
        refs = [x.detach().clone().requires_grad_() for x in (chosen, rejected)]
        loss, positive, negative = sigmoid_dpo_loss(chosen, rejected, *refs, beta=0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(positive.item(), 0)
        self.assertEqual(negative.item(), 0)
        loss.mean().backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertIsNone(refs[0].grad)
        self.assertIsNone(refs[1].grad)
        for beta in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                sigmoid_dpo_loss(chosen, rejected, *refs, beta=beta)

    def test_collator_orders_pairs_masks_padding_and_keeps_refs(self):
        rows = [{"chosen_input_ids": [1, 2, 3], "chosen_labels": [-100, 2, 3],
                 "rejected_input_ids": [1, 4], "rejected_labels": [-100, 4],
                 "ref_chosen_logps": -2., "ref_rejected_logps": -3.},
                {"chosen_input_ids": [5, 6], "chosen_labels": [-100, 6],
                 "rejected_input_ids": [5, 7, 8], "rejected_labels": [-100, 7, 8],
                 "ref_chosen_logps": -4., "ref_rejected_logps": -5.}]
        batch = PreferenceCollator(0)(rows)
        self.assertEqual(batch['input_ids'].tolist(), [[1, 2, 3], [5, 6, 0], [1, 4, 0], [5, 7, 8]])
        self.assertEqual(batch['labels'][1].tolist(), [-100, 6, -100])
        self.assertEqual(batch['attention_mask'][2].tolist(), [1, 1, 0])
        self.assertEqual(batch['ref_chosen_logps'].tolist(), [-2., -4.])

    def test_tokenization_supervises_response_and_end_marker_only(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                prefix = '<user>' + messages[0]['content'] + '<assistant>'
                return prefix if add_generation_prompt else prefix + messages[-1]['content'] + '<end>'
            def __call__(self, text, add_special_tokens):
                return {'input_ids': [ord(c) for c in text]}
        row = {'prompt': [{'role': 'user', 'content': 'board'}],
               'chosen': [{'role': 'assistant', 'content': 'CLUE: orbit\nNUMBER: 1'}],
               'rejected': [{'role': 'assistant', 'content': 'CLUE: sports\nNUMBER: 2'}]}
        encoded = tokenize_preference(row, Tokenizer(), 200)
        for side in ('chosen', 'rejected'):
            response = ''.join(chr(x) for x in encoded[side + '_labels'] if x != -100)
            self.assertEqual(response, row[side][0]['content'] + '<end>')
        with self.assertRaises(ValueError):
            tokenize_preference(row, Tokenizer(), 10)

    def test_reference_cache_requires_identical_identity_and_valid_scores(self):
        cache = {'identity': {'beta': 0.1}, 'logps': {'train': [[-2., -3.]], 'validation': []}}
        validate_reference_cache(cache, cache['identity'], {'train': 2, 'validation': 1})
        for changed in ({'beta': 0.2}, {}):
            with self.assertRaises(ValueError):
                validate_reference_cache(cache, changed, {'train': 2, 'validation': 1})
        for invalid in (float('nan'), 1.):
            broken = copy.deepcopy(cache)
            broken['logps']['train'][0][0] = invalid
            with self.assertRaises(ValueError):
                validate_reference_cache(broken, cache['identity'], {'train': 2, 'validation': 1})

    def test_training_arguments_use_explicit_warmup(self):
        from transformers import TrainingArguments
        args = parse_args(['--max-steps', '5'])
        values = build_dpo_arguments(args, 298, TrainingArguments, cuda=False, bf16=False)
        self.assertEqual(values.warmup_steps, 1)
        self.assertFalse(values.remove_unused_columns)
        self.assertEqual(values.label_names, [])

    @unittest.skipUnless(Path('data/dpo/audit.json').exists() and
                         Path('outputs/codenames-sft-1p5b/adapter_model.safetensors').exists(),
                         'original local SFT adapter required for weight-provenance check')
    def test_full_dataset_certificate_and_saved_outcomes(self):
        train, validation, _ = load_preferences(Path('data/dpo'), Path('outputs/codenames-sft-1p5b'),
                                                'Qwen/Qwen2.5-1.5B-Instruct')
        self.assertEqual((len(train), len(validation)), (298, 62))
        broken = copy.deepcopy(train[0])
        broken['chosen_outcome']['reward'] += 1
        with self.assertRaises(ValueError):
            validate_preference(broken)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'preferences'
            shutil.copytree('data/dpo', copied)
            with (copied / 'train.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'Audited artifact changed'):
                load_preferences(copied, Path('outputs/codenames-sft-1p5b'), 'Qwen/Qwen2.5-1.5B-Instruct')
        broken = copy.deepcopy(train[0])
        broken['prompt'][0]['content'] += ' altered'
        with self.assertRaises(ValueError):
            validate_preference(broken)

    def test_published_certificates_and_outcomes_without_model_weights(self):
        for directory, expected, mode in ((Path('data/dpo'), (298, 62), 'gameplay'),
                                           (Path('data/dpo-clue-only'), (261, 57), 'clue-only')):
            audit = json.loads((directory / 'audit.json').read_text())
            self.assertEqual(set(audit['audited_files']),
                             {'manifest.json', 'boards.jsonl', 'candidates.jsonl',
                              'train.jsonl', 'validation.jsonl'})
            for name, digest in audit['audited_files'].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)
            counts = []
            boards = set()
            for split in ('train', 'validation'):
                rows = [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]
                counts.append(len(rows))
                for row in rows:
                    validate_preference(row, mode)
                    self.assertEqual(row['split'], split)
                    self.assertNotIn(row['board_key'], boards)
                    boards.add(row['board_key'])
            self.assertEqual(tuple(counts), expected)
            self.assertEqual(audit['pairs_replayed'], len(boards))

    def test_real_tiny_lora_training_and_evaluation(self):
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
        torch.manual_seed(4)
        model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, max_position_embeddings=32)),
                LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        rows = [{'chosen_input_ids': [1, 2, 3, 4], 'chosen_labels': [-100, -100, 3, 4],
                 'rejected_input_ids': [1, 2, 5, 6], 'rejected_labels': [-100, -100, 5, 6]}] * 2
        collator = PreferenceCollator(0)
        with torch.no_grad():
            full = model(input_ids=collator(rows)['input_ids']).logits
            tail = forward_response_logps(model, collator(rows))
            torch.testing.assert_close(tail, response_logps(full, collator(rows)['labels']))
        for i, row in enumerate(rows):
            row['ref_chosen_logps'], row['ref_rejected_logps'] = tail[i].item(), tail[len(rows) + i].item()
        original = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_dpo_trainer_class()(model=model, beta=.1, data_collator=collator,
                train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows),
                compute_metrics=preference_metrics, args=TrainingArguments(output_dir=directory,
                max_steps=2, learning_rate=.01, per_device_train_batch_size=1,
                report_to='none', save_strategy='no', use_cpu=True, remove_unused_columns=False, label_names=[]))
            self.assertAlmostEqual(trainer.evaluate()['eval_loss'], math.log(2), places=5)
            trainer.train()
            metrics = trainer.evaluate()
            self.assertLess(metrics['eval_loss'], math.log(2))
            self.assertTrue(any(not torch.equal(p, original[n]) for n, p in model.named_parameters() if p.requires_grad))
            self.assertEqual(metrics['eval_preference_accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()
