"""Standard sigmoid DPO with a precomputed, frozen SFT reference.

This small Trainer extension avoids TRL-version-specific adapter defaults. The
objective is the ordinary summed response-log-probability DPO loss, not SFT,
reference-free training, a length-normalized variant or a learned reward model.
"""

from __future__ import annotations

import inspect
import math


def response_logps(logits, labels):
    """Sum causal log probabilities only for assistant tokens; keep gradients."""
    import torch
    from torch.nn import functional as F
    labels = labels[:, -logits.shape[1]:]
    targets = labels[:, 1:]
    mask = targets != -100
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Each response must have supervised tokens")
    # Materialize FP32 vocabulary logits only for the short response, not the prompt.
    selected_logits = logits[:, :-1, :][mask].float()
    token_logps = -F.cross_entropy(selected_logits, targets[mask], reduction="none")
    batch_indices = mask.nonzero(as_tuple=True)[0]
    return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device).scatter_add(
        0, batch_indices, token_logps)


def forward_response_logps(model, batch):
    labels = batch["labels"]
    # Keep logits from just before the first supervised token through the response.
    # Both stock Qwen and Unsloth support this, avoiding prompt-length vocabulary tensors.
    positions = (labels != -100).nonzero(as_tuple=True)[1]
    if positions.numel() == 0:
        raise ValueError("No response tokens")
    keep = labels.shape[1] - int(positions.min()) + 1
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    supported = inspect.signature(base.forward).parameters
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
              "use_cache": False, "return_dict": True}
    if "logits_to_keep" in supported:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in supported:
        kwargs["num_logits_to_keep"] = keep
    output = model(**kwargs)
    return response_logps(output.logits, labels)


def sigmoid_dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta: float):
    """Reference scores are constants even if a caller accidentally marks them trainable."""
    from torch.nn import functional as F
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    chosen_reward = beta * (chosen_logps - ref_chosen_logps.detach())
    rejected_reward = beta * (rejected_logps - ref_rejected_logps.detach())
    return -F.logsigmoid(chosen_reward - rejected_reward), chosen_reward.detach(), rejected_reward.detach()


def preference_metrics(prediction) -> dict:
    import numpy as np
    values = prediction.predictions
    margin = values[:, 0] - values[:, 1]
    return {"preference_accuracy": float(np.mean(margin > 0)),
            "preference_margin": float(margin.mean()),
            "chosen_implicit_reward": float(values[:, 0].mean()),
            "rejected_implicit_reward": float(values[:, 1].mean()),
            "chosen_response_logp": float(values[:, 2].mean()),
            "rejected_response_logp": float(values[:, 3].mean())}


def make_dpo_trainer_class():
    # Import AFTER optional Unsloth patches, just as train_sft.py does.
    from transformers import Trainer

    class OutcomeDPOTrainer(Trainer):
        def __init__(self, *args, beta=0.1, **kwargs):
            self.beta = beta
            super().__init__(*args, **kwargs)
            # Loss is a mean over pairs, not num_items_in_batch supervised tokens.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            logps = forward_response_logps(model, inputs)
            size = inputs["ref_chosen_logps"].shape[0]
            if logps.shape[0] != 2 * size:
                raise ValueError("Chosen/rejected batch ordering is invalid")
            losses, chosen_reward, rejected_reward = sigmoid_dpo_loss(
                logps[:size], logps[size:], inputs["ref_chosen_logps"], inputs["ref_rejected_logps"], self.beta)
            loss = losses.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite DPO loss")
            outputs = {"logits": torch.stack((chosen_reward, rejected_reward,
                                              logps[:size].detach(), logps[size:].detach()), dim=1)}
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            if prediction_loss_only:
                return loss.detach(), None, None
            # Metrics describe pairs, whereas the model input contains two responses per pair.
            scores = outputs["logits"].detach()
            return loss.detach(), scores, torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)

    return OutcomeDPOTrainer
