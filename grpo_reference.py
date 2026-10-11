"""A frozen SFT adapter reference without allocating a second base model.

Import torch/PEFT only after Unsloth has applied its patches.
"""

from contextlib import contextmanager
import hashlib
from functools import wraps


def adapter_digest(model, name):
    import torch
    from peft import get_peft_model_state_dict
    digest = hashlib.sha256()
    for key, tensor in sorted(get_peft_model_state_dict(model, adapter_name=name).items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def frozen_adapter(model, name):
    previous = model.active_adapter
    trainable = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.set_adapter(name, inference_mode=True)
        yield
    finally:
        model.set_adapter(previous)
        for parameter, requires_grad in trainable:
            parameter.requires_grad_(requires_grad)


def make_sft_reference(model, adapter_path):
    import torch

    name = "sft_reference"
    if name in model.peft_config:
        raise ValueError("Reference adapter already exists")
    policy_name = model.active_adapter
    model.load_adapter(str(adapter_path), adapter_name=name, is_trainable=False)
    model.set_adapter(policy_name)
    model.set_requires_grad(name, False)
    expected = adapter_digest(model, name)
    if adapter_digest(model, policy_name) != expected:
        raise ValueError("Policy and SFT reference weights differ before training")

    original_save = model.save_pretrained

    @wraps(original_save)
    def save_policy(*args, **kwargs):
        # Transformers' resume loader treats ANY named-adapter subdirectory as
        # a multi-adapter checkpoint, and then skips the default adapter at its
        # root. Save only the policy; reconstruct the pinned SFT reference from
        # its original artifact on resume, before loading policy/optimizer state.
        kwargs.setdefault("selected_adapters", [policy_name])
        return original_save(*args, **kwargs)

    model.save_pretrained = save_policy

    class FrozenSFTReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Do not register the policy as a child: eval()/to()/requires_grad_()
            # on this view must never change the trainable policy.
            object.__setattr__(self, "_policy", model)
            self.config = model.config
            self.forward_calls = 0

        def get_output_embeddings(self):
            return self._policy.get_output_embeddings()

        def forward(self, *args, **kwargs):
            self.forward_calls += 1
            with torch.no_grad(), frozen_adapter(self._policy, name):
                return self._policy(*args, **kwargs)

        def audit(self):
            current = adapter_digest(self._policy, name)
            if current != expected:
                raise RuntimeError("Frozen SFT reference weights changed")
            if any(parameter.requires_grad for key, parameter in self._policy.named_parameters()
                   if f".{name}." in key):
                raise RuntimeError("SFT reference unexpectedly became trainable")
            return {"reference_adapter_sha256": current, "forward_calls": self.forward_calls}

    return FrozenSFTReference()
