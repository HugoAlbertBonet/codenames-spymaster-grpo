"""Visualize which board words influence a generated Codenames clue.

This tool uses ordinary Transformers eager attention rather than the optimized
Unsloth/vLLM generation path. It captures the attention used to predict each
generated token, aggregates it over the token spans of board words, and writes:

* a board heatmap with attention mass per word;
* a heatmap for the most board-focused layer/head combinations;
* a JSON trace with the numerical values;
* optionally, an occlusion chart measuring how much each word changes the
  likelihood of the original completion.

Attention is evidence of information routing, not a proof of reasoning. The
optional occlusion analysis is included as a more causal companion signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from codenames_env import BoardWord, CodenamesBoard
from evaluate import format_prompt_for_model, load_model_and_tokenizer
from render_evaluation import board_from_evaluation_row, read_evaluation_rows, select_rows


@dataclass(frozen=True)
class AttentionTrace:
    """Attention collected while greedily predicting one completion."""

    formatted_prompt: str
    # The attention word axis follows the grouped order used by board.prompt().
    prompt_words: tuple[str, ...]
    completion: str
    generated_ids: tuple[int, ...]
    generated_token_text: tuple[str, ...]
    # Shape: generation step x layer x head x board word.
    attention: Any


def find_word_token_spans(
    formatted_prompt: str,
    tokenizer: Any,
    words: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Find tokenizer positions belonging to each board word.

    The prompt contains each board word once in its role-tagged board section.
    Character offsets are used instead of tokenizing words independently so
    multi-token words are handled correctly.
    """
    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("The attention visualizer requires a fast tokenizer with offset mappings")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]

    result: list[tuple[int, ...]] = []
    for word in words:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        matches = list(pattern.finditer(formatted_prompt))
        if not matches:
            raise ValueError(f"Could not find board word {word!r} in the formatted prompt")
        start, end = matches[-1].span()
        token_indices = tuple(
            index
            for index, offset in enumerate(offsets)
            if len(offset) == 2 and offset[1] > start and offset[0] < end
        )
        if not token_indices:
            raise ValueError(f"No tokenizer span found for board word {word!r}")
        result.append(token_indices)
    return tuple(result)


def aggregate_last_token_attention(
    attentions: Sequence[Any],
    word_spans: Sequence[Sequence[int]],
    prompt_length: int,
) -> Any:
    """Aggregate each layer/head's last-query attention over board words.

    Returns a tensor with shape ``(layers, heads, words)``. Attention directed
    to non-board prompt tokens is intentionally not included in the word
    scores, but the total board mass is still available by summing the result.
    """
    import torch

    layer_scores = []
    for layer_attention in attentions:
        if layer_attention is None:
            raise RuntimeError("The model returned an empty attention tensor")
        # Standard eager self-attention shape: batch x heads x query x key.
        query_attention = layer_attention[0, :, -1, :prompt_length]
        word_scores = []
        for span in word_spans:
            indices = torch.tensor(span, device=query_attention.device)
            word_scores.append(query_attention.index_select(-1, indices).sum(dim=-1))
        layer_scores.append(torch.stack(word_scores, dim=-1))
    return torch.stack(layer_scores, dim=0)


def generate_with_attention(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    focus: str = "clue",
) -> AttentionTrace:
    """Greedily generate while recording attention for each prediction step.

    ``focus='clue'`` stops after the first generated newline, so NUMBER tokens
    do not dominate the clue map. The recorded attention is the context used
    to predict each next token, which is the useful causal direction for a
    decoder-only model.
    """
    import torch

    formatted_prompt = format_prompt_for_model(prompt, tokenizer)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = input_ids.shape[-1]
    words = _prompt_words(prompt)
    word_spans = find_word_token_spans(formatted_prompt, tokenizer, words)

    attention_steps = []
    generated_ids: list[int] = []
    generated_text: list[str] = []
    eos_ids = {token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None}

    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    try:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
            if outputs.attentions is None:
                raise RuntimeError(
                    "The model did not return attentions. Load it with eager attention "
                    "and avoid the Unsloth/vLLM fast-inference path."
                )
            step_attention = aggregate_last_token_attention(
                outputs.attentions,
                word_spans,
                prompt_length,
            )
            next_id = int(outputs.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            token_text = tokenizer.decode([next_id], skip_special_tokens=True)
            generated_text.append(token_text)
            candidate_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Include the token that completes the clue line, then stop before
            # attention from NUMBER and any explanation can enter the map.
            attention_steps.append(step_attention.detach().float().cpu())
            if next_id in eos_ids or (focus == "clue" and "\n" in candidate_text):
                break

            next_tensor = torch.tensor([[next_id]], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat((input_ids, next_tensor), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)),
                dim=-1,
            )
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache

    if not attention_steps:
        raise RuntimeError("The model generated no tokens")
    return AttentionTrace(
        formatted_prompt=formatted_prompt,
        prompt_words=words,
        completion=tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        generated_ids=tuple(generated_ids),
        generated_token_text=tuple(generated_text),
        attention=torch.stack(attention_steps, dim=0),
    )


def _prompt_words(prompt: str) -> tuple[str, ...]:
    """Extract board words from the prompt's four role-tagged sections."""
    board_marker = prompt.split("BOARD:\n", 1)
    if len(board_marker) != 2:
        raise ValueError("Prompt does not contain the expected BOARD section")
    board_text = board_marker[1].split("\n\nOUTPUT:", 1)[0]
    words: list[str] = []
    for line in board_text.splitlines():
        if ":" not in line:
            continue
        _, values = line.split(":", 1)
        words.extend(value.strip() for value in values.split(",") if value.strip())
    if len(words) != 16:
        raise ValueError(f"Expected 16 board words in prompt, found {len(words)}")
    return tuple(words)


def average_word_attention(trace: AttentionTrace) -> Any:
    """Average attention over generation steps, layers, and heads."""
    return trace.attention.mean(dim=(0, 1, 2))


def reorder_scores_to_board(
    scores: Sequence[float],
    prompt_words: Sequence[str],
    board: CodenamesBoard,
) -> list[float]:
    """Map grouped-prompt scores back to the board's tile order."""
    positions = {word.lower(): index for index, word in enumerate(prompt_words)}
    try:
        return [float(scores[positions[item.word.lower()]]) for item in board.words]
    except KeyError as exc:
        raise ValueError(f"Board word {exc.args[0]!r} is missing from the prompt attention axis") from exc


def top_attention_heads(trace: AttentionTrace, count: int = 12) -> list[tuple[int, int, float]]:
    """Return the most board-focused layer/head pairs."""
    scores = trace.attention.mean(dim=(0, 3))
    layers, heads = scores.shape
    flattened = scores.reshape(-1)
    count = min(count, flattened.numel())
    indices = flattened.topk(count).indices.tolist()
    return [
        (index // heads, index % heads, float(flattened[index]))
        for index in indices
    ]


def replace_board_word(prompt: str, word: str, replacement: str = "[REMOVED]") -> str:
    """Create an occluded prompt by replacing one board-word occurrence."""
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    updated, count = pattern.subn(replacement, prompt, count=1)
    if count != 1:
        raise ValueError(f"Could not occlude board word {word!r}")
    return updated


def sequence_log_probability(
    model: Any,
    tokenizer: Any,
    device: Any,
    formatted_prompt: str,
    generated_ids: Sequence[int],
) -> float:
    """Score a fixed completion token sequence under a prompt."""
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    completion_ids = torch.tensor([list(generated_ids)], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat((prompt_ids, completion_ids), dim=-1)
    mask = torch.ones_like(full_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
    start = prompt_ids.shape[-1] - 1
    logits = outputs.logits[:, start : start + len(generated_ids), :]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = completion_ids.unsqueeze(-1)
    return float(log_probs.gather(-1, targets).sum().item())


def run_occlusion_analysis(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    words: Sequence[str],
    generated_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Measure the log-probability drop caused by removing each board word."""
    original_formatted = format_prompt_for_model(prompt, tokenizer)
    baseline = sequence_log_probability(model, tokenizer, device, original_formatted, generated_ids)
    results = []
    for word in words:
        occluded_prompt = replace_board_word(prompt, word)
        occluded_formatted = format_prompt_for_model(occluded_prompt, tokenizer)
        occluded_score = sequence_log_probability(
            model,
            tokenizer,
            device,
            occluded_formatted,
            generated_ids,
        )
        results.append({
            "word": word,
            "baseline_log_probability": baseline,
            "occluded_log_probability": occluded_score,
            "log_probability_drop": baseline - occluded_score,
        })
    return results


def render_attention_board(
    board: CodenamesBoard,
    scores: Sequence[float],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Render attention mass as a Codenames-style 4x4 board."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    total = sum(max(float(score), 0.0) for score in scores) or 1.0
    proportions = [max(float(score), 0.0) / total for score in scores]
    figure, axis = plt.subplots(figsize=(11, 10))
    axis.set_xlim(0, 4)
    axis.set_ylim(0, 4)
    axis.set_aspect("equal")
    axis.axis("off")
    normalize = Normalize(vmin=0.0, vmax=max(proportions) or 1.0)
    colormap = plt.get_cmap("YlOrRd")
    label_colors = {"BLUE": "#2F80ED", "RED": "#EB5757", "NEUTRAL": "#9B8065", "ASSASSIN": "#202124"}

    for index, (item, proportion) in enumerate(zip(board.words, proportions)):
        row, column = divmod(index, 4)
        x, y = column, 3 - row
        axis.add_patch(
            Rectangle(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                facecolor=colormap(normalize(proportion)),
                edgecolor=label_colors[item.label], linewidth=4,
            )
        )
        axis.text(x + 0.5, y + 0.57, item.word.upper(), ha="center", va="center", fontsize=12, fontweight="bold")
        axis.text(x + 0.5, y + 0.28, f"{item.label} • {100 * proportion:.1f}%", ha="center", va="center", fontsize=8)
    axis.set_title(title, fontsize=16, fontweight="bold", pad=18)
    figure.text(0.5, 0.02, "Tile border = hidden role • fill = attention share among board words", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_head_heatmap(
    board: CodenamesBoard,
    trace: AttentionTrace,
    output_path: Path,
    *,
    count: int = 12,
) -> None:
    """Render word attention for the most board-focused attention heads."""
    import matplotlib.pyplot as plt

    selected = top_attention_heads(trace, count=count)
    rows = []
    labels = []
    board_positions = [trace.prompt_words.index(item.word) for item in board.words]
    for layer, head, _ in selected:
        head_scores = trace.attention[:, layer, head, :].mean(dim=0)
        rows.append(head_scores[board_positions].tolist())
        labels.append(f"L{layer} H{head}")
    figure_height = max(4.0, 0.42 * len(rows) + 2.0)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    image = axis.imshow(rows, aspect="auto", cmap="YlOrRd")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(range(len(board.words)), [f"{item.word}\n{item.label}" for item in board.words], rotation=45, ha="right")
    axis.set_title("Most board-focused attention heads", fontweight="bold")
    axis.set_xlabel("Board word")
    figure.colorbar(image, ax=axis, label="Attention mass")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_occlusion_chart(
    board: CodenamesBoard,
    results: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Render the change in completion likelihood after word removal."""
    import matplotlib.pyplot as plt

    ordered = sorted(results, key=lambda item: float(item["log_probability_drop"]), reverse=True)
    labels = [str(item["word"]) for item in ordered]
    values = [float(item["log_probability_drop"]) for item in ordered]
    colors = {item.word: {"BLUE": "#2F80ED", "RED": "#EB5757", "NEUTRAL": "#9B8065", "ASSASSIN": "#202124"}[item.label] for item in board.words}
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels[::-1], values[::-1], color=[colors[label] for label in labels[::-1]])
    axis.axvline(0.0, color="#202124", linewidth=1)
    axis.set_xlabel("Original completion log-probability − occluded log-probability")
    axis.set_title("Causal word-occlusion signal", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_trace_json(
    path: Path,
    board: CodenamesBoard,
    trace: AttentionTrace,
    occlusion: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Write numerical trace data for later analysis."""
    payload = {
        "completion": trace.completion,
        "generated_ids": list(trace.generated_ids),
        "generated_token_text": list(trace.generated_token_text),
        "attention_word_order": list(trace.prompt_words),
        "board_words": [item.word for item in board.words],
        "board_labels": [item.label for item in board.words],
        "attention_shape": list(trace.attention.shape),
        "average_word_attention": average_word_attention(trace).tolist(),
        "board_order_average_word_attention": reorder_scores_to_board(
            average_word_attention(trace).tolist(), trace.prompt_words, board
        ),
        "top_heads": [
            {"layer": layer, "head": head, "board_attention": score}
            for layer, head, score in top_attention_heads(trace)
        ],
        "head_word_attention": trace.attention.mean(dim=0).tolist(),
    }
    if occlusion is not None:
        payload["occlusion"] = list(occlusion)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_row(
    row: Mapping[str, str],
    model: Any,
    tokenizer: Any,
    device: Any,
    output_dir: Path,
    *,
    max_new_tokens: int,
    focus: str,
    do_occlusion: bool,
) -> None:
    board = board_from_evaluation_row(row)
    prompt = board.prompt()
    trace = generate_with_attention(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=max_new_tokens,
        focus=focus,
    )
    prefix = output_dir / f"board_{int(row['board_index']):04d}_{row['model']}"
    render_attention_board(
        board,
        reorder_scores_to_board(average_word_attention(trace).tolist(), trace.prompt_words, board),
        prefix.with_name(prefix.name + "_attention.png"),
        title=f"{row['model'].upper()} board {row['board_index']} • {trace.completion!r}",
    )
    render_head_heatmap(board, trace, prefix.with_name(prefix.name + "_heads.png"))

    occlusion = None
    if do_occlusion:
        occlusion = run_occlusion_analysis(
            model,
            tokenizer,
            device,
            prompt,
            [item.word for item in board.words],
            trace.generated_ids,
        )
        render_occlusion_chart(board, occlusion, prefix.with_name(prefix.name + "_occlusion.png"))
    _write_trace_json(prefix.with_name(prefix.name + ".json"), board, trace, occlusion)
    print(f"Saved attention outputs for board {row['board_index']} ({row['model']}) to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-file", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/codenames-grpo"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--model", choices=("base", "trained", "both"), default="trained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--focus", choices=("clue", "all"), default="clue")
    parser.add_argument("--occlusion", action="store_true", help="Also run the slower word-occlusion analysis")
    parser.add_argument("--output-dir", type=Path, default=Path("results/attention"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_evaluation_rows(args.evaluation_file)
    selected = select_rows(rows, args.count, args.model, args.seed)
    loaded_models: dict[str, tuple[Any, Any, Any]] = {}
    for row in selected:
        label = row["model"]
        if label not in loaded_models:
            loaded_models[label] = load_model_and_tokenizer(
                args.base_model,
                args.adapter_dir if label == "trained" else None,
                attn_implementation="eager",
            )
        process_row(
            row,
            *loaded_models[label],
            args.output_dir,
            max_new_tokens=args.max_new_tokens,
            focus=args.focus,
            do_occlusion=args.occlusion,
        )


if __name__ == "__main__":
    main()
