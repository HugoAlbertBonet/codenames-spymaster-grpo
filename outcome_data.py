"""Shared, resumable clue sampling and board-leakage checks for outcome datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from codenames_env import BoardWord, CodenamesBoard, InvalidClue, parse_clue
from evaluate import load_model_and_tokenizer
from train_grpo import build_training_examples, format_prompt_for_model, load_vocabulary_file, validate_initial_adapter


def board_from_row(row: dict) -> CodenamesBoard:
    words, labels = row["board_words"], row["board_labels"]
    if len(words) != len(labels):
        raise ValueError("Mismatched board columns")
    return CodenamesBoard(tuple(BoardWord(w, label) for w, label in zip(words, labels)),
                          tuple(row.get("board_revealed", ())))


def board_key(row: dict) -> str:
    # Exclude permutations AND alternative partial states of the same labeled board.
    board = board_from_row(row)
    payload = sorted((tile.word, tile.label) for tile in board.words)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_artifact_hashes(model_name: str) -> dict[str, str]:
    """Pin cached teammate weights/tokenization, not only a mutable Hub model name."""
    from huggingface_hub import try_to_load_from_cache
    hashes = {}
    for name in ("model.safetensors", "pytorch_model.bin", "config.json", "modules.json",
                 "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        if Path(model_name).is_dir():
            path = Path(model_name) / name
        else:
            cached = try_to_load_from_cache(model_name, name)
            path = Path(cached) if isinstance(cached, str) else None
        if path is not None and path.is_file():
            hashes[name] = file_hash(path)
    if not ({"model.safetensors", "pytorch_model.bin"} & hashes.keys()):
        raise FileNotFoundError("Cannot fingerprint the cached embedding weights")
    return hashes


def prime_embeddings(guesser, rows: Sequence[dict]) -> None:
    """Batch the exact same embeddings to avoid thousands of tiny CPU forwards."""
    boards = [board_from_row(row) for row in rows]
    words = sorted({tile.word for board in boards for tile in board.active_words})
    clues = set()
    for row in rows:
        for completion in row["completions"]:
            try:
                clues.add(parse_clue(completion).word)
            except InvalidClue:
                pass
    vectors = guesser.model.encode(words, normalize_embeddings=True, batch_size=256)
    index = {word: i for i, word in enumerate(words)}
    for board in boards:
        names = tuple(tile.word for tile in board.active_words)
        guesser._word_embedding_cache[names] = vectors[[index[word] for word in names]]
    ordered_clues = sorted(clues)
    if ordered_clues:
        clue_vectors = guesser.model.encode(ordered_clues, normalize_embeddings=True, batch_size=256)
        guesser._clue_embedding_cache.update(zip(ordered_clues, clue_vectors))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def known_boards(extra_paths: Sequence[Path] = ()) -> tuple[set[str], dict[str, str]]:
    """Read self-contained prior evaluations and SFT boards, without model imports."""
    paths = sorted(set(Path("results").glob("evaluation*.csv")) |
                   set(Path("results").glob("abstention*/boards.jsonl")) |
                   set(Path("data").glob("dpo*/boards.jsonl")) |
                   set(Path("data/sft").glob("*.jsonl")) | set(extra_paths))
    keys: set[str] = set()
    sources = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        sources[str(path)] = file_hash(path)
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows = [{key: json.loads(row[key]) for key in
                     ("board_words", "board_labels", "board_revealed") if row.get(key)}
                    for row in rows if row.get("board_words") and row.get("board_labels")]
        else:
            rows = read_jsonl(path)
        keys.update(board_key(row) for row in rows)
    return keys, sources


def fresh_boards(count: int, seed: int, vocabulary_file: Path, max_revealed: int,
                 suite: str, excluded: set[str]) -> list[dict]:
    if count < 1:
        raise ValueError("Board count must be positive")
    vocabulary = load_vocabulary_file(vocabulary_file)
    rows = []
    next_seed = seed
    while len(rows) < count:
        if next_seed - seed > count * 100:
            raise ValueError("Cannot find enough unique fresh boards")
        example = build_training_examples(1, next_seed, vocabulary, max_revealed)[0]
        row = example.as_dict() | {"board_seed": next_seed, "suite": suite,
                                   "board_index": len(rows)}
        key = board_key(row)
        if key not in excluded:
            excluded.add(key)
            rows.append(row | {"board_key": key})
        next_seed += 1
    return rows


def prepare_run(output_dir: Path, config: dict, make_boards, *, resume: bool) -> tuple[list[dict], dict]:
    """Freeze the experiment before sampling; don't overwrite an existing run."""
    manifest_path = output_dir / "manifest.json"
    config = json.loads(json.dumps(config, default=str))
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Resume settings or input hashes differ; use a new output directory")
        if file_hash(output_dir / "boards.jsonl") != manifest["boards_sha256"]:
            raise ValueError("Saved board plan changed")
        return read_jsonl(output_dir / "boards.jsonl"), manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume or a new directory")
    rows, excluded_sources = make_boards()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "boards.jsonl", rows)
    manifest = {"schema_version": 1, "config": config,
                "boards_sha256": file_hash(output_dir / "boards.jsonl"),
                "excluded_sources": excluded_sources}
    write_json(manifest_path, manifest)
    return rows, manifest


def sampling_config(args) -> dict:
    validate_initial_adapter(args.adapter_dir, args.base_model)
    weights = sorted(args.adapter_dir.glob("adapter_model.*"))
    if not weights:
        raise FileNotFoundError("No SFT adapter weights found")
    config = {"base_model": args.base_model, "adapter_dir": str(args.adapter_dir.resolve()),
            "adapter_hashes": {p.name: file_hash(p) for p in weights + [args.adapter_dir / "adapter_config.json"]},
            "max_new_tokens": args.max_new_tokens, "samples_per_board": args.samples_per_board,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "repetition_penalty": 1.1, "sampling_seed": args.seed}
    if args.samples_per_board > 1:
        config["sampling_protocol"] = "greedy_plus_sampled_v1_up_to_4_boards_28_sequences"
    return config


def add_sampling_args(parser) -> None:
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/codenames-sft-1p5b"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit slow policy generation without CUDA")


def collect_candidates(rows: Sequence[dict], output_dir: Path, args) -> list[dict]:
    """One greedy clue plus sampled alternatives; save each complete board immediately."""
    path = output_dir / "candidates.jsonl"
    completed = read_jsonl(path) if path.exists() else []
    planned = {row["board_key"]: row for row in rows}
    done = set()
    for row in completed:
        key = row["board_key"]
        expected = planned.get(key)
        if key in done or expected is None or any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Candidate log contains duplicate, unknown or changed boards")
        if len(row["completions"]) != args.samples_per_board:
            raise ValueError("Candidate count differs from run settings")
        done.add(key)
    pending = [row for row in rows if row["board_key"] not in done]
    if not pending:
        return completed
    import torch
    from transformers import GenerationConfig, set_seed
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable here. Run in your GPU terminal, or explicitly use --allow-cpu")
    torch.set_num_threads(4)
    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.adapter_dir)
    with path.open("a", encoding="utf-8") as handle:
        if args.samples_per_board == 1:
            # Left-padded greedy batches keep this confirmation practical on a laptop GPU.
            # Both teammates still see the very same completion for each board.
            for offset in range(0, len(pending), 16):
                batch_rows = pending[offset:offset + 16]
                prompts = [format_prompt_for_model(row["prompt"], tokenizer) for row in batch_rows]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                length = inputs["input_ids"].shape[-1]
                config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None, repetition_penalty=1.1,
                                          eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                                          pad_token_id=tokenizer.pad_token_id)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, generation_config=config)
                for row, ids in zip(batch_rows, outputs):
                    completion = tokenizer.decode(ids[length:], skip_special_tokens=True).strip()
                    saved = row | {"completions": [completion]}
                    handle.write(json.dumps(saved) + "\n")
                    completed.append(saved)
                handle.flush()
                print(f"Sampled {len(completed)}/{len(rows)} boards", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return completed
        batch_size = max(1, min(4, 28 // (args.samples_per_board - 1)))
        for offset in range(0, len(pending), batch_size):
            batch_rows = pending[offset:offset + batch_size]
            set_seed(args.seed + batch_rows[0]["board_seed"])
            prompts = [format_prompt_for_model(row["prompt"], tokenizer) for row in batch_rows]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            length = inputs["input_ids"].shape[-1]
            def generate(sample: bool, returns: int) -> list[str]:
                config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=sample,
                                          temperature=args.temperature if sample else None,
                                          top_p=args.top_p if sample else None, top_k=args.top_k if sample else None,
                                          repetition_penalty=1.1, num_return_sequences=returns,
                                          eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                                          pad_token_id=tokenizer.pad_token_id)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, generation_config=config)
                return [tokenizer.decode(ids[length:], skip_special_tokens=True).strip() for ids in outputs]
            completions = [[text] for text in generate(False, 1)]
            remaining = args.samples_per_board - 1
            while remaining:
                returns = min(remaining, 28 // len(batch_rows))
                sampled = generate(True, returns)
                for i in range(len(batch_rows)):
                    completions[i].extend(sampled[i * returns:(i + 1) * returns])
                remaining -= returns
            for row, texts in zip(batch_rows, completions):
                saved = row | {"completions": texts}
                handle.write(json.dumps(saved) + "\n")
                completed.append(saved)
            handle.flush()
            if len(completed) % 20 == 0 or len(completed) == len(rows):
                print(f"Sampled {len(completed)}/{len(rows)} boards", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return completed
