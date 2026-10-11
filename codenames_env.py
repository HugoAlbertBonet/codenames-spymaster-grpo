"""A small, deterministic Codenames environment for spymaster training.

The environment deliberately keeps the guesser separate from the spymaster.  A
spymaster produces a clue; a fixed embedding-based guesser turns that clue into
guesses; this module scores the resulting turn.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Protocol, Sequence


BLUE = "BLUE"
RED = "RED"
NEUTRAL = "NEUTRAL"
ASSASSIN = "ASSASSIN"
LABELS = (BLUE, RED, NEUTRAL, ASSASSIN)

DEFAULT_VOCABULARY: tuple[str, ...] = (
    "apple", "arm", "bank", "beach", "bolt", "book", "bridge", "button",
    "candle", "car", "castle", "cell", "circle", "cloud", "code", "cold",
    "crown", "dance", "date", "diamond", "doctor", "dragon", "dress", "eagle",
    "engine", "face", "field", "file", "fire", "fish", "foot", "forest",
    "fork", "game", "glass", "gold", "grass", "green", "hand", "head",
    "heart", "horse", "ice", "jam", "key", "king", "lab", "light",
    "line", "lock", "map", "match", "moon", "mouse", "night", "park",
    "paper", "pilot", "pipe", "plane", "point", "port", "ring", "robot",
    "rock", "rose", "round", "school", "screen", "seal", "shoe", "shot",
    "spring", "star", "stick", "stone", "suit", "table", "tag", "tooth",
    "train", "tree", "trip", "truck", "wave", "web", "wind", "yard",
)

REWARD_BY_LABEL = {
    BLUE: 1.0,
    NEUTRAL: -0.3,
    RED: -1.0,
    ASSASSIN: -5.0,
}

FORBIDDEN_CLUE_WORDS = frozenset(label.lower() for label in LABELS)
MIN_CLUE_LENGTH = 2


def validate_abstention(min_similarity: float | None, abstention_mode: str) -> None:
    if min_similarity is not None and (not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("min_similarity must be a finite cosine similarity between -1 and 1")
    if abstention_mode not in {"all", "extra"}:
        raise ValueError("abstention_mode must be 'all' or 'extra'")


def add_abstention_args(parser) -> None:
    """Shared CLI contract for the fixed embedding teammate."""
    parser.add_argument("--min-similarity", type=float, default=None,
                        help="Abstain below this cosine similarity (-1..1); omitted means no abstention")
    parser.add_argument("--abstention-mode", choices=("all", "extra"), default="all",
                        help="Apply the cutoff to every guess or only the NUMBER+1 extra guess")


@dataclass(frozen=True)
class BoardWord:
    word: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Unknown board label: {self.label}")


@dataclass(frozen=True)
class CodenamesBoard:
    words: tuple[BoardWord, ...]
    revealed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.words) != 16:
            raise ValueError("A Codenames board must contain exactly 16 words")
        names = [item.word.lower() for item in self.words]
        if len(set(names)) != len(names):
            raise ValueError("Board words must be unique")
        counts = {label: sum(item.label == label for item in self.words) for label in LABELS}
        expected = {BLUE: 6, RED: 6, NEUTRAL: 3, ASSASSIN: 1}
        if counts != expected:
            raise ValueError(f"Expected labels {expected}, got {counts}")
        if len(set(self.revealed)) != len(self.revealed) or not set(self.revealed) <= set(names):
            raise ValueError("Revealed words must be unique words from this board")
        if any(item.word in self.revealed and item.label == ASSASSIN for item in self.words):
            raise ValueError("A board with a revealed assassin is terminal")
        if not any(item.label == BLUE and item.word not in self.revealed for item in self.words):
            raise ValueError("A board with no remaining BLUE words is terminal")
        if not any(item.label == RED and item.word not in self.revealed for item in self.words):
            raise ValueError("A board with no remaining RED words is terminal")

    @property
    def active_words(self) -> tuple[BoardWord, ...]:
        return tuple(item for item in self.words if item.word not in self.revealed)

    @property
    def by_word(self) -> dict[str, BoardWord]:
        return {item.word.lower(): item for item in self.words}

    def prompt(self) -> str:
        """Serialize a compact description of the board and game rules."""
        words_by_label = {
            label: [item.word for item in self.active_words if item.label == label]
            for label in LABELS
        }
        board_sections = "\n".join(
            f"{label}: {', '.join(words_by_label[label])}"
            for label in LABELS
        )
        if self.revealed:
            board_sections += "\nALREADY GUESSED (unavailable): " + ", ".join(self.revealed)
        state_rule = (
            "- Already guessed words cannot be selected or counted; target only remaining BLUE words.\n"
            if self.revealed else ""
        )
        return (
            "You are the Codenames spymaster. Choose one clue word of at least "
            "two letters "
            "that helps a fixed semantic guesser select as many BLUE words as "
            "possible while avoiding RED, NEUTRAL, and ASSASSIN words.\n\n"
            "TURN RULES:\n"
            "- The board has 16 words. Position does not matter; the words are "
            "grouped below by role tag.\n"
            "- After your clue, the guesser ranks board words by semantic similarity.\n"
            f"{state_rule}"
            "- NUMBER is 1-4. The guesser may make NUMBER + 1 guesses, then stops "
            "at the first non-BLUE word.\n"
            "- BLUE: +1 point. NEUTRAL: -0.3 points and turn ends. RED: -1 point "
            "to you, +1 to the other team, and turn ends. ASSASSIN: -5 points and "
            "the game ends.\n\n"
            "CLUE RULES:\n"
            "- Use exactly one alphabetic word with at least two letters; output "
            "no explanation.\n"
            "- Do not use a board word, a word containing one, or a word contained "
            "by one. An invalid clue gives -2 points and ends the turn.\n"
            "- BLUE, RED, NEUTRAL, and ASSASSIN are metadata tags, not clue words; "
            "never output one of them as the clue.\n\n"
            "BOARD:\n"
            f"{board_sections}\n\n"
            "OUTPUT:\n"
            "CLUE: <single word>\n"
            "NUMBER: <integer from 1 to 4>"
        )


def generate_board(seed: int | None = None, vocabulary: Sequence[str] = DEFAULT_VOCABULARY) -> CodenamesBoard:
    """Generate a reproducible random board with the standard label counts."""

    rng = random.Random(seed)
    normalized = [word.strip().lower() for word in vocabulary]
    if len(normalized) < 16 or len(set(normalized)) < 16:
        raise ValueError("Vocabulary must contain at least 16 unique words")
    selected = rng.sample(list(dict.fromkeys(normalized)), 16)
    labels = [BLUE] * 6 + [RED] * 6 + [NEUTRAL] * 3 + [ASSASSIN]
    rng.shuffle(labels)
    return CodenamesBoard(tuple(BoardWord(word, label) for word, label in zip(selected, labels)))


@dataclass(frozen=True)
class Clue:
    word: str
    number: int


class InvalidClue(ValueError):
    """Raised when a model completion does not follow the game rules."""


_CLUE_RE = re.compile(
    r"^\s*CLUE\s*:\s*([^\s]+)\s*\n\s*NUMBER\s*:\s*([1-4])\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def parse_clue(completion: str) -> Clue:
    """Parse the strict two-line completion format."""
    match = _CLUE_RE.fullmatch(completion)
    if not match:
        raise InvalidClue("Expected CLUE: <single word> and NUMBER: <1-4>")
    word, number = match.groups()
    if not _WORD_RE.fullmatch(word):
        raise InvalidClue("The clue must be one alphabetic word")
    if len(word) < MIN_CLUE_LENGTH:
        raise InvalidClue(f"The clue must contain at least {MIN_CLUE_LENGTH} letters")
    return Clue(word=word.lower(), number=int(number))


def validate_clue(clue: Clue, board: CodenamesBoard) -> None:
    """Apply the board-overlap and anti-shortcut clue rules."""
    clue_word = clue.word.lower()
    if clue_word in FORBIDDEN_CLUE_WORDS:
        raise InvalidClue(f"Clue '{clue.word}' is a reserved board tag")
    for board_item in board.words:
        board_word = board_item.word.lower()
        if clue_word in board_word or board_word in clue_word:
            raise InvalidClue(f"Clue '{clue.word}' overlaps board word '{board_item.word}'")


def parse_and_validate_clue(completion: str, board: CodenamesBoard) -> Clue:
    clue = parse_clue(completion)
    validate_clue(clue, board)
    return clue


class Guesser(Protocol):
    def rank(self, clue: str, words: Sequence[str]) -> list[str]:
        """Return board words from most to least likely for the clue."""


class EmbeddingGuesser:
    """Fixed sentence-transformer guesser used by training and evaluation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        min_similarity: float | None = None,
        abstention_mode: str = "all",
    ) -> None:
        validate_abstention(min_similarity, abstention_mode)
        self.min_similarity = min_similarity
        self.abstention_mode = abstention_mode
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "EmbeddingGuesser requires sentence-transformers; install requirements.txt"
            ) from exc
        # Keeping this on CPU by default leaves the training GPU's VRAM for the
        # policy model and GRPO's sampled completions.
        model_kwargs = {} if device is None else {"device": device}
        self.model = SentenceTransformer(model_name, **model_kwargs)
        self._word_embedding_cache: dict[tuple[str, ...], object] = {}
        self._clue_embedding_cache: dict[str, object] = {}

    def rank(self, clue: str, words: Sequence[str]) -> list[str]:
        if not words:
            return []
        similarities = self.similarities(clue, words)
        return [
            word
            for _, word in sorted(
                ((similarities[word], word) for word in words),
                key=lambda pair: -pair[0],
            )
        ]

    def similarities(self, clue: str, words: Sequence[str]) -> dict[str, float]:
        """Return normalized cosine similarity for each board word."""
        if not words:
            return {}
        word_key = tuple(words)
        if word_key not in self._word_embedding_cache:
            # In GRPO, the same board is scored for every sampled completion.
            # Cache its word embeddings instead of recomputing them per clue.
            embeddings = self.model.encode(list(words), normalize_embeddings=True)
            self._word_embedding_cache[word_key] = embeddings
        if clue not in self._clue_embedding_cache:
            self._clue_embedding_cache[clue] = self.model.encode(
                [clue], normalize_embeddings=True
            )[0]
        word_embeddings = self._word_embedding_cache[word_key]
        clue_embedding = self._clue_embedding_cache[clue]
        values = word_embeddings @ clue_embedding
        return {word: float(value) for word, value in zip(words, values)}


class LLMGuesser:
    """Frozen language-model teammate that ranks board words from a clue.

    The model receives only the clue and board words, never hidden role labels.
    It is intended for evaluation first because it is substantially slower than
    ``EmbeddingGuesser`` and is not used by training by default.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 64,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LLMGuesser requires torch and transformers; install requirements.txt"
            ) from exc

        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        use_bf16 = bool(self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._rank_cache: dict[tuple[str, tuple[str, ...]], list[str]] = {}

    @staticmethod
    def parse_ranking(text: str, words: Sequence[str]) -> list[str]:
        """Extract a complete, duplicate-free ranking from model output."""
        occurrences: list[tuple[int, str]] = []
        for word in words:
            match = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE)
            if match:
                occurrences.append((match.start(), word))
        occurrences.sort(key=lambda item: item[0])
        ranked: list[str] = []
        seen: set[str] = set()
        for _, word in occurrences:
            if word.lower() not in seen:
                ranked.append(word)
                seen.add(word.lower())
        # A malformed/incomplete answer still becomes a valid permutation; the
        # omitted words retain board order instead of being silently discarded.
        ranked.extend(word for word in words if word.lower() not in seen)
        return ranked

    def _prompt(self, clue: str, words: Sequence[str]) -> str:
        board = "\n".join(f"- {word}" for word in words)
        return (
            "You are the guessing teammate in Codenames. The spymaster gave "
            "one clue. Rank every board word from most related to least related "
            "to that clue. You do not know the hidden team labels.\n\n"
            f"CLUE: {clue}\n"
            "BOARD WORDS:\n"
            f"{board}\n\n"
            "Return only the board words in ranked order, one per line. Do not "
            "add explanations, guesses, labels, or new words."
        )

    def rank(self, clue: str, words: Sequence[str]) -> list[str]:
        if not words:
            return []
        key = (clue.lower(), tuple(words))
        if key in self._rank_cache:
            return list(self._rank_cache[key])

        import torch

        prompt = self._prompt(clue, words)
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[-1]
        output_text = self.tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
        )
        ranked = self.parse_ranking(output_text, words)
        self._rank_cache[key] = ranked
        return list(ranked)


@dataclass(frozen=True)
class TurnResult:
    completion: str
    clue: Clue | None
    guesses: tuple[str, ...]
    guessed_labels: tuple[str, ...]
    reward: float
    invalid: bool
    invalid_reason: str | None
    abstained: bool = False
    stop_reason: str = "invalid"
    guess_similarities: tuple[float, ...] = ()


def simulate_turn(completion: str, board: CodenamesBoard, guesser: Guesser) -> TurnResult:
    """Score one spymaster completion, stopping at the first non-BLUE guess."""
    try:
        clue = parse_and_validate_clue(completion, board)
    except InvalidClue as exc:
        return TurnResult(completion, None, (), (), -2.0, True, str(exc))

    remaining = [item.word for item in board.active_words]
    ranked = guesser.rank(clue.word, remaining)
    cutoff = getattr(guesser, "min_similarity", None)
    mode = getattr(guesser, "abstention_mode", "all")
    validate_abstention(cutoff, mode)
    similarity_fn = getattr(guesser, "similarities", None)
    if cutoff is not None and similarity_fn is None:
        raise ValueError("Similarity abstention requires a guesser with cosine scores")
    scores = similarity_fn(clue.word, remaining) if similarity_fn else {}
    max_guesses = min(clue.number + 1, len(ranked))
    guesses: list[str] = []
    labels: list[str] = []
    reward = 0.0
    abstained = False
    stop_reason = "guess_limit"
    guess_scores: list[float] = []
    lookup = board.by_word
    for index, word in enumerate(ranked[:max_guesses]):
        if cutoff is not None and (mode == "all" or index >= clue.number) and scores[word] < cutoff:
            abstained = True
            stop_reason = "similarity_below_threshold"
            break
        label = lookup[word.lower()].label
        guesses.append(word)
        labels.append(label)
        if scores:
            guess_scores.append(scores[word])
        reward += REWARD_BY_LABEL[label]
        if label != BLUE:
            stop_reason = label.lower()
            break
    return TurnResult(completion, clue, tuple(guesses), tuple(labels), reward, False, None,
                      abstained, stop_reason, tuple(guess_scores))


class OrderedGuesser:
    """Small deterministic guesser useful for tests and examples."""

    def __init__(self, ordered_words: Sequence[str]) -> None:
        self.ordered_words = list(ordered_words)

    def rank(self, clue: str, words: Sequence[str]) -> list[str]:
        allowed = {word.lower(): word for word in words}
        ranked = [word for word in self.ordered_words if word.lower() in allowed]
        ranked.extend(word for word in words if word not in ranked)
        return ranked


__all__ = [
    "ASSASSIN", "BLUE", "CodenamesBoard", "Clue", "DEFAULT_VOCABULARY",
    "EmbeddingGuesser", "InvalidClue", "LLMGuesser", "NEUTRAL", "OrderedGuesser", "RED",
    "FORBIDDEN_CLUE_WORDS", "MIN_CLUE_LENGTH", "TurnResult", "generate_board", "parse_and_validate_clue",
    "parse_clue", "simulate_turn", "validate_clue",
    "add_abstention_args", "validate_abstention",
]
