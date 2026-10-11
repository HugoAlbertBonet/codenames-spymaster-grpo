"""Render Codenames boards and simulated turns for portfolio visuals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from codenames_env import CodenamesBoard, generate_board


LABEL_COLORS = {
    "BLUE": "#2F80ED",
    "RED": "#EB5757",
    "NEUTRAL": "#D8C3A5",
    "ASSASSIN": "#202124",
}
HIDDEN_COLOR = "#D9DEE5"
TEXT_COLORS = {
    "BLUE": "white",
    "RED": "white",
    "NEUTRAL": "#202124",
    "ASSASSIN": "white",
}


def build_reveal_states(board: CodenamesBoard, guesses: Sequence[str]) -> list[tuple[str, ...]]:
    """Return the initial hidden state followed by one state per valid guess."""
    known_words = {item.word.lower() for item in board.words}
    revealed: list[str] = []
    states = [tuple(revealed)]
    for guess in guesses:
        normalized = guess.lower()
        if normalized not in known_words:
            raise ValueError(f"Guess is not on this board: {guess}")
        if normalized not in revealed:
            revealed.append(normalized)
        states.append(tuple(revealed))
    return states


def render_board(
    board: CodenamesBoard,
    revealed_words: Iterable[str] = (),
    *,
    title: str | None = None,
    output_path: Path | None = None,
    show_labels: bool = False,
) -> Path | None:
    """Render one board frame; unrevealed tiles remain slate-colored."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError("Rendering requires matplotlib; install it with `uv add matplotlib`") from exc

    revealed = {word.lower() for word in revealed_words} | set(board.revealed)
    fig, axis = plt.subplots(figsize=(10, 10))
    axis.set_xlim(0, 4)
    axis.set_ylim(0, 4)
    axis.set_aspect("equal")
    axis.axis("off")

    for index, item in enumerate(board.words):
        row, column = divmod(index, 4)
        x, y = column, 3 - row
        is_revealed = item.word.lower() in revealed
        face_color = LABEL_COLORS[item.label] if is_revealed else HIDDEN_COLOR
        text_color = TEXT_COLORS[item.label] if is_revealed else "#202124"
        axis.add_patch(
            Rectangle(
                (x + 0.04, y + 0.04),
                0.92,
                0.92,
                facecolor=face_color,
                edgecolor="white",
                linewidth=3,
                joinstyle="round",
            )
        )
        label = item.word.upper()
        if show_labels and is_revealed:
            label += f"\n{item.label}"
        axis.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=max(9, min(13, 130 / len(item.word))),
            fontweight="bold",
        )

    if title:
        axis.set_title(title, fontsize=18, fontweight="bold", pad=20)
    fig.text(
        0.5,
        0.02,
        "BLUE  •  RED  •  NEUTRAL  •  ASSASSIN     |     unrevealed tiles are gray",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if output_path is None:
        plt.close(fig)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_gif(frame_paths: Sequence[Path], output_path: Path, duration_ms: int = 900) -> Path:
    """Combine PNG frames into a looping GIF."""
    if not frame_paths:
        raise ValueError("At least one frame is required to make a GIF")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("GIF creation requires Pillow; install it with `uv add pillow`") from exc

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    return output_path


def render_turn_sequence(
    board: CodenamesBoard,
    guesses: Sequence[str],
    *,
    clue: str | None = None,
    number: int | None = None,
    output_dir: Path = Path("results/board_frames"),
    gif_path: Path | None = None,
    show_labels: bool = False,
    title_prefix: str | None = None,
    duration_ms: int = 900,
    completion: str | None = None,
) -> list[Path]:
    """Create an initial frame and one frame after each guessed word."""
    states = build_reveal_states(board, guesses)
    frames: list[Path] = []
    for frame_index, revealed in enumerate(states):
        if frame_index == 0:
            step_title = "Codenames spymaster turn"
        else:
            step_title = f"Guessed: {guesses[frame_index - 1].upper()}"
        if clue:
            step_title = f"CLUE: {clue.upper()}  |  NUMBER: {number or '?'}\n{step_title}"
        elif completion:
            visible_completion = " | ".join(completion.strip().splitlines())
            step_title = f"OUTPUT: {visible_completion}\n{step_title}"
        if title_prefix:
            step_title = f"{title_prefix}\n{step_title}"
        frame_path = output_dir / f"turn_{frame_index:02d}.png"
        render_board(
            board,
            revealed,
            title=step_title,
            output_path=frame_path,
            show_labels=show_labels,
        )
        frames.append(frame_path)
    if gif_path:
        make_gif(frames, gif_path, duration_ms=duration_ms)
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clue", default="portfolio")
    parser.add_argument("--number", type=int, default=2)
    parser.add_argument("--guesses", default="", help="Comma-separated board words to reveal in order")
    parser.add_argument("--output-dir", type=Path, default=Path("results/board_frames"))
    parser.add_argument("--gif", type=Path, default=Path("results/codenames-turn.gif"))
    parser.add_argument("--show-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = generate_board(seed=args.seed)
    guesses = [word.strip() for word in args.guesses.split(",") if word.strip()]
    if not guesses:
        # A no-dependency default: reveal the first two tiles as an illustrative turn.
        guesses = [item.word for item in board.words[:2]]
    frames = render_turn_sequence(
        board,
        guesses,
        clue=args.clue,
        number=args.number,
        output_dir=args.output_dir,
        gif_path=args.gif,
        show_labels=args.show_labels,
    )
    print(f"Saved {len(frames)} frames to {args.output_dir}")
    print(f"Saved GIF to {args.gif}")


if __name__ == "__main__":
    main()
