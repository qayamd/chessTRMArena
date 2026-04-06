#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#head-to-head arena for comparing model checkpoints
#1-ply engine: eval all legal next positions, pick best expected value (always white's perspective)

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import tokenize


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> nn.Module:
    #reconstruct and load a TRM or Transformer from a train.py checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    if not saved_args:
        import warnings
        warnings.warn(f"Checkpoint {ckpt_path} has no saved args; using defaults -- "
                       "architecture may not match saved weights.")

    model_type = saved_args.get("model", "trm")
    embedding_dim = saved_args.get("embedding_dim", 64)
    num_heads = saved_args.get("num_heads", 8)
    widening_factor = saved_args.get("widening_factor", 4)
    num_return_buckets = saved_args.get("num_return_buckets", 128)
    apply_post_ln = saved_args.get("apply_post_ln", True)
    use_causal_mask = saved_args.get("use_causal_mask", False)
    seed = saved_args.get("seed", 42)

    from tokenizer import SEQUENCE_LENGTH
    shared = dict(
        vocab_size=32,
        output_size=num_return_buckets,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        widening_factor=widening_factor,
        use_causal_mask=use_causal_mask,
        apply_post_ln=apply_post_ln,
        max_sequence_length=SEQUENCE_LENGTH,
        seed=seed,
    )

    if model_type == "trm":
        from TRM import TRM, TRMConfig
        num_layers = saved_args.get("num_layers", 2)
        n_recurrence = saved_args.get("n_recurrence", 8)
        cfg = TRMConfig(**shared, num_layers=num_layers, n_recurrence=n_recurrence)
        model = TRM(cfg)
    else:
        from transformer import TransformerDecoder, TransformerConfig
        num_layers = saved_args.get("num_layers", 4)
        cfg = TransformerConfig(**shared, num_layers=num_layers)
        model = TransformerDecoder(cfg)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class Engine:
    #base class for chess engines
    def play(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class RandomEngine(Engine):
    #picks a uniformly random legal move
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def play(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return self._rng.choice(moves)


class ModelEngine(Engine):
    #1-ply engine: batches all legal next positions, picks best expected value
    #white maximizes, black minimizes; value is always from white's perspective

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_return_buckets: int = 128,
        temperature: float = 1.0,
    ) -> None:
        self._model = model
        self._device = device
        self._num_buckets = num_return_buckets
        self._temperature = temperature
        # bucket values linearly from 0 (white loses) to 1 (white wins)
        self._bucket_values = torch.linspace(
            0.0, 1.0, num_return_buckets, device=device
        )

    @torch.no_grad()
    def play(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves available")
        if len(moves) == 1:
            return moves[0]

        # build a batch of resulting FEN strings
        fens: list[str] = []
        for move in moves:
            board.push(move)
            fens.append(board.fen())
            board.pop()

        # tokenize and stack into a single batch [M, T]
        tokens = torch.stack([tokenize(fen) for fen in fens])  # [M, T] int32
        tokens = tokens.to(self._device).long()

        # forward pass -- use last-position output for board evaluation
        logits = self._model(tokens)            # [M, T, V]
        last_logits = logits[:, -1, :]          # [M, V]

        if self._temperature != 1.0:
            # re-normalize after temperature scaling (last_logits are log-softmax)
            last_logits = last_logits / self._temperature
            last_logits = last_logits - torch.logsumexp(last_logits, dim=-1, keepdim=True)
        probs = last_logits.exp()               # valid probability distribution

        # expected win-probability for white
        ev = (probs * self._bucket_values).sum(dim=-1)   # [M]

        if board.turn == chess.WHITE:
            best_idx = int(ev.argmax())
        else:
            best_idx = int(ev.argmin())

        return moves[best_idx]


RESULT_SCORES = {"1-0": (1, 0), "0-1": (0, 1), "1/2-1/2": (0.5, 0.5), "*": (0.5, 0.5)}


def _random_opening(board: chess.Board, depth: int, rng: random.Random) -> None:
    #play depth random moves to diversify opening positions
    for _ in range(depth):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))


def play_game(
    engine_white: Engine,
    engine_black: Engine,
    opening_depth: int = 0,
    max_moves: int = 200,
    rng: random.Random | None = None,
) -> tuple[str, chess.pgn.Game]:
    #plays a single game, returns (result_str, pgn_game)
    if rng is None:
        rng = random.Random()
    board = chess.Board()
    if opening_depth > 0:
        _random_opening(board, opening_depth, rng)

    game = chess.pgn.Game()
    game.setup(board)
    node = game

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        engine = engine_white if board.turn == chess.WHITE else engine_black
        try:
            move = engine.play(board)
        except Exception as e:
            print(f"  Engine error: {e} -- adjudicating as draw")
            break
        board.push(move)
        node = node.add_variation(move)

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    return result, game


def run_match(
    engine_a: Engine,
    engine_b: Engine,
    n_games: int,
    opening_depth: int = 4,
    max_moves: int = 200,
    seed: int = 0,
    verbose: bool = True,
    save_pgn: Optional[str] = None,
) -> dict[str, object]:
    #run a match between two engines; games alternate colors to reduce color advantage bias
    rng = random.Random(seed)
    wins_a = wins_b = draws = 0
    pgn_games: list[str] = []

    for game_idx in range(n_games):
        # alternate colours every game
        if game_idx % 2 == 0:
            white, black = engine_a, engine_b
            a_is_white = True
        else:
            white, black = engine_b, engine_a
            a_is_white = False

        result, pgn_game = play_game(
            white, black,
            opening_depth=opening_depth,
            max_moves=max_moves,
            rng=rng,
        )
        w_score, b_score = RESULT_SCORES.get(result, (0.5, 0.5))
        a_score = w_score if a_is_white else b_score

        if a_score > 0.5:
            wins_a += 1
        elif a_score < 0.5:
            wins_b += 1
        else:
            draws += 1

        if verbose:
            colour_a = "W" if a_is_white else "B"
            print(
                f"  Game {game_idx+1:>4d}/{n_games}  "
                f"A({colour_a})={a_score:.1f}  result={result}"
            )

        if save_pgn is not None:
            pgn_games.append(str(pgn_game))

    score_a = wins_a + 0.5 * draws
    score_b = wins_b + 0.5 * draws

    if save_pgn is not None:
        with open(save_pgn, "w") as f:
            f.write("\n\n".join(pgn_games))
        print(f"PGN saved to {save_pgn}")

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "score_a": score_a,
        "score_b": score_b,
        "games": n_games,
    }


def run_puzzle_eval(
    engine: Engine,
    puzzles_csv: str,
    num_puzzles: Optional[int] = None,
) -> dict[str, float]:
    #evaluate an engine on a Lichess puzzle CSV
    import pandas as pd
    from puzzles import evaluate_puzzle_from_pandas_row

    df = pd.read_csv(puzzles_csv, nrows=num_puzzles)
    correct = total = 0
    for _, row in df.iterrows():
        try:
            ok = evaluate_puzzle_from_pandas_row(row, engine)
            correct += int(ok)
        except Exception as e:
            print(f"  Puzzle error: {e}")
        total += 1

    acc = correct / max(1, total)
    return {"correct": correct, "total": total, "accuracy": acc}


def _select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arena: head-to-head competition between model checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-a", required=True,
                   help="Path to checkpoint A, or 'random' for a random engine.")
    p.add_argument("--model-b", required=True,
                   help="Path to checkpoint B, or 'random' for a random engine.")
    p.add_argument("--games", type=int, default=100,
                   help="Number of games to play.")
    p.add_argument("--opening-depth", type=int, default=4,
                   help="Random moves to play before handing off to engines.")
    p.add_argument("--max-moves", type=int, default=200,
                   help="Max half-moves per game before declaring a draw.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature for move selection (1.0 = greedy).")
    p.add_argument("--num-return-buckets", type=int, default=128,
                   help="Override number of return buckets (if not in checkpoint).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--save-pgn", default=None,
                   help="Save all games to this PGN file.")
    p.add_argument("--puzzles", default=None,
                   help="Evaluate on a Lichess puzzle CSV instead of playing games.")
    p.add_argument("--num-puzzles", type=int, default=None,
                   help="Number of puzzles to evaluate (default: all).")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _make_engine(
    spec: str,
    device: torch.device,
    temperature: float,
    num_return_buckets: int,
    seed: int,
) -> Engine:
    if spec.lower() == "random":
        return RandomEngine(seed=seed)
    model = load_model_from_checkpoint(spec, device)
    return ModelEngine(model, device, num_return_buckets=num_return_buckets,
                       temperature=temperature)


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    print(f"Device: {device}\n")

    print(f"Loading engine A: {args.model_a}")
    engine_a = _make_engine(
        args.model_a, device, args.temperature, args.num_return_buckets, args.seed
    )
    print(f"Loading engine B: {args.model_b}")
    engine_b = _make_engine(
        args.model_b, device, args.temperature, args.num_return_buckets, args.seed + 1
    )

    if args.puzzles:
        print(f"\n=== Puzzle evaluation ===")
        for name, eng in [("A", engine_a), ("B", engine_b)]:
            print(f"\nEngine {name}: {getattr(args, 'model_' + name.lower())}")
            result = run_puzzle_eval(eng, args.puzzles, args.num_puzzles)
            print(
                f"  Correct: {result['correct']}/{result['total']} "
                f"({result['accuracy']:.1%})"
            )
        return

    print(f"\n=== Match: {args.games} games ===\n")
    t0 = time.time()
    result = run_match(
        engine_a, engine_b,
        n_games=args.games,
        opening_depth=args.opening_depth,
        max_moves=args.max_moves,
        seed=args.seed,
        verbose=args.verbose,
        save_pgn=args.save_pgn,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Results after {result['games']} games ({elapsed:.1f}s):")
    print(f"  Engine A  ({args.model_a}):")
    print(f"    Score: {result['score_a']:.1f} / {result['games']}")
    print(f"    W/D/L:  {result['wins_a']} / {result['draws']} / {result['wins_b']}")
    print(f"  Engine B  ({args.model_b}):")
    print(f"    Score: {result['score_b']:.1f} / {result['games']}")

    score_pct = result["score_a"] / result["games"] * 100
    if score_pct > 55:
        print(f"\n  A wins the match ({score_pct:.1f}% score).")
    elif score_pct < 45:
        print(f"\n  B wins the match ({100-score_pct:.1f}% score).")
    else:
        print(f"\n  Match drawn ({score_pct:.1f}% for A).")


if __name__ == "__main__":
    main()
