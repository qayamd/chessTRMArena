#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Lichess puzzle evaluator -- checks if an engine picks the right move sequence
#puzzle format from DeepMind searchless chess (Ruoss et al., 2024)

from collections.abc import Sequence
import io

import chess
import chess.pgn
import pandas as pd


def evaluate_puzzle_from_pandas_row(
    puzzle: pd.Series,
    engine,
) -> bool:
    #returns True if engine solves the puzzle
    game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
    if game is None:
        raise ValueError(f'Failed to read game from PGN: {puzzle["PGN"]!r}')
    board = game.end().board()
    return evaluate_puzzle_from_board(
        board=board,
        moves=puzzle['Moves'].split(' '),
        engine=engine,
    )


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine,
) -> bool:
    #returns True if engine finds the right reply; any checkmate also counts as correct
    #even-indexed moves are opponent moves (played automatically), odd-indexed are engine replies
    for move_idx, move in enumerate(moves):
        if move_idx % 2 == 1:
            predicted_move = engine.play(board).uci()
            if move != predicted_move:
                board.push(chess.Move.from_uci(predicted_move))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(move))
    return True
