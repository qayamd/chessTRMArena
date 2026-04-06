#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#test suite for the chess evaluation pipeline
#run with: /c/Python312/python -m pytest tests.py -v

import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizer(unittest.TestCase):

    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_output_shape(self):
        from tokenizer import tokenize, SEQUENCE_LENGTH
        tokens = tokenize(self.START_FEN)
        self.assertEqual(tokens.shape, (SEQUENCE_LENGTH,))

    def test_dtype_is_int32(self):
        from tokenizer import tokenize
        tokens = tokenize(self.START_FEN)
        self.assertEqual(tokens.dtype, torch.int32)

    def test_values_in_vocab_range(self):
        from tokenizer import tokenize
        VOCAB_SIZE = 32
        tokens = tokenize(self.START_FEN)
        self.assertTrue((tokens >= 0).all())
        self.assertTrue((tokens < VOCAB_SIZE).all())

    def test_different_fens_differ(self):
        from tokenizer import tokenize
        fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        t1 = tokenize(self.START_FEN)
        t2 = tokenize(fen2)
        self.assertFalse(torch.equal(t1, t2))

    def test_fixed_output_length(self):
        #check that various FEN positions all tokenize to the same fixed length
        from tokenizer import tokenize, SEQUENCE_LENGTH
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "8/8/8/8/8/8/4K3/4k3 w - - 50 100",   # endgame, no castling
        ]
        for fen in fens:
            with self.subTest(fen=fen):
                self.assertEqual(tokenize(fen).shape[0], SEQUENCE_LENGTH)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data
# ─────────────────────────────────────────────────────────────────────────────

class TestData(unittest.TestCase):

    def test_synthetic_dataset_shape(self):
        from data import SyntheticChessDataset
        from tokenizer import SEQUENCE_LENGTH
        ds = SyntheticChessDataset(size=64, num_return_buckets=128)
        self.assertEqual(len(ds), 64)
        tokens, label = ds[0]
        self.assertEqual(tokens.shape, (SEQUENCE_LENGTH,))
        self.assertEqual(tokens.dtype, torch.int32)
        self.assertIsInstance(label.item(), int)

    def test_synthetic_dataloader_batch(self):
        from data import SyntheticChessDataset
        from tokenizer import SEQUENCE_LENGTH
        ds = SyntheticChessDataset(size=128, num_return_buckets=128)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        batch_tokens, batch_labels = next(iter(loader))
        self.assertEqual(batch_tokens.shape, (32, SEQUENCE_LENGTH))
        self.assertEqual(batch_labels.shape, (32,))

    def test_proto_record_parse_roundtrip(self):
        #build a minimal proto record and check that parse_proto_record decodes it
        from data import parse_proto_record

        def encode_varint(value: int) -> bytes:
            buf = []
            while value > 0x7F:
                buf.append((value & 0x7F) | 0x80)
                value >>= 7
            buf.append(value)
            return bytes(buf)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        bucket = 64
        fen_bytes = fen.encode("utf-8")

        # field 1 (fen, tag=0x0A): length-delimited
        record = bytes([0x0A]) + encode_varint(len(fen_bytes)) + fen_bytes
        # field 2 (return_bucket, tag=0x10): varint
        record += bytes([0x10]) + encode_varint(bucket)

        decoded_fen, decoded_bucket = parse_proto_record(record)
        self.assertEqual(decoded_fen, fen)
        self.assertEqual(decoded_bucket, bucket)

    def test_simple_record_roundtrip(self):
        from data import write_simple_record, parse_simple_record
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        bucket = 42
        raw = write_simple_record(fen, bucket)
        decoded_fen, decoded_bucket = parse_simple_record(raw)
        self.assertEqual(decoded_fen, fen)
        self.assertEqual(decoded_bucket, bucket)

    def test_auto_detect_proto(self):
        #auto-detection should pick proto format when record starts with 0x0A
        from data import parse_record, write_simple_record

        def encode_varint(value: int) -> bytes:
            buf = []
            while value > 0x7F:
                buf.append((value & 0x7F) | 0x80)
                value >>= 7
            buf.append(value)
            return bytes(buf)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_bytes = fen.encode("utf-8")
        record = bytes([0x0A]) + encode_varint(len(fen_bytes)) + fen_bytes
        record += bytes([0x10]) + encode_varint(99)

        decoded_fen, bucket = parse_record(record, fmt="auto")
        self.assertEqual(decoded_fen, fen)
        self.assertEqual(bucket, 99)

    def test_auto_detect_simple(self):
        from data import parse_record, write_simple_record
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        raw = write_simple_record(fen, 77)
        # First byte of simple format is part of a uint16, NOT 0x0A (unless bucket==10+256k).
        # Bucket 77 -> bytes [77, 0] -> first byte 0x4D != 0x0A -> auto detects simple.
        decoded_fen, bucket = parse_record(raw, fmt="auto")
        self.assertEqual(decoded_fen, fen)
        self.assertEqual(bucket, 77)

    def test_epoch_shuffle_sampler(self):
        from data import SyntheticChessDataset, EpochShuffleSampler
        ds = SyntheticChessDataset(size=100)
        sampler = EpochShuffleSampler(ds, seed=0)
        sampler.set_epoch(0)
        order_e0 = list(sampler)
        sampler.set_epoch(1)
        order_e1 = list(sampler)
        # different epochs should give different orderings
        self.assertNotEqual(order_e0, order_e1)
        # both orderings should be permutations of [0..99]
        self.assertEqual(sorted(order_e0), list(range(100)))
        self.assertEqual(sorted(order_e1), list(range(100)))

    def test_make_dataloader_no_shuffle(self):
        #make_dataloader with shuffle=False should return (loader, None)
        import gc, tempfile, os
        from data import make_dataloader, write_simple_record
        from bagz import BagWriter

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with tempfile.NamedTemporaryFile(suffix=".bag", delete=False) as f:
            tmp_path = f.name

        try:
            # write 32 records in simple format
            with BagWriter(tmp_path) as bw:
                for bucket in range(32):
                    bw.write(write_simple_record(fen, bucket))

            loader, sampler = make_dataloader(
                tmp_path, batch_size=16, shuffle=False, fmt="simple"
            )
            self.assertIsNone(sampler)
            batches = list(loader)
            self.assertEqual(len(batches), 2)   # 32 / 16 = 2, drop_last=True

            # release mmap before deleting on Windows
            del loader, sampler, batches
            gc.collect()
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRM
# ─────────────────────────────────────────────────────────────────────────────

def _make_trm(d=32, h=4, L=2, r=4, vocab=32, out=128):
    from TRM import TRM, TRMConfig
    from tokenizer import SEQUENCE_LENGTH
    cfg = TRMConfig(
        vocab_size=vocab,
        output_size=out,
        embedding_dim=d,
        num_heads=h,
        num_layers=L,
        n_recurrence=r,
        max_sequence_length=SEQUENCE_LENGTH,
    )
    return TRM(cfg), cfg


class TestTRM(unittest.TestCase):

    def test_forward_shape(self):
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm()
        tokens = torch.randint(0, cfg.vocab_size, (4, SEQUENCE_LENGTH), dtype=torch.long)
        out = model(tokens)
        self.assertEqual(out.shape, (4, SEQUENCE_LENGTH, cfg.output_size))

    def test_output_is_log_softmax(self):
        #last-position output should sum to ~1 when exponentiated
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm()
        model.eval()
        with torch.no_grad():
            tokens = torch.randint(0, cfg.vocab_size, (2, SEQUENCE_LENGTH), dtype=torch.long)
            out = model(tokens)                        # log-softmax
            probs = out[:, -1, :].exp()               # [B, V]
            row_sums = probs.sum(dim=-1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_deep_supervision_length(self):
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(r=4)
        tokens = torch.randint(0, cfg.vocab_size, (2, SEQUENCE_LENGTH), dtype=torch.long)
        outputs = model.forward_deep_supervision(tokens)
        self.assertEqual(len(outputs), cfg.n_recurrence)
        for o in outputs:
            self.assertEqual(o.shape, (2, SEQUENCE_LENGTH, cfg.output_size))

    def test_deep_supervision_shapes_match_forward(self):
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(r=3)
        model.eval()
        tokens = torch.randint(0, cfg.vocab_size, (2, SEQUENCE_LENGTH), dtype=torch.long)
        with torch.no_grad():
            final_out = model(tokens)
            ds_outputs = model.forward_deep_supervision(tokens)
        # last deep-supervision output should equal standard forward output
        self.assertTrue(torch.allclose(final_out, ds_outputs[-1], atol=1e-5))

    def test_parameter_count_reasonable(self):
        model, _ = _make_trm(d=64, h=8, L=2, r=8, vocab=32, out=128)
        n = sum(p.numel() for p in model.parameters())
        # d=64 model should be >10K and <2M params
        self.assertGreater(n, 10_000)
        self.assertLess(n, 2_000_000)

    def test_int32_input_accepted(self):
        #model should accept int32 tokens (as produced by tokenizer.py)
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm()
        tokens = torch.randint(0, cfg.vocab_size, (2, SEQUENCE_LENGTH), dtype=torch.int32)
        out = model(tokens.long())     # train.py calls .long() before passing
        self.assertEqual(out.shape[0], 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer
# ─────────────────────────────────────────────────────────────────────────────

def _make_tx(d=32, h=4, L=4, vocab=32, out=128):
    from transformer import TransformerDecoder, TransformerConfig
    from tokenizer import SEQUENCE_LENGTH
    cfg = TransformerConfig(
        vocab_size=vocab,
        output_size=out,
        embedding_dim=d,
        num_heads=h,
        num_layers=L,
        max_sequence_length=SEQUENCE_LENGTH,
    )
    return TransformerDecoder(cfg), cfg


class TestTransformer(unittest.TestCase):

    def test_forward_shape(self):
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_tx()
        tokens = torch.randint(0, cfg.vocab_size, (4, SEQUENCE_LENGTH), dtype=torch.long)
        out = model(tokens)
        self.assertEqual(out.shape, (4, SEQUENCE_LENGTH, cfg.output_size))

    def test_output_is_log_softmax(self):
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_tx()
        model.eval()
        with torch.no_grad():
            tokens = torch.randint(0, cfg.vocab_size, (2, SEQUENCE_LENGTH), dtype=torch.long)
            out = model(tokens)
            probs = out[:, -1, :].exp()
            row_sums = probs.sum(dim=-1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_parameter_count_reasonable(self):
        model, _ = _make_tx(d=64, h=8, L=4, vocab=32, out=128)
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 10_000)
        self.assertLess(n, 5_000_000)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCompatibility(unittest.TestCase):
    #verify TRM and Transformer have identical forward signatures

    def _run(self, model, vocab_size):
        from tokenizer import SEQUENCE_LENGTH
        B, T = 3, SEQUENCE_LENGTH
        tokens = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
        out = model(tokens)
        return out

    def test_same_output_shape(self):
        from tokenizer import SEQUENCE_LENGTH
        trm, trm_cfg = _make_trm(d=32, h=4, out=128)
        tx, tx_cfg = _make_tx(d=32, h=4, out=128)
        out_trm = self._run(trm, trm_cfg.vocab_size)
        out_tx = self._run(tx, tx_cfg.vocab_size)
        self.assertEqual(out_trm.shape, out_tx.shape)

    def test_shared_config_compatibility(self):
        #both models should accept the same shared config dict
        from TRM import TRM, TRMConfig
        from transformer import TransformerDecoder, TransformerConfig
        from tokenizer import SEQUENCE_LENGTH

        shared = dict(
            vocab_size=32,
            output_size=128,
            embedding_dim=32,
            num_heads=4,
            max_sequence_length=SEQUENCE_LENGTH,
        )
        trm = TRM(TRMConfig(**shared, num_layers=2, n_recurrence=4))
        tx = TransformerDecoder(TransformerConfig(**shared, num_layers=2))

        tokens = torch.randint(0, 32, (2, SEQUENCE_LENGTH), dtype=torch.long)
        out_trm = trm(tokens)
        out_tx = tx(tokens)
        self.assertEqual(out_trm.shape, out_tx.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainUtilities(unittest.TestCase):

    def test_cosine_lr_warmup_peak(self):
        from train import cosine_lr_with_warmup
        lr = cosine_lr_with_warmup(
            step=1000, total_steps=10000, warmup_steps=1000, peak_lr=1e-3
        )
        self.assertAlmostEqual(lr, 1e-3, places=6)

    def test_cosine_lr_warmup_zero(self):
        from train import cosine_lr_with_warmup
        lr = cosine_lr_with_warmup(
            step=0, total_steps=10000, warmup_steps=1000, peak_lr=1e-3
        )
        self.assertAlmostEqual(lr, 0.0, places=6)

    def test_cosine_lr_decay(self):
        from train import cosine_lr_with_warmup
        lr_mid = cosine_lr_with_warmup(
            step=5000, total_steps=10000, warmup_steps=1000, peak_lr=1e-3
        )
        lr_end = cosine_lr_with_warmup(
            step=10000, total_steps=10000, warmup_steps=1000, peak_lr=1e-3
        )
        self.assertGreater(lr_mid, lr_end)
        self.assertLess(lr_end, 1e-4)

    def test_compute_loss_trm(self):
        from train import compute_loss
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(d=16, h=2, L=1, r=2, out=64)
        model.eval()
        B = 4
        tokens = torch.randint(0, cfg.vocab_size, (B, SEQUENCE_LENGTH), dtype=torch.long)
        labels = torch.randint(0, cfg.output_size, (B,), dtype=torch.long)
        loss, acc = compute_loss(model, tokens, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertTrue(0.0 <= acc.item() <= 1.0)
        self.assertTrue(loss.item() > 0)

    def test_compute_loss_deep_supervision(self):
        from train import compute_loss
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(d=16, h=2, L=1, r=3, out=64)
        model.eval()
        B = 2
        tokens = torch.randint(0, cfg.vocab_size, (B, SEQUENCE_LENGTH), dtype=torch.long)
        labels = torch.randint(0, cfg.output_size, (B,), dtype=torch.long)
        loss_std, _ = compute_loss(model, tokens, labels, deep_supervision=False)
        loss_ds, _ = compute_loss(model, tokens, labels, deep_supervision=True)
        # both should be valid scalars
        self.assertFalse(torch.isnan(loss_std))
        self.assertFalse(torch.isnan(loss_ds))

    def test_compute_loss_transformer(self):
        from train import compute_loss
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_tx(d=16, h=2, L=2, out=64)
        model.eval()
        B = 4
        tokens = torch.randint(0, cfg.vocab_size, (B, SEQUENCE_LENGTH), dtype=torch.long)
        labels = torch.randint(0, cfg.output_size, (B,), dtype=torch.long)
        loss, acc = compute_loss(model, tokens, labels)
        self.assertTrue(loss.item() > 0)
        self.assertTrue(0.0 <= acc.item() <= 1.0)

    def test_one_training_step(self):
        #full forward-backward-update cycle should not crash
        from train import compute_loss
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(d=16, h=2, L=1, r=2, out=64)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, cfg.vocab_size, (8, SEQUENCE_LENGTH), dtype=torch.long)
        labels = torch.randint(0, cfg.output_size, (8,), dtype=torch.long)

        before = [p.clone() for p in model.parameters()]
        optimizer.zero_grad()
        loss, _ = compute_loss(model, tokens, labels, deep_supervision=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        after = list(model.parameters())

        # at least one parameter should have changed
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed)

    def test_save_load_checkpoint(self):
        #save_checkpoint / load_checkpoint roundtrip
        import argparse, tempfile
        from train import save_checkpoint, load_checkpoint, build_model

        from tokenizer import SEQUENCE_LENGTH
        from TRM import TRM, TRMConfig
        cfg = TRMConfig(vocab_size=32, output_size=64,
                        embedding_dim=16, num_heads=2,
                        num_layers=1, n_recurrence=2,
                        max_sequence_length=SEQUENCE_LENGTH)
        model = TRM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        args = argparse.Namespace(
            model="trm", embedding_dim=16, num_heads=2, num_layers=1,
            n_recurrence=2, num_return_buckets=64, widening_factor=4,
            use_causal_mask=False, apply_post_ln=True, seed=0,
            data=None, val_data=None, steps=100, batch_size=8, lr=1e-3,
            weight_decay=1e-2, max_grad_norm=1.0, warmup_steps=10,
            deep_supervision=False, use_ema=False, ema_mu=0.999,
            output_dir="results", run_name="test", log_every=10,
            ckpt_every=50, keep_ckpts=3, resume=None, eval_only=False,
            device="cpu", data_fmt="auto", num_records=None,
            num_val_records=None,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, step=100, model=model,
                            optimizer=optimizer, args=args)

            # build a fresh model and load into it
            model2 = TRM(cfg)
            step, _, _ema_sd = load_checkpoint(tmp_path, model2,
                                              device=torch.device("cpu"))
            self.assertEqual(step, 100)

            # weights should match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Arena
# ─────────────────────────────────────────────────────────────────────────────

class TestArena(unittest.TestCase):

    def test_random_engine_returns_legal_move(self):
        import chess
        from arena import RandomEngine
        board = chess.Board()
        engine = RandomEngine(seed=0)
        for _ in range(10):
            move = engine.play(board)
            self.assertIn(move, board.legal_moves)
            board.push(move)

    def test_model_engine_trm(self):
        #ModelEngine (TRM) should return a legal move for a given position
        import chess
        from arena import ModelEngine
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(d=16, h=2, L=1, r=2, out=128)
        model.eval()
        engine = ModelEngine(model, torch.device("cpu"),
                             num_return_buckets=128)
        board = chess.Board()
        move = engine.play(board)
        self.assertIn(move, board.legal_moves)

    def test_model_engine_transformer(self):
        #ModelEngine (Transformer) should return a legal move
        import chess
        from arena import ModelEngine
        model, cfg = _make_tx(d=16, h=2, L=2, out=128)
        model.eval()
        engine = ModelEngine(model, torch.device("cpu"),
                             num_return_buckets=128)
        board = chess.Board()
        move = engine.play(board)
        self.assertIn(move, board.legal_moves)

    def test_play_game_terminates(self):
        #play_game should always produce a result string, not loop forever
        from arena import RandomEngine, play_game
        ea = RandomEngine(seed=1)
        eb = RandomEngine(seed=2)
        result, game = play_game(ea, eb, opening_depth=0, max_moves=300)
        self.assertIn(result, ("1-0", "0-1", "1/2-1/2", "*"))

    def test_run_match_counts(self):
        #run_match totals should add up correctly
        from arena import RandomEngine, run_match
        ea = RandomEngine(seed=3)
        eb = RandomEngine(seed=4)
        n = 6
        result = run_match(ea, eb, n_games=n, opening_depth=2,
                           max_moves=100, verbose=False)
        self.assertEqual(result["games"], n)
        self.assertEqual(
            result["wins_a"] + result["wins_b"] + result["draws"], n
        )
        self.assertAlmostEqual(
            result["score_a"] + result["score_b"], float(n), places=4
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration: main.py preset building
# ─────────────────────────────────────────────────────────────────────────────

class TestMainPresets(unittest.TestCase):

    def test_all_presets_build_valid_argv(self):
        from main import _build_argv, PRESETS
        for name in PRESETS:
            argv = _build_argv(name, [])
            self.assertIsInstance(argv, list)

    def test_user_flag_overrides_preset(self):
        from main import _build_argv
        argv = _build_argv("tiny_trm", ["--embedding-dim", "256"])
        # embedding dim from user should override preset value of 64
        joined = " ".join(argv)
        self.assertIn("256", joined)
        # there should not be a separate --embedding-dim 64 in the output
        idx = argv.index("--embedding-dim")
        self.assertEqual(argv[idx + 1], "256")

    def test_preset_produces_parseable_args(self):
        from main import _build_argv
        from train import parse_args
        argv = _build_argv("tiny_trm", ["--steps", "10", "--no-apply-post-ln"])
        args = parse_args(argv)
        self.assertEqual(args.model, "trm")
        self.assertEqual(args.steps, 10)
        self.assertFalse(args.apply_post_ln)


# ─────────────────────────────────────────────────────────────────────────────
# 9. EMA helper
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAHelper(unittest.TestCase):

    def test_ema_copy_differs_from_original(self):
        from TRM import EMAHelper
        from tokenizer import SEQUENCE_LENGTH
        model, cfg = _make_trm(d=16, h=2, L=1, r=2, out=64)
        model.train()
        ema = EMAHelper(mu=0.9)
        ema.register(model)

        # do several steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(5):
            tokens = torch.randint(0, cfg.vocab_size, (4, SEQUENCE_LENGTH), dtype=torch.long)
            labels = torch.randint(0, cfg.output_size, (4,), dtype=torch.long)
            loss = torch.nn.functional.nll_loss(model(tokens)[:, -1, :], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

        ema_model = ema.ema_copy(model)
        # EMA model parameters should differ from current model params
        any_diff = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(model.parameters(), ema_model.parameters())
        )
        self.assertTrue(any_diff)

    def test_ema_state_dict_roundtrip(self):
        from TRM import EMAHelper
        model, cfg = _make_trm(d=8, h=2, L=1, r=1, out=16)
        ema = EMAHelper(mu=0.99)
        ema.register(model)

        state = ema.state_dict()
        ema2 = EMAHelper(mu=0.99)
        ema2.load_state_dict(state)
        # shadow weights should be identical
        for k in state:
            self.assertTrue(torch.equal(state[k], ema2.shadow[k]))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Real data files (bcTrain.bag / bcTest.bag)
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

BC_TRAIN = _HERE / "bcTrain.bag"
BC_TEST  = _HERE / "bcTest.bag"


@unittest.skipUnless(BC_TRAIN.exists(), "bcTrain.bag not found -- skipping")
class TestBcTrainBag(unittest.TestCase):
    #verify bcTrain.bag records parse into valid (FEN, bucket) pairs

    def setUp(self):
        from bagz import BagReader
        self._reader = BagReader(str(BC_TRAIN))

    def test_record_count_positive(self):
        self.assertGreater(len(self._reader), 0)

    def test_parse_first_records(self):
        #first 20 records should all produce valid FENs and in-range buckets
        import chess
        from data import parse_bc_train_record
        for i in range(min(20, len(self._reader))):
            with self.subTest(record=i):
                raw = self._reader[i]
                fen, bucket = parse_bc_train_record(raw)
                # bucket must be in [0, 127]
                self.assertGreaterEqual(bucket, 0)
                self.assertLessEqual(bucket, 127)
                # FEN must be parseable by python-chess
                board = chess.Board(fen)
                self.assertIsNotNone(board)

    def test_bucket_uses_full_range(self):
        #buckets should span a wide range, not just a few values
        from data import parse_bc_train_record
        buckets = set()
        for i in range(min(5000, len(self._reader))):
            _, bucket = parse_bc_train_record(self._reader[i])
            buckets.add(bucket)
        # real data should produce many distinct buckets across [0, 127]
        self.assertGreater(len(buckets), 80,
                           f"Only {len(buckets)} distinct buckets in 5000 records")
        self.assertIn(0, buckets)     # near-certain white loss
        self.assertIn(127, buckets)   # near-certain white win

    def test_fen_length_prefix_matches(self):
        #byte 0 should equal the length of the decoded FEN string
        from data import parse_bc_train_record
        for i in range(min(50, len(self._reader))):
            raw = self._reader[i]
            fen_len_byte = raw[0]
            fen, _ = parse_bc_train_record(raw)
            self.assertEqual(fen_len_byte, len(fen),
                             f"Record {i}: byte[0]={fen_len_byte} != len(fen)={len(fen)}")

    def test_auto_detect_picks_bc_train(self):
        #auto-detection should choose 'bc_train' for these records
        from data import _auto_detect_fmt
        for i in range(min(10, len(self._reader))):
            raw = self._reader[i]
            self.assertEqual(_auto_detect_fmt(raw), "bc_train",
                             f"Record {i}: expected bc_train, got {_auto_detect_fmt(raw)!r}")

    def test_tokens_shape(self):
        from data import parse_bc_train_record
        from tokenizer import tokenize, SEQUENCE_LENGTH
        raw = self._reader[0]
        fen, _ = parse_bc_train_record(raw)
        tokens = tokenize(fen)
        self.assertEqual(tokens.shape, (SEQUENCE_LENGTH,))


@unittest.skipUnless(BC_TEST.exists(), "bcTest.bag not found -- skipping")
class TestBcTestBag(unittest.TestCase):
    #verify bcTest.bag records parse into valid FENs (no bucket stored)

    def setUp(self):
        from bagz import BagReader
        self._reader = BagReader(str(BC_TEST))

    def test_record_count_positive(self):
        self.assertGreater(len(self._reader), 0)

    def test_parse_first_records(self):
        #first 20 records should produce valid FENs; bucket should be -1
        import chess
        from data import parse_bc_test_record
        for i in range(min(20, len(self._reader))):
            with self.subTest(record=i):
                raw = self._reader[i]
                fen, bucket = parse_bc_test_record(raw)
                self.assertEqual(bucket, -1)
                board = chess.Board(fen)
                self.assertIsNotNone(board)

    def test_fen_length_prefix_matches(self):
        #byte 0 should equal the length of the decoded FEN string
        from data import parse_bc_test_record
        for i in range(min(50, len(self._reader))):
            raw = self._reader[i]
            fen_len_byte = raw[0]
            fen, _ = parse_bc_test_record(raw)
            self.assertEqual(fen_len_byte, len(fen),
                             f"Record {i}: byte[0]={fen_len_byte} != len(fen)={len(fen)}")

    def test_trailing_bytes_are_uci_move(self):
        #bytes after the FEN should be a valid UCI move
        import chess
        from data import parse_bc_test_record
        for i in range(min(20, len(self._reader))):
            with self.subTest(record=i):
                raw = self._reader[i]
                fen_len = raw[0]
                fen, _ = parse_bc_test_record(raw)
                move_str = raw[1 + fen_len :].decode("ascii")
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_str)
                self.assertIn(move, board.legal_moves,
                              f"Record {i}: move {move_str} not legal in {fen}")

    def test_auto_detect_picks_bc_test(self):
        #auto-detection should choose 'bc_test' for these records
        from data import _auto_detect_fmt
        for i in range(min(10, len(self._reader))):
            raw = self._reader[i]
            self.assertEqual(_auto_detect_fmt(raw), "bc_test",
                             f"Record {i}: expected bc_test, got {_auto_detect_fmt(raw)!r}")

    def test_tokens_shape(self):
        from data import parse_bc_test_record
        from tokenizer import tokenize, SEQUENCE_LENGTH
        raw = self._reader[0]
        fen, _ = parse_bc_test_record(raw)
        tokens = tokenize(fen)
        self.assertEqual(tokens.shape, (SEQUENCE_LENGTH,))

    def test_dataloader_bc_test(self):
        #BagChessDataset + DataLoader should iterate without errors
        from data import BagChessDataset
        from tokenizer import SEQUENCE_LENGTH
        ds = BagChessDataset(str(BC_TEST), num_records=64, fmt="bc_test")
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        batch_tokens, batch_labels = next(iter(loader))
        self.assertEqual(batch_tokens.shape, (16, SEQUENCE_LENGTH))
        self.assertEqual(batch_labels.shape, (16,))
        # all labels should be -1 (no bucket in bcTest)
        self.assertTrue((batch_labels == -1).all())


# ─────────────────────────────────────────────────────────────────────────────
# 11. Puzzles
# ─────────────────────────────────────────────────────────────────────────────

_PUZZLES_CSV = _HERE / "puzzles.csv"


@unittest.skipUnless(_PUZZLES_CSV.exists(), "puzzles.csv not found -- skipping")
class TestPuzzles(unittest.TestCase):
    #smoke-test evaluate_puzzle_from_pandas_row with the local puzzle CSV

    def test_import_evaluate_puzzle(self):
        #puzzles.py should be importable without errors
        from puzzles import evaluate_puzzle_from_pandas_row  # noqa: F401

    def test_random_engine_on_first_puzzle(self):
        #RandomEngine should run through a puzzle without crashing
        import pandas as pd
        from puzzles import evaluate_puzzle_from_pandas_row
        from arena import RandomEngine

        df = pd.read_csv(str(_PUZZLES_CSV), nrows=1)
        puzzle = df.iloc[0]
        engine = RandomEngine(seed=99)
        result = evaluate_puzzle_from_pandas_row(puzzle, engine)
        # result should be a bool (correct or incorrect -- we don't care which)
        self.assertIsInstance(result, bool)

    def test_puzzle_csv_columns(self):
        #confirm the CSV has the columns our code depends on
        import pandas as pd
        df = pd.read_csv(str(_PUZZLES_CSV), nrows=1)
        for col in ("PGN", "Moves", "Rating"):
            self.assertIn(col, df.columns, f"Missing column: {col}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
