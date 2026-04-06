#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#training loop for both TRM (Jolicoeur-Martineau, 2025) and Transformer (Ruoss et al., 2024)
#handles checkpointing, EMA, deep supervision, gradient checkpointing, and AMP

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import (
    make_dataloader, SyntheticChessDataset, BagChessDataset,
    EpochShuffleSampler, PrefetchLoader,
)
from tokenizer import SEQUENCE_LENGTH


# lazy imports so the script is importable even if a model file has errors
def _import_trm():
    from TRM import TRM, TRMConfig, EMAHelper
    return TRM, TRMConfig, EMAHelper


def _import_transformer():
    from transformer import TransformerDecoder, TransformerConfig
    return TransformerDecoder, TransformerConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a chess evaluation model (TRM or Transformer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    data = p.add_argument_group("data")
    data.add_argument("--data", metavar="PATH",
                      help="Path to training .bag/.bagz file. "
                           "Omit to use a small synthetic dataset (for testing).")
    data.add_argument("--val-data", metavar="PATH",
                      help="Path to validation .bag/.bagz file (optional). "
                           "If omitted and --val-fraction > 0, a split of "
                           "the training data is used instead.")
    data.add_argument("--val-fraction", type=float, default=0.0,
                      help="Fraction of training data to hold out for "
                           "validation (e.g. 0.001 = 0.1%%). Ignored when "
                           "--val-data is provided.")
    data.add_argument("--data-fmt", choices=["auto", "proto", "simple"],
                      default="auto",
                      help="Bagz record format: auto-detected if 'auto'.")
    data.add_argument("--num-records", type=int, default=None,
                      help="Limit training records (useful for debugging).")
    data.add_argument("--num-val-records", type=int, default=None,
                      help="Limit validation records.")
    data.add_argument("--num-return-buckets", type=int, default=128,
                      help="Number of return/value buckets (output size).")

    # model
    model = p.add_argument_group("model")
    model.add_argument("--model", choices=["trm", "transformer"],
                       default="trm",
                       help="Architecture to train.")
    model.add_argument("--embedding-dim", type=int, default=64)
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--widening-factor", type=int, default=4)
    model.add_argument("--num-layers", type=int, default=None,
                       help="Transformer layers per recurrence step (TRM) or "
                            "total stacked layers (Transformer). "
                            "Default: 2 for TRM, 4 for Transformer.")
    model.add_argument("--n-recurrence", type=int, default=8,
                       help="[TRM only] Number of recurrence steps.")
    model.add_argument("--apply-post-ln", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Apply final LayerNorm before output projection.")
    model.add_argument("--use-causal-mask", action="store_true",
                       help="Use causal (autoregressive) attention mask.")
    model.add_argument("--gradient-checkpointing", action="store_true",
                       help="[TRM only] Recompute activations during backward "
                            "to save VRAM. Essential for large n_recurrence.")

    # training
    train = p.add_argument_group("training")
    train.add_argument("--steps", type=int, default=100_000,
                       help="Total gradient steps.")
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-3,
                       help="Peak learning rate for AdamW.")
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Gradient clipping (0 to disable).")
    train.add_argument("--warmup-steps", type=int, default=1000,
                       help="Linear LR warmup steps before cosine decay.")
    train.add_argument("--deep-supervision", action="store_true",
                       help="[TRM only] Use forward_deep_supervision() with "
                            "linearly-increasing step weights.")
    train.add_argument("--use-ema", action="store_true",
                       help="Maintain an EMA of model weights (TRM paper 4.7).")
    train.add_argument("--ema-mu", type=float, default=0.999,
                       help="EMA decay factor (higher = slower update).")
    train.add_argument("--seed", type=int, default=42)

    # i/o
    io = p.add_argument_group("i/o")
    io.add_argument("--output-dir", metavar="PATH", default="results",
                    help="Directory to save checkpoints and logs.")
    io.add_argument("--run-name", default=None,
                    help="Sub-directory name under output-dir. "
                         "Default: {model}_{timestamp}.")
    io.add_argument("--log-every", type=int, default=100,
                    help="Log metrics every N steps.")
    io.add_argument("--ckpt-every", type=int, default=10_000,
                    help="Save a checkpoint every N steps (0 to disable).")
    io.add_argument("--keep-ckpts", type=int, default=3,
                    help="How many recent checkpoints to keep (0 = keep all).")
    io.add_argument("--resume", metavar="PATH",
                    help="Resume training from this checkpoint file.")
    io.add_argument("--eval-only", action="store_true",
                    help="Skip training; only run evaluation on --val-data.")
    io.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                    default="auto")

    return p.parse_args(argv)


def build_model(args: argparse.Namespace) -> nn.Module:
    #instantiate the model described by args
    shared = dict(
        vocab_size=32,                        # tokenizer.py vocabulary
        output_size=args.num_return_buckets,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        widening_factor=args.widening_factor,
        use_causal_mask=args.use_causal_mask,
        apply_post_ln=args.apply_post_ln,
        max_sequence_length=SEQUENCE_LENGTH,
        seed=args.seed,
    )
    if args.model == "trm":
        TRM, TRMConfig, _ = _import_trm()
        num_layers = args.num_layers if args.num_layers is not None else 2
        gc = getattr(args, 'gradient_checkpointing', False)
        cfg = TRMConfig(**shared, num_layers=num_layers,
                        n_recurrence=args.n_recurrence,
                        gradient_checkpointing=gc)
        return TRM(cfg)
    else:
        TransformerDecoder, TransformerConfig = _import_transformer()
        num_layers = args.num_layers if args.num_layers is not None else 4
        cfg = TransformerConfig(**shared, num_layers=num_layers)
        return TransformerDecoder(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    #linear warmup then cosine decay to min_lr
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def compute_loss(
    model: nn.Module,
    tokens: torch.Tensor,           # [B, T]  int32
    labels: torch.Tensor,           # [B]     int64
    deep_supervision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    #compute NLL loss and top-1 accuracy
    #for TRM with deep_supervision, applies linearly-weighted loss across all recurrence steps
    tokens = tokens.long()

    if deep_supervision and hasattr(model, "forward_deep_supervision"):
        outputs = model.forward_deep_supervision(tokens)   # list of [B,T,V]
        n = len(outputs)
        weights = torch.linspace(1.0, float(n), n, device=tokens.device)
        total_loss = torch.zeros((), device=tokens.device)
        for w, logits in zip(weights, outputs):
            step_logits = logits[:, -1, :]                 # [B, V]
            total_loss = total_loss + w * F.nll_loss(step_logits, labels)
        loss = total_loss / weights.sum()
        acc_logits = outputs[-1][:, -1, :]
    else:
        logits = model(tokens)                             # [B, T, V]
        loss = F.nll_loss(logits[:, -1, :], labels)
        acc_logits = logits[:, -1, :]

    preds = acc_logits.argmax(dim=-1)
    acc = (preds == labels).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    #run evaluation on a dataloader; returns dict with 'loss' and 'acc'
    #batches where all labels are -1 (bcTest) are silently skipped
    model.eval()
    total_loss = total_acc = n = 0
    for i, (tokens, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # skip batches with no valid labels (bcTest returns -1)
        if (labels < 0).all():
            continue
        tokens, labels = tokens.to(device), labels.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            loss, acc = compute_loss(model, tokens, labels, deep_supervision=False)
        total_loss += loss.item()
        total_acc += acc.item()
        n += 1
    model.train()
    if n == 0:
        print("  Warning: no valid labels in val data (all -1). "
              "bcTest.bag is an action-value dataset with no bucket labels; "
              "use bcTrain.bag or a state-value dataset for --val-data.")
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / n, "acc": total_acc / n}


def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ema_state: dict | None = None,
    metrics: dict | None = None,
) -> None:
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "ema_state_dict": ema_state,
        "metrics": metrics or {},
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[int, dict, dict | None]:
    #load checkpoint; returns (step, metrics, ema_state_dict)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("metrics", {}), ckpt.get("ema_state_dict")


def _prune_old_checkpoints(ckpt_dir: Path, keep: int) -> None:
    #delete oldest checkpoints if more than keep exist
    if keep <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


def _infinite_loader(loader, sampler) -> Iterator:
    #yields batches forever, re-shuffling at each epoch boundary
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _make_split_dataloaders(
    path: str,
    batch_size: int,
    val_fraction: float,
    *,
    num_records: int | None = None,
    fmt: str = "auto",
    num_return_buckets: int = 128,
    seed: int = 0,
):
    #split a single .bag file into train and val DataLoaders
    #uses the last val_fraction of records for val; both share the same mmap'd BagReader
    from torch.utils.data import Subset

    dataset = BagChessDataset(
        path, num_records=num_records, fmt=fmt,
        num_return_buckets=num_return_buckets,
    )
    total = len(dataset)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size

    train_subset = Subset(dataset, range(train_size))
    val_subset = Subset(dataset, range(train_size, total))

    train_sampler = EpochShuffleSampler(train_subset, seed=seed)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_loader = PrefetchLoader(train_loader)

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = PrefetchLoader(val_loader)

    print(f"Train/val split: {train_size:,} train, {val_size:,} val "
          f"({val_fraction:.2%} held out)")
    return train_loader, val_loader, train_sampler


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    # device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # TF32 gives ~2x speedup on Ampere+ matmuls with <0.1% precision loss
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TF32 enabled for matmul and cuDNN")

    torch.manual_seed(args.seed)

    run_name = args.run_name or f"{args.model}_{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = build_model(args).to(device)
    n_params = count_params(model)
    print(f"Model: {args.model}  |  parameters: {n_params:,}")

    # EMA setup
    ema = None
    if args.use_ema and args.model == "trm":
        _, _, EMAHelper = _import_trm()
        ema = EMAHelper(mu=args.ema_mu)
        ema.register(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # AMP (mixed precision)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP (float16 autocast) enabled")

    # data loaders
    val_loader = None

    if args.data:
        use_split = (args.val_fraction > 0 and not args.val_data)
        if use_split:
            train_loader, val_loader, train_sampler = _make_split_dataloaders(
                args.data,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                num_records=args.num_records,
                fmt=args.data_fmt,
                num_return_buckets=args.num_return_buckets,
                seed=args.seed,
            )
        else:
            train_loader, train_sampler = make_dataloader(
                args.data,
                batch_size=args.batch_size,
                shuffle=True,
                num_records=args.num_records,
                fmt=args.data_fmt,
                num_return_buckets=args.num_return_buckets,
                seed=args.seed,
            )
    else:
        print("No --data provided; using synthetic dataset for testing.")
        from torch.utils.data import DataLoader as _DL
        train_ds = SyntheticChessDataset(
            size=max(args.batch_size * 20, 2048),
            num_return_buckets=args.num_return_buckets,
        )
        train_loader = _DL(train_ds, batch_size=args.batch_size, shuffle=True)
        train_sampler = None

    if args.val_data:
        val_loader, _ = make_dataloader(
            args.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_records=args.num_val_records,
            fmt=args.data_fmt,
            num_return_buckets=args.num_return_buckets,
        )

    # resume from checkpoint
    start_step = 0
    if args.resume:
        start_step, _, ema_sd = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resumed from {args.resume} at step {start_step}")
        if ema is not None:
            _, _, EMAHelper = _import_trm()
            ema = EMAHelper(mu=args.ema_mu)
            ema.register(model)
            if ema_sd is not None:
                ema.load_state_dict(ema_sd)
                print("  EMA state restored from checkpoint.")
            else:
                print("  Warning: no EMA state in checkpoint; using current weights.")

    if args.eval_only:
        if val_loader is None:
            print("--eval-only requires --val-data")
            return
        metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        print(f"Eval: loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}")
        return

    # training loop
    model.train()
    data_iter = _infinite_loader(train_loader, train_sampler)
    log_file = open(run_dir / "train.log", "a")

    t0 = time.time()
    running_loss = running_acc = running_grad_norm = 0.0
    peak_loss = float("-inf")
    min_loss = float("inf")
    log_count = 0
    total_tokens = 0
    log_interval_t0 = time.time()

    total_steps = args.steps - start_step
    pbar = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        ascii=True,
    )

    for step in pbar:
        # LR schedule
        lr = cosine_lr_with_warmup(step, args.steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        tokens, labels = next(data_iter)
        tokens, labels = tokens.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            loss, acc = compute_loss(
                model, tokens, labels,
                deep_supervision=(args.deep_supervision and args.model == "trm"),
            )
        scaler.scale(loss).backward()

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            ).item()
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")
            ).item()

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        step_loss = loss.item()
        running_loss += step_loss
        running_acc += acc.item()
        running_grad_norm += grad_norm
        peak_loss = max(peak_loss, step_loss)
        min_loss = min(min_loss, step_loss)
        log_count += 1
        total_tokens += tokens.numel()

        pbar.set_postfix_str(
            f"loss={step_loss:.4f} acc={acc.item():.4f} lr={lr:.1e}",
            refresh=False,
        )

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            interval_elapsed = time.time() - log_interval_t0
            avg_loss = running_loss / log_count
            avg_acc = running_acc / log_count
            avg_grad_norm = running_grad_norm / log_count
            steps_per_sec = log_count / max(interval_elapsed, 1e-9)
            samples_per_sec = (log_count * args.batch_size) / max(interval_elapsed, 1e-9)
            steps_remaining = args.steps - (step + 1)
            eta_sec = steps_remaining / max(steps_per_sec, 1e-9)

            # VRAM usage (CUDA only)
            vram_str = ""
            if device.type == "cuda":
                vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                vram_str = f"  vram_peak={vram_gb:.2f}GB"

            line = (
                f"step={step+1:>7d}  loss={avg_loss:.4f}  acc={avg_acc:.4f}"
                f"  lr={lr:.2e}  grad_norm={avg_grad_norm:.4f}"
                f"  loss_min={min_loss:.4f}  loss_max={peak_loss:.4f}"
                f"  steps/s={steps_per_sec:.1f}  samples/s={samples_per_sec:.0f}"
                f"  tokens={total_tokens:,}{vram_str}"
                f"  elapsed={elapsed:.1f}s  eta={eta_sec:.0f}s"
            )
            tqdm.write(line)
            log_file.write(line + "\n")
            log_file.flush()

            # reset running stats
            running_loss = running_acc = running_grad_norm = 0.0
            peak_loss = float("-inf")
            min_loss = float("inf")
            log_count = 0
            log_interval_t0 = time.time()

        # periodic validation
        if val_loader is not None and args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
            val_line = (
                f"  [val] step={step+1:>7d}  val_loss={val_metrics['loss']:.4f}"
                f"  val_acc={val_metrics['acc']:.4f}"
            )
            tqdm.write(val_line)
            log_file.write(val_line + "\n")
            log_file.flush()
            model.train()

        # checkpointing
        if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            ckpt_path = str(run_dir / f"step_{step+1:07d}.pt")
            save_checkpoint(
                ckpt_path, step + 1, model, optimizer, args,
                ema_state=ema.state_dict() if ema is not None else None,
            )
            tqdm.write(f"  -> checkpoint: {ckpt_path}")
            _prune_old_checkpoints(run_dir, args.keep_ckpts)

    pbar.close()

    # final checkpoint
    final_path = str(run_dir / f"step_{args.steps:07d}_final.pt")
    save_checkpoint(
        final_path, args.steps, model, optimizer, args,
        ema_state=ema.state_dict() if ema is not None else None,
    )
    elapsed_total = time.time() - t0
    print(f"\nTraining complete in {elapsed_total:.1f}s. Final checkpoint: {final_path}")

    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        print(f"Val: loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}")

    log_file.close()


if __name__ == "__main__":
    main()
