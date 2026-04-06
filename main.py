#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#preset launcher for train.py -- wraps named size configs (tiny_trm, small_trm, tiny_tx, etc.)

import argparse
import sys
from train import main as train_main, parse_args


# preset definitions -- CLI flags as defaults, user flags override

PRESETS: dict[str, dict[str, object]] = {
    "tiny_trm": {
        "model": "trm",
        "embedding_dim": 64,
        "num_heads": 8,
        "num_layers": 2,
        "n_recurrence": 8,
        "deep_supervision": True,
        "use_ema": True,
        "steps": 100_000,
        "batch_size": 512,
        "lr": 1e-3,
        "warmup_steps": 1_000,
        "run_name": "tiny_trm",
    },
    "small_trm": {
        "model": "trm",
        "embedding_dim": 128,
        "num_heads": 8,
        "num_layers": 2,
        "n_recurrence": 8,
        "deep_supervision": True,
        "use_ema": True,
        "steps": 200_000,
        "batch_size": 512,
        "lr": 5e-4,
        "warmup_steps": 2_000,
        "run_name": "small_trm",
    },
    "medium_trm": {
        "model": "trm",
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "n_recurrence": 8,
        "deep_supervision": True,
        "use_ema": True,
        "steps": 200_000,
        "batch_size": 512,
        "lr": 3e-4,
        "warmup_steps": 5_000,
        "run_name": "medium_trm",
    },
    "tiny_tx": {
        "model": "transformer",
        "embedding_dim": 64,
        "num_heads": 8,
        "num_layers": 4,
        "steps": 100_000,
        "batch_size": 512,
        "lr": 1e-3,
        "warmup_steps": 1_000,
        "run_name": "tiny_tx",
    },
    "small_tx": {
        "model": "transformer",
        "embedding_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "steps": 200_000,
        "batch_size": 512,
        "lr": 5e-4,
        "warmup_steps": 2_000,
        "run_name": "small_tx",
    },
    "medium_tx": {
        "model": "transformer",
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "steps": 500_000,
        "batch_size": 256,
        "lr": 3e-4,
        "warmup_steps": 5_000,
        "run_name": "medium_tx",
    },
    "custom": {},   # No preset — all flags must be supplied by the user.
}


def _param_count_estimate(preset: dict) -> str:
    #rough param count estimate for display
    d = preset.get("embedding_dim", 64)
    if preset.get("model") == "trm":
        # Rough: 2 * (4*d^2 + 2*(8/3)*d^2) per block + embedding
        # = ~9d^2 per layer
        n_layers = preset.get("num_layers", 2)
        approx = int(9 * d * d * n_layers + d * 32 + d * 128)
    else:
        n_layers = preset.get("num_layers", 4)
        approx = int(12 * d * d * n_layers + d * 32 + d * 128)
    if approx >= 1_000_000:
        return f"~{approx/1e6:.1f}M"
    return f"~{approx/1000:.0f}K"


def _list_presets() -> None:
    print("Available presets:\n")
    for name, cfg in PRESETS.items():
        if name == "custom":
            print(f"  {name:<14} (all flags required)")
        else:
            count = _param_count_estimate(cfg)
            model = cfg.get("model", "trm")
            d = cfg.get("embedding_dim", 64)
            steps = cfg.get("steps", 0)
            print(f"  {name:<14} {model}  d={d}  {count} params  {steps:,} steps")
    print()


def _build_argv(preset_name: str, extra_argv: list[str]) -> list[str]:
    #merge preset defaults with user-supplied CLI flags; user flags take priority
    preset = PRESETS[preset_name]

    # collect user-supplied flags so we don't override them
    user_flags: set[str] = set()
    i = 0
    while i < len(extra_argv):
        tok = extra_argv[i]
        if tok.startswith("--"):
            flag = tok.lstrip("-").replace("-", "_").split("=")[0]
            user_flags.add(flag)
        i += 1

    # build argv from preset defaults, skipping anything the user already set
    argv: list[str] = []
    for key, value in preset.items():
        if key in user_flags:
            continue    # user override takes priority
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key.replace('_', '-')}")
            # False booleans: just omit (argparse default is False)
        else:
            argv.extend([f"--{key.replace('_', '-')}", str(value)])

    argv.extend(extra_argv)
    return argv


def main() -> None:
    # first positional arg is the preset name
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        _list_presets()
        print("Run `python train.py --help` for the full flag reference.")
        return

    preset_name = sys.argv[1]

    if preset_name in ("list", "--list"):
        _list_presets()
        return

    if preset_name not in PRESETS:
        known = ", ".join(PRESETS)
        print(f"Unknown preset '{preset_name}'. Known: {known}")
        print("Use 'python main.py list' to show available presets.")
        sys.exit(1)

    extra_argv = sys.argv[2:]
    argv = _build_argv(preset_name, extra_argv)

    print(f"Preset: {preset_name}")
    print(f"Effective argv: python train.py {' '.join(argv)}\n")

    args = parse_args(argv)
    train_main(args)


if __name__ == "__main__":
    main()
