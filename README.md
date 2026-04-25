# The Power of Recursive Transformers: A Case Study in Chess

We investigate whether recursive shared-weight Transformers can match standard untied Transformers for searchless chess evaluation. We adapt the **Tiny Recursive Model (TRM)** (Jolicoeur-Martineau 2025) to the Searchless Chess setting and compare it against baseline Transformer models across multiple scales. Both model families predict Stockfish win-probability buckets from tokenised FEN positions.

> *Our results show that all trained models learn meaningful chess representations, but standard Transformers consistently outperform the recursive models at comparable scales.*

## Model Configurations

| Preset | Architecture | d | L | r | Params |
|--------|-------------|---|---|---|--------|
| `tiny_trm` | TRM | 64 | 2 | 8 | ~142K |
| `small_trm` | TRM | 128 | 2 | 8 | ~500K |
| `medium_trm` | TRM | 256 | 2 | 16 | ~2M |
| `tiny_tx` | Transformer | 64 | 4 | — | ~274K |
| `small_tx` | Transformer | 128 | 6 | — | ~1M |
| `medium_tx` | Transformer | 256 | 8 | — | ~4M |

The TRM has roughly half the unique parameters of a comparably sized Transformer at each scale, since its block is shared across all recurrence steps. However, effective compute (FLOPs per forward pass) is comparable, because each recurrence step re-executes the shared block.

## Running Experiments

**Recommended: full automated run**
```bash
/c/Python312/python run_experiment.py --data bcTrain.bag --val-data bcTest.bag
```

This trains all model pairs, runs arena matches, and evaluates puzzle accuracy unattended. Results are logged to `results/experiment_log.txt`.

**Preset run (single model)**
```bash
/c/Python312/python main.py tiny_trm --data bcTrain.bag
/c/Python312/python main.py small_tx --data bcTrain.bag
```

**Custom training**
```bash
/c/Python312/python train.py \
    --model trm --data bcTrain.bag \
    --embedding-dim 128 --num-heads 8 --num-layers 2 --n-recurrence 8 \
    --batch-size 512 --lr 5e-4 --steps 200000 \
    --deep-supervision --use-ema \
    --run-name my_run
```

**Arena (head-to-head)**
```bash
/c/Python312/python arena.py \
    --model-a results/small_trm/step_0200000_final.pt \
    --model-b results/small_tx/step_0200000_final.pt \
    --games 100
```

**Puzzle benchmark**
```bash
/c/Python312/python arena.py \
    --model-a results/small_tx/step_0200000_final.pt \
    --model-b random \
    --puzzles puzzles.csv --num-puzzles 1000
```

## Key Results

| Model | Train Acc. (%) | Puzzle Acc. (%) | Arena |
|-------|---------------|----------------|-------|
| `small_trm` | 13.61 | 12.8 | Lost to `small_tx` |
| `small_tx` | 16.31 | 21.6 | **Strongest arena model** |
| `medium_trm` | 14.67 | 15.4 | Better than small TRM |
| `medium_tx` | 16.23 | 23.6 | **Best puzzle model** |

The strongest puzzle model was `medium_tx` (23.6% accuracy), but the strongest arena model was `small_tx`, which defeated `medium_tx` head-to-head. Puzzle accuracy and arena strength are positively related, but not perfectly aligned.

## Requirements

```bash
pip install torch etils python-chess
```

Assumes that the dataset from (Amortized Planning with Large-Scale Transformers: A Case Study on Chess)[https://github.com/google-deepmind/searchless_chess] is present in the working directory.
