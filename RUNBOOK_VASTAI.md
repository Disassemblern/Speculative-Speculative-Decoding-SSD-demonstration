# Vast.ai Runbook: Qwen 2.5 SSD benchmark

End-to-end procedure for renting a 2x A100 PCIe 40 GB box, running the AR
vs SD vs SSD benchmark, and tearing the instance down within the budget.

## 1. Instance filter

- GPUs: 2x NVIDIA A100 PCIe 40 GB (NVLink not required)
- PCIe: prefer Gen4 x16 per GPU; Gen3 x8 acceptable, x4 not
- Disk: >= 80 GB
- Host: verified, on-demand
- Image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`
- Reliability: >= 99% on host metrics
- Inet down: >= 200 Mbps (model download)

## 2. Boot sequence

```
ssh root@<host> -p <port>
git clone <repo-url> ssd_res_paper
cd ssd_res_paper
pip install -r requirements.txt
chmod +x scripts/run_all.sh
bash scripts/run_all.sh configs/qwen_a100.yaml
```

If the repo is rsynced instead of cloned, the `chmod +x` step is mandatory
(Windows hosts do not preserve the executable bit).

## 3. Expected timing

| Phase                 | Wall time |
|-----------------------|-----------|
| Image pull (if cold)  | 2-3 min   |
| Pip install           | 2-3 min   |
| Model download (cold) | 5 min     |
| Model load            | 5 min     |
| Benchmarks            | 60-75 min |
| Plots + tar           | 1 min     |
| Pull-down rsync       | 2-5 min   |

Rent for 3 hours. Total expected wall time ~90 min.

## 4. Kill-switch criteria

Stop and re-rent or downscope if any of the following hit:

- First (strategy, dataset) pair produces no `raw.jsonl` row within
  10 min of model load completion.
- Sustained VRAM > 38 GB on either GPU: reduce `--max-new-tokens` to 128,
  or tighten `--max-input-tokens` (default 2048) further.
- AR baseline < 15 tok/s on HumanEval: re-rent (likely a bad host).
- At t = 2h45m, unconditionally save partial results and tear down.

### Reproducibility

`--seed` (default 0) is the master RNG seed. `benchmark_paper.py` calls
`torch.manual_seed(seed)` at startup and re-seeds with `seed + prompt_id`
before each prompt. Re-running with the same `--seed` produces
byte-identical `output_text` fields in `raw.jsonl` (verified by the smoke
test). A single prompt can be replayed deterministically by reusing the
recorded `seed` field from the row.

### Prompt-length cap

Prompts longer than `--max-input-tokens` (default 2048) are truncated to
the last N tokens (the most recent context is kept, the start is dropped)
before being fed to any strategy. This bounds target-model VRAM on
Qwen-14B bf16 across UltraFeedback's long instructions. Each `raw.jsonl`
row records `input_token_count` (the original length) and `truncated`
(boolean), so audits can identify affected prompts.

### Run discovery

`benchmark_paper.py` writes `results/LATEST` (plain text, contains the
absolute run directory path) immediately after creating the run dir, and
re-affirms it on success. The final line of stdout on a successful run
is the run-dir path. `scripts/run_all.sh` consumes `results/LATEST` to
locate the run for plotting; partial-failure runs are still discoverable.
`scripts/run_all.sh` also tees stdout+stderr to `results/run.log`.

## 5. Pull-down

```
rsync -avz -e "ssh -p <port>" root@<host>:/workspace/ssd_res_paper/results/ ./results/
```

## 6. Teardown checklist

- [ ] rsync exited 0
- [ ] `results/run_*/raw.jsonl` is present locally and non-empty
- [ ] `results/run_*/summary.csv` is present locally
- [ ] `results/run_*/plots/*.png` are present locally
- [ ] Destroy the instance via the vast.ai UI

## 7. Smoke test on the box (optional, before long run)

```
python scripts/benchmark_paper.py \
    --config configs/qwen_a100.yaml \
    --num-prompts 2 \
    --max-new-tokens 16 \
    --strategies ar
```

If this completes in < 2 min, the full run is safe to start.
