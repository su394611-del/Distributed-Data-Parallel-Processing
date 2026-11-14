# Distributed LLM Training with PyTorch DDP and Qwen LoRA

A Runnable demonstration of **data-parallel language-model training** with PyTorch **DistributedDataParallel (DDP)**. Run a small Transformer **without downloading a model**, or fine-tune **Qwen2.5-0.5B-Instruct** with LoRA on a CUDA GPU. A standalone Jupyter notebook recreates the training script and launches it with `torchrun`.

> **Scope:** This is a training demonstration, not an LLM serving application. Multi-GPU Qwen runs are examples, not verified performance claims.

## Features

- **Tiny causal Transformer** built in PyTorch: byte-level embeddings, learned positions, causal attention, next-token loss; no Hugging Face download.
- **Qwen LoRA training:** `Qwen/Qwen2.5-0.5B-Instruct` with PEFT adapters targeting query and value projections.
- **Data parallelism:** `torchrun` launches one process per worker; `DistributedSampler` partitions examples; DDP synchronizes gradients with AllReduce.
- **CPU demonstration:** run 2 processes with Gloo, even on computers with a single GPU or no CUDA GPU.
- **Single- and multi-GPU launch:** use 1 GPU or a Linux notebook environment with at least 2 CUDA GPUs.
- **Custom corpus:** read any UTF-8 text file with one sample per line using `--corpus-file`.
- **Model saving:** rank 0 writes the checkpoint (tiny mode) or LoRA adapters (Qwen mode).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── llm_ddp_parallel_training.py
└── LLM_DDP_Parallel_Training.ipynb
```

The notebook is **self-contained**: it writes the full Python training file and requirements file to the notebook's working directory. Training runs in **subprocesses** rather than directly inside the Jupyter kernel, avoiding notebook multiprocessing / `argparse` problems.

## Hardware and prerequisites

- **Python:** 3.10+ (Python 3.12 recommended on Windows).
- **CPU mode:** any computer with a supported PyTorch installation.
- **CUDA mode:** a compatible NVIDIA driver and **CUDA-enabled PyTorch**; check with `torch.cuda.is_available()`.
- **Two GPUs:** only if two NVIDIA GPUs are available to your runtime (for example, a suitable Kaggle dual-GPU session or a dual-GPU Linux workstation).
- **Google Colab:** commonly provides **one** GPU; use `--nproc_per_node=1` unless `torch.cuda.device_count()` reports 2 or more.
- **Kaggle :** commonly provides **two** GPUs; use `--nproc_per_node=1` unless `torch.cuda.device_count()` reports 2 or more.

> DDP makes a **complete model replica on each GPU**. Two 4 GB GPUs do **not** provide an 8 GB combined model-memory pool. Larger models may require FSDP / tensor parallelism instead.

### Install (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

**For GPU training:** first install an official CUDA-enabled PyTorch wheel compatible with your hardware from <https://pytorch.org/get-started/locally/>. The `torch>=2.6` dependency alone **does not guarantee** CUDA support.

Check device availability:

```powershell
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU count:',torch.cuda.device_count())"
```

### Install in Jupyter / Colab / Kaggle

Open `LLM_DDP_Parallel_Training.ipynb`. It creates `requirements.txt` and the training script. When dependencies are missing, run the following as a **separate notebook cell**, then restart the kernel if prompted:

```python
%pip install -r requirements.txt
```

For a GPU runtime, do **not** blindly replace an already-working CUDA-enabled PyTorch installation. If needed, install other packages separately (`transformers`, `peft`, `accelerate`, `safetensors`).

## Quick start: tiny Transformer (no downloads)

From the repository directory, test 1 CPU process:

```powershell
python llm_ddp_parallel_training.py --model tiny --cpu --max-steps 2
```

Test **2 CPU processes** with real DDP and Gloo (also works without CUDA):

```powershell
python -m torch.distributed.run --standalone --nproc_per_node=2 llm_ddp_parallel_training.py --model tiny --cpu --batch-size 2 --max-steps 2
```

Expected log patterns (exact losses and sample IDs may vary):

```text
Mode=tiny, world_size=2, backend=gloo, per_device_batch=2, global_batch=4
Rank 0/2 on cpu: first batch has sample IDs [...]
Rank 1/2 on cpu: first batch has sample IDs [...]
epoch=1 step=1 mean_loss=...
Tiny Transformer saved at: .../tiny_transformer.pt
```

## Qwen2.5-0.5B LoRA fine-tuning

Requires downloading `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face. It uses 0.5B-scale pretrained parameters and only trains a small number of LoRA parameters. It **does not perform QLoRA/4-bit quantization**.

### Single NVIDIA GPU (Colab, Kaggle, consumer desktop)

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 llm_ddp_parallel_training.py --model qwen --batch-size 1 --seq-len 64 --max-steps 2
```

### Two NVIDIA GPUs (Linux/Kaggle dual-GPU runtime)

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 llm_ddp_parallel_training.py --model qwen --batch-size 1 --seq-len 64 --max-steps 2
```

Check the actual number of GPUs **before** launching multiple processes:

```python
import torch
print(torch.cuda.device_count())
```

Only choose `--nproc_per_node=2` if your runtime reports at least 2 CUDA GPUs. On Windows, the script uses the **Gloo** backend; **NCCL is not available on Windows** in standard PyTorch builds. Linux CUDA installations use NCCL. GPU runs on Windows may be less performant than Linux. PyTorch's distributed Windows support remains more limited than on Linux.

### Custom training data

Create `corpus.txt` with **one UTF-8 text sample per line**. Then run:

```bash
python llm_ddp_parallel_training.py --model tiny --cpu --corpus-file corpus.txt --repeat 1 --max-steps 4
```

This script's included text corpus is pedagogical. The Qwen mode applies next-token loss to the entire input text. It is **not** a supervised chat instruction dataset with response-only masking or a validation split; use a properly designed dataset and masking strategy for serious fine-tuning.

## CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `tiny` | `tiny` (from scratch) or `qwen` (LoRA) |
| `--cpu` | off | Force CPU; suitable for DDP demonstrations |
| `--corpus-file` | built-in | Optional one-example-per-line UTF-8 dataset |
| `--batch-size` | `2` | Microbatch size **per rank** |
| `--seq-len` | `64` | Maximum tokens/bytes per sample |
| `--epochs` | `1` | Epoch count |
| `--max-steps` | `4` | Maximum optimizer steps per rank per epoch; `0` = all |
| `--lr` | mode-specific | Learning-rate override |
| `--repeat` | `4` | Repeat the demonstration corpus |
| `--save-dir` | `ddp_example_output` | Model/adapters output location |

The **global batch size** in this example is `batch_size × world_size`. No gradient accumulation is implemented.

## How DDP works

1. `torchrun` launches N processes with environment variables `WORLD_SIZE`, `RANK,` and `LOCAL_RANK`.
2. Each rank initializes a process group (Gloo for CPU/Windows, NCCL for Linux CUDA).
3. `DistributedSampler` gives each rank a different partition of the input examples.
4. Each rank computes a local forward pass, loss, and backward pass.
5. DDP synchronizes model gradients across ranks before the optimizer update.
6. Rank 0 saves the checkpoint or the adapter; all ranks shut down the group cleanly.

## Verification and limitations

Verified in a local **PyTorch 2.10.0+cpu** runtime:

- Python syntax compilation: **passed**.
- Tiny Transformer, 1 CPU process, 2 steps: **passed**, checkpoint created.
- Tiny Transformer, 2 CPU processes, 2 steps, Gloo backend: **passed**, checkpoint created.

**Not executed here:** Qwen model downloads, single-GPU CUDA training, and physical multi-GPU training. No production benchmark, quality metric, model-card evaluation or memory guarantee is claimed.

**Potential limitations:** small corpus, no evaluation split, no robust fault tolerance, no accumulation, possible GPU out-of-memory errors, basic LoRA configuration, and low-speed Gloo on Windows. For production training, introduce proper validation, data pipelines, observability, mixed-precision/autocast testing, reproducibility controls, and sharded training as needed.

## Troubleshooting

- **`CUDA available: False`:** `nvidia-smi` can detect a GPU even when your current PyTorch build is CPU-only (e.g., `+cpu`). Install the correct CUDA-enabled wheel in the active environment.
- **`LOCAL_RANK` exceeds GPU count:** do not launch more ranks than CUDA GPUs; use CPU mode to demonstrate multiple ranks.
- **Notebook `argparse` errors:** don't run the CLI source with `%run` from `ipykernel`; use the provided notebook's `subprocess` launch cells.
- **Out of memory:** start with `--batch-size 1 --seq-len 64`; fewer steps do not necessarily reduce peak memory usage.
- **Qwen dependency errors:** ensure `transformers`, `peft`, `accelerate`, and `safetensors` are in the same Python environment; use `python -m pip check`.
- **Repeated model downloads:** configure Hugging Face cache paths, ensure internet access and sufficient disk space.

## References

- [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch Distributed Communication Backends](https://docs.pytorch.org/docs/stable/distributed.html)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)

## License

No license has been added. If publishing as an open-source project, choose and add a suitable `LICENSE` file, considering the pretrained model's own license and dataset permissions.
