#!/usr/bin/env python3
"""Single/multi-GPU data-parallel language-model training demo.

Modes
-----
A) PyTorch-only runnable causal Transformer, no downloads:
   python -m torch.distributed.run --standalone --nproc_per_node=2 llm_ddp_parallel_training.py --model tiny --max-steps 2

B) Real pretrained Qwen 0.5B LoRA tuning (requires transformers, peft):
   python -m torch.distributed.run --standalone --nproc_per_node=2 llm_ddp_parallel_training.py --model qwen --max-steps 8

If only one GPU is available, use --nproc_per_node=1 (or plain python).
DistributedSampler divides examples; DDP synchronizes gradients. DDP does not pool GPU memory.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Examples are illustrative public-domain-style sentences, not a performance benchmark.
EXAMPLES = [
    "Transformers use self attention to model the relationships between tokens.",
    "Distributed data parallel training places a model replica on each GPU.",
    "A distributed sampler sends different training records to each rank.",
    "Gradient all reduce synchronizes model updates across the workers.",
    "Causal language models predict the next token in a sequence.",
    "LoRA fine tuning updates small low rank adaptation matrices.",
    "The optimizer applies parameter updates after gradient computation.",
    "GPU memory usage grows with batch size and sequence length.",
    "The tokenizer transforms human readable strings into integer tokens.",
    "A CUDA GPU can accelerate matrix multiplications and attention.",
    "The validation split evaluates generalization to unseen examples.",
    "A reproducible experiment reports its model and dataset versions.",
    "The attention mask helps distinguish data tokens from padding.",
    "Models trained in parallel still need the weights to fit per device.",
    "Sharded training uses different strategies from data parallel training.",
    "The training loss shows how well next token prediction is learned.",
]

class IndexedTextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, index: int) -> tuple[int, str]:
        return index, self.texts[index]


class TinyCausalTransformer(nn.Module):
    """A small causal Transformer built from scratch for offline testing, not an LLM."""
    def __init__(self, seq_len: int, width: int = 64, heads: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(258, width)  # PAD=0, EOS=1, byte+2
        self.pos_emb = nn.Embedding(seq_len, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 258)
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        _, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        hidden = self.token_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).triu(1)
        hidden = self.encoder(hidden, mask=causal_mask)
        logits = self.lm_head(self.norm(hidden))
        # Predict next token. Padded label positions are ignored.
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1), ignore_index=-100,
        )


def tiny_collator(items: list[tuple[int, str]], seq_len: int) -> dict:
    ids, texts = zip(*items)
    sequences = []
    for text in texts:
        tokens = [b + 2 for b in text.encode("utf-8")[:seq_len - 1]] + [1]
        sequences.append(tokens + [0] * (seq_len - len(tokens)))
    input_ids = torch.tensor(sequences, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == 0] = -100
    return {"input_ids": input_ids, "labels": labels, "record_ids": list(ids)}


def main():
    parser = argparse.ArgumentParser(description="DDP language-model example", allow_abbrev=False)
    parser.add_argument("--model", choices=["tiny", "qwen"], default="tiny")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training, useful for learning 2-rank DDP without GPUs")
    parser.add_argument("--corpus-file", type=Path, default=None,
                        help="Optional UTF-8 text file: one example per line")
    parser.add_argument("--batch-size", type=int, default=2, help="Microbatch size per GPU")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4, help="Optimizer updates per rank per epoch; 0 means all")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=4, help="Repeat demonstration corpus")
    parser.add_argument("--save-dir", type=Path, default=Path("ddp_example_output"))
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.epochs, args.repeat) < 1 or args.max_steps < 0:
        parser.error("batch-size, seq-len, epochs, repeat must be positive; max-steps cannot be negative")
    if args.seq_len < 4:
        parser.error("seq-len must be at least 4")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available() and not args.cpu
    if cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"Local rank {local_rank} but only {torch.cuda.device_count()} GPUs visible")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(min(2, os.cpu_count() or 1))

    if world > 1:
        backend = "gloo" if os.name == "nt" or not cuda else "nccl"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=10))
    else:
        backend = "none"

    try:
        torch.manual_seed(2026)
        if cuda:
            torch.cuda.manual_seed_all(2026)
        if args.corpus_file:
            texts = [x.strip() for x in args.corpus_file.read_text(encoding="utf-8").splitlines() if x.strip()]
            if not texts:
                raise ValueError("No nonempty lines in corpus file")
        else:
            texts = EXAMPLES
        texts = texts * args.repeat
        data = IndexedTextDataset(texts)
        if len(data) < world:
            raise ValueError(f"At least {world} examples are needed")

        if args.model == "tiny":
            model = TinyCausalTransformer(seq_len=args.seq_len)
            collate_fn = lambda items: tiny_collator(items, args.seq_len)
            lr = args.lr if args.lr is not None else 3e-4
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise SystemExit("Qwen mode requires: pip install transformers peft accelerate") from exc
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            def collate_fn(items):
                ids, batch_text = zip(*items)
                batch = tokenizer(list(batch_text), truncation=True, padding="max_length",
                                  max_length=args.seq_len, return_tensors="pt")
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100
                return {"input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "labels": labels, "record_ids": list(ids)}
            dtype = torch.float16 if cuda else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            model.config.use_cache = False
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
            ))
            lr = args.lr if args.lr is not None else 2e-4

        model = model.to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"Mode={args.model}, world_size={world}, backend={backend}, per_device_batch={args.batch_size}, "
                  f"global_batch={args.batch_size * world}", flush=True)
            print(f"Model trainable parameters: {trainable:,}", flush=True)

        model_ddp = (DDP(model, device_ids=[local_rank], output_device=local_rank)
                     if cuda and world > 1 else DDP(model) if world > 1 else model)
        sampler = DistributedSampler(data, num_replicas=world, rank=rank,
                                     shuffle=True, seed=2026) if world > 1 else None
        loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                            shuffle=sampler is None, collate_fn=collate_fn, num_workers=0)
        optimizer = torch.optim.AdamW([p for p in model_ddp.parameters() if p.requires_grad], lr=lr)
        use_scaler = cuda and args.model == "qwen"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model_ddp.train()
            for step, batch in enumerate(loader, start=1):
                if args.max_steps and step > args.max_steps:
                    break
                if step == 1:
                    print(f"Rank {rank}/{world} on {device}: first batch has sample IDs {batch['record_ids']}",
                          flush=True)
                inp = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if args.model == "qwen":
                    mask = batch["attention_mask"].to(device, non_blocking=True)
                    loss = model_ddp(input_ids=inp, attention_mask=mask, labels=labels).loss
                else:
                    loss = model_ddp(input_ids=inp, labels=labels)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), 1.0)
                    optimizer.step()
                # Report the mean over ranks rather than just GPU 0's local loss.
                reduced = loss.detach().float().clone()
                if world > 1:
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    reduced /= world
                if rank == 0:
                    print(f"epoch={epoch+1} step={step} mean_loss={reduced.item():.4f}", flush=True)
        if rank == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            if args.model == "qwen":
                model.save_pretrained(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
                print(f"LoRA adapter saved at: {args.save_dir.resolve()}", flush=True)
            else:
                torch.save({"model_state_dict": model.state_dict(),
                            "seq_len": args.seq_len, "width": 64, "heads": 4},
                           args.save_dir / "tiny_transformer.pt")
                print(f"Tiny Transformer saved at: {(args.save_dir / 'tiny_transformer.pt').resolve()}", flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
