import copy
import json
import math  # added for cosine annealing
import os
import time
from typing import Optional
import glob
import re
import psutil
import torch
import torch.distributed
import torch._dynamo
import h5py
import numpy as np
from era5_videogen.training.loss import DiffusionLoss
from era5_videogen.utils import io, stats

class Trainer:
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        total_kimg: int = 200000,  # n steps, measured in thousands of training images.
        ema_halflife_kimg: int = 500,  # half-life of EMA of model weights.
        ema_rampup_ratio: float = 0.05,  # EMA ramp-up coefficient, None = disable.
        lr_rampup_kimg: int = 10000,  # n learning rate ramp-up.
        lr_min_factor: float = 0.01,  # min factor relative to lr [0,1].
        kimg_per_tick: int = 50,  # interval of progress prints.
        checkpoint_ticks: int = 50,  # n dump state, None = disable.
        validation_ticks: Optional[int] = None,  # n ticks between validations, None = disable.
        resume_kimg=0,
        device: Optional[str | torch.device] = None,
        amp_type: Optional[str] = "bfloat16",  # None (float32), "bfloat16", or "float16"
        compile: bool = False,
        checkpoint_dir = False,
        conditional : bool = False,
        reset_lr: Optional[float] = None,  # If set, reset LR to this value when loading checkpoint (ignores saved LR)
        gradient_accumulation_steps: int = 1,  # Number of batches to accumulate gradients over
        grad_clip: Optional[float] = 1.0,  # max global grad norm; None disables clipping
    ):
        # Select device based on availability and local rank.
        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device
        
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        # Move the network to the device before wrapping in DDP.
        net = net.to(self.device)
        self.net = net
        # Only print the model summary on rank 0
        if io.get_rank() == 0:
            io.log0("Model summary:")
            io.log0(str(self.net))

        self.enable_amp = amp_type is not None
        self.amp_type = (
            torch.bfloat16
            if amp_type == "bfloat16"
            else torch.float16 if amp_type == "float16" else None
        )
        self.scaler = torch.GradScaler(
            self.device.type,
            enabled=(amp_type == "float16" and torch.cuda.is_available()),
        )
        # Compile the inner module BEFORE DDP-wrapping. Compiling the DDP
        # wrapper engages Dynamo's DDPOptimizer, which cannot split graphs
        # containing the higher-order op emitted by non-reentrant activation
        # checkpointing (pytorch#104674) and crashes multi-node runs.
        # optimize_ddp=False additionally guards any remaining
        # compile-of-DDP path. self.net stays bound to the raw module, so
        # checkpoints and the EMA copy keep clean (un-prefixed) state dicts.
        torch._dynamo.config.optimize_ddp = False
        net_fwd = net
        if compile and hasattr(torch, "compile") and torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability(self.device)
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                # mode="default" (not "max-autotune"): autotune benchmarking
                # allocates large workspaces and tipped 40 GiB a100s into OOM
                # on the first compiled step (2026-07-03).
                net_fwd = torch.compile(net, fullgraph=False)
            else:
                if io.get_rank() == 0:
                    io.log0(
                        "GPU is not NVIDIA V100, A100, or H100. Speedup may be lower than expected.",
                        mode="warning",
                    )

        # Wrap with DDP if the process group is initialized (parameters are
        # shared between net and the compiled wrapper, so DDP hooks are
        # registered on the same tensors either way).
        if torch.distributed.is_initialized():
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                net_fwd,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                static_graph=False,
            )
        else:
            self.ddp = net_fwd

        # EMA init: create a copy of the net (on the same device) for exponential moving average.
        self.ema = copy.deepcopy(net).eval().requires_grad_(False)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_rampup_kimg = lr_rampup_kimg
        self.lr_min_factor = lr_min_factor
        self.base_lr = [g["lr"] for g in optimizer.param_groups]

        # bookkeeping
        self.total_kimg = total_kimg
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.kimg_per_tick = kimg_per_tick
        self.checkpoint_ticks = checkpoint_ticks
        self.validation_ticks = validation_ticks
        self.resume_kimg = resume_kimg
        self.checkpoint_dir = checkpoint_dir
        self.reset_lr = reset_lr
        self.loaded_from_checkpoint = False
        if self.checkpoint_dir:
            loaded_nimg, _ = self.load_checkpoint(self.checkpoint_dir)
            if loaded_nimg is None:
                self.cur_nimg = 0
            else:
                self.cur_nimg = loaded_nimg
                self.loaded_from_checkpoint = True
        else:
            self.cur_nimg = 0
        self.conditional = conditional
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        if io.get_rank() == 0:
            io.log0(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            
    def train(self, dataloader, val_dataloader=None):
        rank = io.get_rank()
        if rank == 0:
            io.log0(f"Training for {self.total_kimg} kimg...")
        
        # Run validation at start if validation is enabled
        if self.validation_ticks is not None:
            val_metrics = None
            if val_dataloader is not None:
                if rank == 0:
                    io.log0("Running validation at start of training...")
                val_metrics = self.validate(val_dataloader)
            # Report on ALL ranks (val_dataloader is None on non-zero ranks) so the
            # distributed stats collector registers the same counter names on every
            # rank; report0 keeps only rank 0's value. A name set that differs across
            # ranks desyncs stats.update()'s all_reduce -> NCCL collective-size timeout.
            stats.report0(
                "Validation/avg_error",
                val_metrics.get("avg_error", 0.0) if val_metrics else 0.0,
            )
            stats.report0(
                "Validation/avg_error_denorm",
                val_metrics.get("avg_error_denorm", 0.0) if val_metrics else 0.0,
            )
            # Synchronize all ranks after validation (ensure all ranks wait for validation to complete)
            # Barrier must be outside the val_dataloader check so all ranks participate
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        cur_nimg = self.cur_nimg#self.resume_kimg * 1000
        cur_tick = int(cur_nimg // (self.kimg_per_tick * 1000))
        tick_start_nimg = cur_nimg
        tick_start_time = start_time = time.time()
        maintenance_time = 0
        stats_jsonl = None
        accumulation_step = 0  # Track current step in accumulation cycle
        # On-device accumulator: per-step losses are summed here without a host
        # sync, then reduced across ranks once per tick (not once per step).
        tick_loss_sum = torch.zeros((), device=self.device)
        tick_loss_count = 0

        try:
            while True:
                # Zero gradients only at the start of each accumulation cycle
                if accumulation_step == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                data_start_time = time.time()
                if self.conditional:
                    X, T = next(dataloader)
                    X = X.to(self.device,dtype=torch.float32, non_blocking=True)
                    T = T.to(self.device,dtype=torch.float32, non_blocking=True)
                else: 
                    T = next(dataloader)
                    T = T.to(self.device,dtype=torch.float32, non_blocking=True)
                    X = None
                data_time = time.time() - data_start_time

                with torch.autocast(
                    self.device.type, enabled=self.enable_amp, dtype=self.amp_type
                ):
                    if isinstance(self.loss_fn, DiffusionLoss):
                        loss = self.loss_fn(self.ddp, T, condition=X)
                    else:  # e.g., torch MSELoss
                        loss = self.loss_fn(self.ddp(X), T)

                # Scale loss by accumulation steps to get average
                loss = loss / self.gradient_accumulation_steps

                # Update learning rate
                warmup_nimg = self.lr_rampup_kimg * 1000
                if cur_nimg < warmup_nimg:  # linear warmup
                    progress = cur_nimg / warmup_nimg
                    for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                        min_lr = base_lr * self.lr_min_factor
                        g["lr"] = min_lr + (base_lr - min_lr) * progress
                else:  # cosine annealing
                    progress_cos = min(
                        1.0,
                        (cur_nimg - warmup_nimg) / (self.total_kimg * 1000 - warmup_nimg),
                    )
                    for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                        min_lr = base_lr * self.lr_min_factor
                        g["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress_cos))
                
                self.scaler.scale(loss).backward()  # no-op for bfloat16/None amp_type

                wld_size = io.get_world_size()
                # Accumulate the LOCAL loss on-device only. The old code reduced
                # across ranks and synced to host (.item()) every step, putting a
                # blocking collective + device->host sync on the critical path of
                # every step; we now do that once per tick instead.
                tick_loss_sum += loss.detach() * self.gradient_accumulation_steps
                tick_loss_count += 1

                accumulation_step += 1
                batch_size = T.shape[0] * wld_size
                cur_nimg += batch_size  # Update cur_nimg per batch for LR scheduling

                # Only step optimizer and update EMA at the end of accumulation cycle
                if accumulation_step >= self.gradient_accumulation_steps:
                    self.scaler.unscale_(self.optimizer)

                    # NaN/inf guard (bf16 safety) followed by global grad-norm clipping.
                    for param in self.net.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(
                                param.grad,
                                nan=0,
                                posinf=1e5,
                                neginf=-1e5,
                                out=param.grad,
                            )

                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), max_norm=self.grad_clip
                        )
                        stats.report0("grad_norm", float(grad_norm))

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # EMA update with effective batch size (accumulated)
                    total_batch_size = batch_size * self.gradient_accumulation_steps
                    ema_halflife_nimg = self.ema_halflife_kimg * 1000
                    if self.ema_rampup_ratio is not None:
                        ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)
                    ema_beta = 0.5 ** (total_batch_size / max(ema_halflife_nimg, 1e-8))
                    for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
                        p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

                    accumulation_step = 0  # Reset accumulation counter
                
                total_batch_size = batch_size  # For progress tracking 
                done = cur_nimg >= self.total_kimg * 1000

                if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + self.kimg_per_tick * 1000):
                    continue

                tick_end_time = time.time()
                if torch.cuda.is_available():
                    peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 2**30
                    reserved_mem_gb = torch.cuda.max_memory_reserved(self.device) / 2**30
                else:
                    peak_mem_gb = reserved_mem_gb = 0
                # Reduce the per-tick loss across ranks ONCE (replaces the former
                # per-step all_reduce + .item() on the training critical path).
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        tick_loss_sum, op=torch.distributed.ReduceOp.SUM
                    )
                    tick_loss_mean = (
                        tick_loss_sum / max(tick_loss_count * wld_size, 1)
                    ).item()
                else:
                    tick_loss_mean = (tick_loss_sum / max(tick_loss_count, 1)).item()
                fields = [
                    f'tick {stats.report0("Progress/tick", cur_tick):<5d}',
                    f'lr {g["lr"]:<1.7f}',
                    f'loss {stats.report("Loss", tick_loss_mean):<2.4f}',
                    f'kimg {stats.report0("Progress/kimg", cur_nimg / 1e3):<9.2f}',
                    f'time {stats.format_time(stats.report0("total_T", tick_end_time - start_time)):<12s}',
                    f'sec/tick {stats.report0("sec_per_tick", tick_end_time - tick_start_time):<7.1f}',
                    f'sec/kimg {stats.report0("sec_per_kimg", (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}',
                    f'data_T {stats.report0("data_T", data_time):<7.2f}',
                    f'maintenance_T {stats.report0("maintenance_T", maintenance_time):<6.1f}',
                    f'cpu_mem_gb {stats.report0("cpu_mem_gb", psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}',
                    f'gpu_mem_gb {stats.report0("peak_gpu_mem_gb", peak_mem_gb):<6.2f}',
                    f'reserved_mem_gb {stats.report0("peak_gpu_mem_reserved_gb", reserved_mem_gb):<6.2f}',
                ]
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                if rank == 0:
                    io.log0(" ".join(fields))

                if (
                    (self.checkpoint_ticks is not None)
                    and (done or cur_tick % self.checkpoint_ticks == 0)
                    and cur_tick != 0
                    and rank == 0
                ):
                    self._save_checkpoint(cur_nimg)

                # Run validation if enabled and it's time
                if (
                    (self.validation_ticks is not None)
                    and (cur_tick % self.validation_ticks == 0)
                    and cur_tick != 0
                ):
                    val_metrics = None
                    if val_dataloader is not None:
                        val_metrics = self.validate(val_dataloader)
                    # Report on ALL ranks (see start-of-training note) so the stats
                    # collector keeps a consistent set of counter names across ranks.
                    stats.report0(
                        "Validation/avg_error",
                        val_metrics.get("avg_error", 0.0) if val_metrics else 0.0,
                    )
                    stats.report0(
                        "Validation/avg_error_denorm",
                        val_metrics.get("avg_error_denorm", 0.0) if val_metrics else 0.0,
                    )
                    # Synchronize all ranks after validation (ensure all ranks wait for validation to complete)
                    # Barrier must be outside the val_dataloader check so all ranks participate
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()

                if torch.distributed.is_initialized():
                    stats.default_collector.update()

                if rank == 0:
                    stats_jsonl = self._update_stats_log(stats_jsonl)

                cur_tick += 1
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()
                maintenance_time = tick_start_time - tick_end_time
                tick_loss_sum = torch.zeros((), device=self.device)
                tick_loss_count = 0
                if done:
                    break
        finally:
            if stats_jsonl is not None:
                stats_jsonl.close()

        if rank == 0:
            io.log0("Exiting...")

    @torch.no_grad()
    def validate(self, val_dataloader):
        """
        Run validation on rank 0 using the same batch structure as training.
        """
        rank = io.get_rank()
        if rank != 0:
            return None

        net_eval = self.ema.eval()
        total_loss = 0.0
        num_batches = len(val_dataloader)
        io.log0("-------------Computing Validation Metrics------------")

        for batch_idx, batch in enumerate(val_dataloader):
            if isinstance(batch, (tuple, list)):
                X = batch[0].to(self.device, dtype=torch.float32, non_blocking=True)
                T = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
            else:
                X = None
                T = batch.to(self.device, dtype=torch.float32, non_blocking=True)

            # Match the training precision (bf16 autocast). Running validation in
            # FP32 doubled activation memory and, combined with the compiled
            # training graph's reserved CUDA-graph pools, OOM'd on rank 0; bf16
            # also makes the full-val pass ~2x faster.
            with torch.autocast(
                self.device.type, enabled=self.enable_amp, dtype=self.amp_type
            ):
                if isinstance(self.loss_fn, DiffusionLoss):
                    batch_loss = self.loss_fn(net_eval, T, condition=X)
                else:
                    batch_loss = self.loss_fn(net_eval(X), T)

            total_loss += batch_loss.item()

            if (batch_idx + 1) % 50 == 0:
                io.log0(f"batch {batch_idx + 1}/{num_batches}")

        if num_batches == 0:
            io.log0("Warning: No validation batches processed")
            return None

        avg_loss = total_loss / num_batches
        io.log0(f"Validation loss: {avg_loss:.6f}")

        return {
            "avg_error": avg_loss,
            "avg_error_denorm": avg_loss,
            "num_batches": num_batches,
        }
      
            

    def _save_checkpoint(self, cur_nimg):
        rank = io.get_rank()
        if rank != 0:
            return
        state = {
            "ema": self.ema.state_dict(),
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "cur_nimg": cur_nimg,
        }
        path = self.checkpoint_dir or os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"checkpoint-{cur_nimg // 1000:06d}.pt")
        torch.save(state, checkpoint_path)
        io.log0(f"Checkpoint saved at {checkpoint_path}")

    def _update_stats_log(self, stats_jsonl):
        rank = io.get_rank()
        if rank != 0:
            return stats_jsonl
        if stats_jsonl is None:
            stats_jsonl = open(os.path.join(os.getcwd(), "stats.jsonl"), "at")
        stats_jsonl.write(
            json.dumps({**stats.default_collector.as_dict(), "timestamp": time.time()})
            + "\n"
        )
        stats_jsonl.flush()
        return stats_jsonl

    def load_checkpoint(self,checkpoint_dir):
        """
        Searches the "checkpoints" directory for the latest checkpoint file,
        loads its state (including EMA, network, optimizer, and scaler),
        updates the trainer's resume progress (self.resume_kimg), and returns a tuple:
            (cur_nimg, state)
        where cur_nimg is the absolute number of images processed.
        If no checkpoint is found, returns (None, None).
        """
        rank = io.get_rank()
        # checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(checkpoint_dir):
            if rank == 0:
                io.log0(f"No checkpoint directory found at {checkpoint_dir}")
            return None, None

        # Look for all checkpoint files matching the pattern.
        pattern = os.path.join(checkpoint_dir, "checkpoint-*.pt")
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            if rank == 0:
                io.log0("No checkpoint found.")
            return None, None

        # Extract the checkpoint number from the filename.
        def extract_ckpt_number(fname):
            m = re.search(r"checkpoint-(\d{6})\.pt", os.path.basename(fname))
            return int(m.group(1)) if m else -1

        # Build a list of (filename, checkpoint_number) and filter invalid ones.
        ckpt_files = [(fname, extract_ckpt_number(fname)) for fname in ckpt_files]
        ckpt_files = [item for item in ckpt_files if item[1] >= 0]
        if not ckpt_files:
            if rank == 0:
                io.log0("No valid checkpoint found.")
            return None, None

        # Sort by checkpoint number in descending order and select the latest.
        ckpt_files.sort(key=lambda x: x[1], reverse=True)
        latest_ckpt, ckpt_number = ckpt_files[0]
        checkpoint_path = latest_ckpt
        # The checkpoint filename encodes cur_nimg//1000, so compute cur_nimg:
        cur_nimg = ckpt_number * 1000

        # Load the checkpoint.
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if rank == 0:
            io.log0(f"Loaded checkpoint from {checkpoint_path}")

        # Restore EMA state.
        self.ema.load_state_dict(state["ema"])
        # Load into the RAW module (self.net): checkpoints are saved from it
        # with clean (un-prefixed) keys, and it shares parameters with both the
        # compiled wrapper (OptimizedModule, whose own state_dict keys carry an
        # '_orig_mod.' prefix) and the DDP wrapper, so this updates all views.
        self.net.load_state_dict(state["net"])
        # Restore optimizer and scaler states.
        self.optimizer.load_state_dict(state["optimizer"])
        
        # Reset LR if reset_lr is specified, otherwise keep the saved LR
        if self.reset_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.reset_lr
            if rank == 0:
                io.log0(f"LR reset to {self.reset_lr} (ignoring saved LR state)")
        self.scaler.load_state_dict(state["scaler"])

        # If the checkpoint state has a "cur_nimg" key, override the value.
        if "cur_nimg" in state:
            cur_nimg = state["cur_nimg"]

        # Update the trainer's resume progress (in thousands).
        self.resume_kimg = cur_nimg // 1000
        if rank == 0:
            io.log0(f"Resuming training at {self.resume_kimg} kimg.")

        return cur_nimg, state
