from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.engine.evaluator import validate


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.use_amp = bool(cfg.train.amp)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.output_dir = Path(cfg.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0

    def _compute_loss(self, outputs, labels):
        if self.cfg.model.mode == 'encoder_only':
            loss = self.criterion(outputs['semantic_logits'], labels)
            return {'loss_total': loss}
        return self.criterion(outputs, labels)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self._compute_loss(outputs, labels)
                loss = loss_dict['loss_total']
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))
        return running / max(1, len(self.train_loader))

    def save_checkpoint(self, epoch: int, name: str):
        path = self.output_dir / name
        torch.save(
            {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'cfg_path': self.cfg.config_path,
            },
            path,
        )

    def fit(self):
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss = self.train_one_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % int(self.cfg.eval.val_every) == 0:
                metrics = validate(self.model, self.val_loader, self.device, self.cfg)
                mean_dice = metrics['mean_dice']
                print(f'\nEpoch {epoch} | train_loss={train_loss:.6f} | val_mean_dice={mean_dice:.4f} | metrics={metrics}\n')
                self.save_checkpoint(epoch, 'last.pt')
                if mean_dice > self.best_metric:
                    self.best_metric = mean_dice
                    self.save_checkpoint(epoch, 'best.pt')
            else:
                print(f'\nEpoch {epoch} | train_loss={train_loss:.6f}\n')
                self.save_checkpoint(epoch, 'last.pt')
