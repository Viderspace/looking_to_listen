# train_mixed.py
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
import json

from model.av_model import AudioVisualModel
from model.loss import ComplexCompressedLoss
from utils.dataset_mixed import (
    PreStagingManager,
    create_mixed_dataloader,
    AVSpeechDataset1SNoise,
    AVSpeechDataset2SClean,
    AVSpeechDataset2SNoise
)
from evaluation.metrics import sdr_improvement, si_sdr


class MixedTrainer:

    def __init__(self, config):
        self.config = config

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = AudioVisualModel().to(self.device)

        # Loss
        self.criterion = ComplexCompressedLoss()

        # Optimizer with reduced learning rate
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate']
        )

        # Learning rate scheduler
        total_steps = config['num_epochs'] * config.get('steps_per_epoch', 1000)
        self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=config.get('min_lr', 2e-5)
        )

        # Gradient clipping value
        self.grad_clip = config.get('gradient_clip', 1.0)

        # Metrics tracking
        self.metrics_history = {
                '1s_noise': [],
                '2s_clean': [],
                '2s_noise': []
        }

        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0
        self.start_epoch = 1

        # Load checkpoint if specified
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint - compatible with your old checkpoints!"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)

        # Load metrics history if available
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']

        print(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

    def train_epoch(self, epoch, dataloader):
        self.model.train()

        # Track losses per mix type
        epoch_losses = {'1s_noise': [], '2s_clean': [], '2s_noise': [], 'overall': []}

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                mixture = batch['mixture'].to(self.device)
                clean = batch['clean'].to(self.device)
                face = batch['face'].to(self.device)
                mix_types = batch['mix_type']

                # Forward pass
                self.optimizer.zero_grad()
                masks = self.model(mixture, face)
                separated = mixture * masks

                # Compute loss
                loss = self.criterion(separated, clean)

                # Track loss per mix type
                for i, mix_type in enumerate(mix_types):
                    # We can't easily separate individual losses in a batch,
                    # so we'll track the batch loss for statistics
                    epoch_losses[mix_type].append(loss.item())
                epoch_losses['overall'].append(loss.item())

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate

                # Logging
                self.global_step += 1
                current_lr = self.scheduler.get_last_lr()[0]

                # Update progress bar
                pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr'  : f'{current_lr:.2e}'
                })

                # Log to tensorboard
                if self.global_step % self.config['log_interval'] == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)

                    # Log mix type distribution
                    mix_type_counts = {mt: mix_types.count(mt) for mt in set(mix_types)}
                    for mt, count in mix_type_counts.items():
                        self.writer.add_scalar(f'train/batch_composition/{mt}',
                                               count / len(mix_types),
                                               self.global_step)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM at batch {batch_idx}. Clearing cache...")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Calculate average losses
        avg_losses = {}
        for key in epoch_losses:
            if epoch_losses[key]:
                avg_losses[key] = np.mean(epoch_losses[key])
            else:
                avg_losses[key] = float('inf')

        return avg_losses

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss'                : loss,
                'config'              : self.config,
                'global_step'         : self.global_step,
                'metrics_history'     : self.metrics_history
        }

        # Regular checkpoint
        checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Saved checkpoint: {checkpoint_path}")

        # Save as latest
        latest_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save as best if specified
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_best_2s.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ Saved as best model (2S performance)")

    def train(self, dataloader, val_dataloader=None):
        print("=" * 60)
        print(f"Starting mixed training for {self.config['num_epochs']} epochs")
        print(f"Training schedule: {self.config['schedule_phase']}")
        print(f"Mixture ratios: {self.config['probabilities']}")
        print(f"Learning rate: {self.config['learning_rate']:.2e} -> {self.config['min_lr']:.2e}")
        print("=" * 60)

        best_2s_loss = float('inf')

        for epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            # Train one epoch
            avg_losses = self.train_epoch(epoch, dataloader)

            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            for mix_type, loss in avg_losses.items():
                if mix_type != 'overall':
                    print(f"  {mix_type}: {loss:.4f}")
            print(f"  Overall: {avg_losses['overall']:.4f}")

            # Track if this is best for 2S tasks
            current_2s_loss = (avg_losses.get('2s_clean', float('inf')) +
                               avg_losses.get('2s_noise', float('inf'))) / 2
            is_best = current_2s_loss < best_2s_loss
            if is_best:
                best_2s_loss = current_2s_loss

            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, avg_losses['overall'], is_best)

            # Flush tensorboard
            self.writer.flush()

        print("\n✅ Training completed!")
        self.writer.close()