# train.py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader

from model.av_model import AudioVisualModel
from utils.dataset import create_dataloader
from model.loss import ComplexCompressedLoss


class Trainer:

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

        # Loss and optimizer
        self.criterion = ComplexCompressedLoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate']
        )

        # Create dataloaders - check if using GCS or local
        if 'gcs_helper' in config and config.get('use_gcs', False):
            print("Using GCS-aware dataloader...")
            # Import and use GCS dataset
            from utils.gcs_dataset import GCSAVSpeechDataset

            train_dataset = GCSAVSpeechDataset(
                    sample_list=config['sample_list'],
                    gcs_helper=config['gcs_helper']
            )

            self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'],
                    pin_memory=True,
                    persistent_workers=True if config['num_workers'] > 0 else False
            )
        else:
            print("Using local file dataloader...")
            self.train_loader = create_dataloader(
                    config['train_dir'],
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers']
            )

        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0
        self.start_epoch = 1

        # Resume from checkpoint if specified
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                mixture = batch['mixture'].to(self.device)
                clean = batch['clean'].to(self.device)
                face = batch['face'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                masks = self.model(mixture, face)
                separated = mixture * masks

                # Compute loss
                loss = self.criterion(separated, clean)

                # Backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                # Logging
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Log to tensorboard every N steps
                if self.global_step % self.config['log_interval'] == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate',
                                           self.optimizer.param_groups[0]['lr'],
                                           self.global_step)

                # Save checkpoint at intervals
                if self.global_step % self.config['save_interval'] == 0:
                    avg_loss = epoch_loss / num_batches
                    self.save_checkpoint(epoch, avg_loss, is_interval=True)

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                if 'sample_id' in batch:
                    print(f"Sample IDs: {batch['sample_id']}")
                continue

        return epoch_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, loss, is_interval=False):
        checkpoint = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss'                : loss,
                'config'              : self.config,
                'global_step'         : self.global_step
        }

        # Save both interval and epoch checkpoints
        if is_interval:
            checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_step_{self.global_step}.pt'
            )
        else:
            checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch}.pt'
            )

        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved checkpoint: {checkpoint_path}")

        # Also save as 'latest' for easy resuming
        latest_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

    def train(self):
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Total batches per epoch: {len(self.train_loader)}")
        print(f"Starting from epoch: {self.start_epoch}")

        for epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, avg_loss, is_interval=False)

            # Flush tensorboard
            self.writer.flush()

        print("Training completed!")
        self.writer.close()

    # test_step and profile_forward_pass methods remain the same...
    def test_step(self):
        """Run a single training step for testing"""
        self.model.train()

        # Get one batch
        batch = next(iter(self.train_loader))

        # Move to device
        mixture = batch['mixture'].to(self.device)
        clean = batch['clean'].to(self.device)
        face = batch['face'].to(self.device)

        # Time the forward pass
        import time

        start = time.time()

        # Forward pass
        masks = self.model(mixture, face)
        separated = mixture * masks
        loss = self.criterion(separated, clean)

        # Backward pass
        loss.backward()

        end = time.time()

        print(f"Single batch test:")
        print(f"  Batch size: {mixture.shape[0]}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Time: {end - start:.3f}s")
        print(f"  Device: {self.device}")

    def profile_forward_pass(self):
        """Profile each component of the forward pass"""
        import time

        self.model.eval()
        batch = next(iter(self.train_loader))

        mixture = batch['mixture'].to(self.device)
        clean = batch['clean'].to(self.device)
        face = batch['face'].to(self.device)

        with torch.no_grad():
            # Profile each component
            times = {}

            # Audio CNN
            start = time.time()
            audio_features = self.model.audio_cnn(mixture)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['audio_cnn'] = time.time() - start

            # Visual CNN
            start = time.time()
            visual_features = self.model.visual_cnn(face)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['visual_cnn'] = time.time() - start

            # Full forward pass
            start = time.time()
            masks = self.model(mixture, face)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['total'] = time.time() - start

        print("Component times:")
        for name, t in times.items():
            print(f"  {name}: {t:.3f}s")


def main():
    dataset_dir = "/Users/jonatanvider/Documents/LookingToListenProject/avspeech_prepro/processed_xaa"

    # Training configuration
    config = {
            'train_dir'     : dataset_dir,
            'batch_size'    : 16,
            'learning_rate' : 3e-4,
            'num_epochs'    : 1,
            'num_workers'   : 4,
            'log_interval'  : 10,  # Log every 10 batches
            'save_interval' : 1,  # Save checkpoint every epoch
            'log_dir'       : f'runs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'checkpoint_dir': 'checkpoints',
            'use_gcs'       : False  # Set to True when using GCS
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()