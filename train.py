# train.py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime

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

        # Create dataloaders
        self.train_loader = create_dataloader(
                config['train_dir'],
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers']
        )

        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
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
            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to tensorboard every N steps
            if self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        return epoch_loss / len(self.train_loader)

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss'                : loss,
                'config'              : self.config
        }

        checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Total batches per epoch: {len(self.train_loader)}")

        for epoch in range(1, self.config['num_epochs'] + 1):
            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, avg_loss)

        print("Training completed!")
        self.writer.close()

    # train.py (add this method to Trainer class)
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

    # Add this profiling method to Trainer class
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
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['audio_cnn'] = time.time() - start

            # Visual CNN
            start = time.time()
            visual_features = self.model.visual_cnn(face)
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['visual_cnn'] = time.time() - start

            # Upsampling
            start = time.time()
            from model.visual_model import upsample_visual_features

            visual_features = upsample_visual_features(visual_features)
            torch.mps.synchronize() if self.device.type == 'mps' else None
            times['upsample'] = time.time() - start

            # Fusion + LSTM + FC
            start = time.time()
            # ... rest of forward pass
            masks = self.model(mixture, face)
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
            'save_interval' : 1,  # Save checkpoint every 5 epochs
            'log_dir'       : f'runs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'checkpoint_dir': 'checkpoints'
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    # import sys
    #
    # # Get dataset path from command line (PyCharm will provide this)
    # if len(sys.argv) > 1:
    #     config['train_dir'] = sys.argv[1]

    main()