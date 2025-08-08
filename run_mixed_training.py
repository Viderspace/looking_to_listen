# run_mixed_training.py
import os
from datetime import datetime
from utils.dataset_mixed import PreStagingManager
from train_mixed import MixedTrainer


def get_phase_config(phase, base_lr=3e-4):
    """Get configuration for different training phases"""

    configs = {
            'phase_a_warmstart': {
                    'probabilities' : {'1s_noise': 0.05, '2s_clean': 0.50, '2s_noise': 0.45},
                    'learning_rate' : base_lr / 3,  # 1e-4
                    'min_lr'        : 7e-5,
                    'num_epochs'    : 3,
                    'schedule_phase': 'Phase A - Warm Start'
            },
            'phase_b_main'     : {
                    'probabilities' : {'1s_noise': 0.10, '2s_clean': 0.45, '2s_noise': 0.45},
                    'learning_rate' : 8e-5,
                    'min_lr'        : 2e-5,
                    'num_epochs'    : 15,
                    'schedule_phase': 'Phase B - Main Training'
            },
            'phase_c_polish'   : {
                    'probabilities' : {'1s_noise': 0.07, '2s_clean': 0.465, '2s_noise': 0.465},
                    'learning_rate' : 3e-5,
                    'min_lr'        : 1e-5,
                    'num_epochs'    : 5,
                    'schedule_phase': 'Phase C - Polish'
            }
    }

    return configs.get(phase, configs['phase_b_main'])


def main():
    # Configuration
    IN_COLAB = 'COLAB_GPU' in os.environ

    # Base configuration
    base_config = {
            'batch_size'       : 32,
            'num_workers'      : 2 if IN_COLAB else 4,
            'gradient_clip'    : 1.0,
            'log_interval'     : 10,
            'save_interval'    : 1,
            'checkpoint_dir'   : '/content/drive/MyDrive/looking_to_listen/checkpoints_mixed',
            'log_dir'          : f'runs/mixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'resume_checkpoint': '/content/drive/MyDrive/looking_to_listen/checkpoints/checkpoint_epoch_20.pt'
    }

    # Create checkpoint directory
    os.makedirs(base_config['checkpoint_dir'], exist_ok=True)

    # Pre-stage datasets
    print("🚀 Pre-staging datasets...")
    manager = PreStagingManager(use_colab=IN_COLAB)

    # Set warm_start=True if you want to skip loading 1S+Noise initially
    dataset_paths = manager.prepare_all_datasets(warm_start=False)

    # Choose training phase
    TRAINING_PHASE = 'phase_a_warmstart'  # Change this as you progress
    phase_config = get_phase_config(TRAINING_PHASE)

    # Merge configs
    config = {**base_config, **phase_config}

    # Create mixed dataloader
    print("\n📊 Creating mixed dataloader...")
    from utils.dataset_mixed import create_mixed_dataloader

    train_dataloader, dataset_sizes = create_mixed_dataloader(
            dir_1s_noise=dataset_paths['1s_noise'],
            dir_2s_clean=dataset_paths['2s_clean'],
            dir_2s_noise=dataset_paths['2s_noise'],
            batch_size=config['batch_size'],
            probabilities=config['probabilities'],
            num_workers=config['num_workers'],
            phase='train'
    )

    # Update steps per epoch
    config['steps_per_epoch'] = len(train_dataloader)

    print(f"\n📈 Dataset sizes:")
    for name, size in dataset_sizes.items():
        print(f"  {name}: {size:,} samples")
    print(f"  Total batches per epoch: {len(train_dataloader)}")

    # Create trainer and start training
    print("\n🎯 Starting training...")
    trainer = MixedTrainer(config)
    trainer.train(train_dataloader)

    # Cleanup if needed (comment out if you want to keep data for next run)
    # manager.cleanup()


if __name__ == "__main__":
    main()