
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import tarfile


class PreStagingManager:
    """
    Simplified version for Colab - uses ! commands directly
    """

    def __init__(self, local_base_dir="/content/datasets"):
        self.local_base_dir = local_base_dir
        self.paths = {}

    def prepare_all_datasets(self, warm_start=False):
        """
        Pre-stage all three datasets using Colab's ! commands
        """
        import subprocess

        os.makedirs(self.local_base_dir, exist_ok=True)

        # Dataset configurations
        datasets_config = {
                '1s_noise': {
                        'gcs'            : 'gs://av_speech_60k_1s_noise/AV_SPEECH_60K_DATASET.tar.gz',
                        'local'          : f'{self.local_base_dir}/1s_noise',
                        'skip_warm_start': True
                },
                '2s_clean': {
                        'gcs'            : 'gs://av_speech_2s_clean_14k/2s_clean.tar.gz',
                        'local'          : f'{self.local_base_dir}/2s_clean',
                        'skip_warm_start': False
                },
                '2s_noise': {
                        'gcs'            : 'gs://av_speech_2s_clean_14k/2s_noise.tar.gz',
                        'local'          : f'{self.local_base_dir}/2s_noise',
                        'skip_warm_start': False
                }
        }

        for name, config in datasets_config.items():
            if warm_start and config['skip_warm_start']:
                print(f"⏭️  Skipping {name} (warm start mode)")
                continue

            # Check if already extracted
            if os.path.exists(config['local']) and any(Path(config['local']).iterdir()):
                print(f"✓ {name} already extracted at {config['local']}")
                self.paths[name] = config['local']
                continue

            # Download and extract
            print(f"\n{'=' * 60}")
            print(f"Processing {name}")
            print('=' * 60)

            tar_name = os.path.basename(config['gcs'])
            tar_path = f"/content/{tar_name}"  # Download to /content instead of /tmp

            # Download using gsutil
            print(f"📥 Downloading from {config['gcs']}...")
            cmd = f"gsutil -m cp {config['gcs']} {tar_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                raise RuntimeError(f"Failed to download {name}")

            # Check file size
            if os.path.exists(tar_path):
                size_mb = os.path.getsize(tar_path) / (1024 * 1024)
                print(f"✓ Downloaded {tar_name} ({size_mb:.1f} MB)")
            else:
                raise FileNotFoundError(f"Download failed - {tar_path} not found")

            # Extract
            print(f"📦 Extracting to {config['local']}...")
            os.makedirs(config['local'], exist_ok=True)

            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(config['local'])

            # Clean up tar
            os.remove(tar_path)
            print(f"🗑️ Removed {tar_path}")

            # Find actual data path
            extracted_items = list(Path(config['local']).iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                # Check if this is a wrapper directory
                potential_samples = list(extracted_items[0].iterdir())
                if potential_samples and (potential_samples[0] / 'audio').exists():
                    actual_path = str(extracted_items[0])
                    print(f"📂 Data found in subdirectory: {actual_path}")
                else:
                    actual_path = config['local']
            else:
                actual_path = config['local']

            # Verify
            sample_count = len([d for d in Path(actual_path).iterdir() if d.is_dir()])
            print(f"✅ {name}: {sample_count} samples ready at {actual_path}")

            self.paths[name] = actual_path

        return self.paths



# ... rest of the code remains the same
# Keep the same dataset classes from before, but with minor modification
class AVSpeechDataset1SNoise(Dataset):
    """Original 1 speaker + noise dataset"""

    def __init__(self, root_dir, max_samples=None):
        self.root_dir = Path(root_dir)
        self.sample_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # Option to limit samples (useful for testing)
        if max_samples:
            self.sample_dirs = self.sample_dirs[:max_samples]

        print(f"[1S+Noise] Found {len(self.sample_dirs)} samples in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        mixture_audio = torch.load(sample_dir / 'audio' / 'mixture_embs.pt', map_location='cpu')
        clean_audio = torch.load(sample_dir / 'audio' / 'clean_embs.pt', map_location='cpu')
        face_embs = torch.load(sample_dir / 'face' / 'face_embs.pt', map_location='cpu')

        return {
                'mixture'  : mixture_audio,
                'clean'    : clean_audio,
                'face'     : face_embs,
                'sample_id': sample_dir.name,
                'mix_type' : '1s_noise'
        }


class AVSpeechDataset2SClean(Dataset):
    """2 speakers mixed, no additional noise"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.sample_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.sample_dirs.sort()
        print(f"[2S Clean] Found {len(self.sample_dirs)} samples in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Your 2S Clean preprocessing should have saved these
        mixture_audio = torch.load(sample_dir / 'audio' / 'mixture_embs.pt')
        clean_audio = torch.load(sample_dir / 'audio' / 'clean_embs.pt')  # Target speaker's clean audio
        face_embs = torch.load(sample_dir / 'face' / 'face_embs.pt')  # Target speaker's face

        return {
                'mixture'  : mixture_audio,
                'clean'    : clean_audio,
                'face'     : face_embs,
                'sample_id': sample_dir.name,
                'mix_type' : '2s_clean'
        }


class AVSpeechDataset2SNoise(Dataset):
    """2 speakers + background noise"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.sample_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.sample_dirs.sort()
        print(f"[2S+Noise] Found {len(self.sample_dirs)} samples in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        mixture_audio = torch.load(sample_dir / 'audio' / 'mixture_embs.pt')
        clean_audio = torch.load(sample_dir / 'audio' / 'clean_embs.pt')
        face_embs = torch.load(sample_dir / 'face' / 'face_embs.pt')

        return {
                'mixture'  : mixture_audio,
                'clean'    : clean_audio,
                'face'     : face_embs,
                'sample_id': sample_dir.name,
                'mix_type' : '2s_noise'
        }


class MixedBatchSampler:
    """
    Samples batches with specified probabilities from different datasets
    """

    def __init__(self, dataset_sizes, batch_size, probabilities, seed=42):
        """
        Args:
            dataset_sizes: dict with keys '1s_noise', '2s_clean', '2s_noise' and values as dataset lengths
            batch_size: batch size
            probabilities: dict with same keys and probability values (must sum to 1.0)
            seed: random seed for reproducibility
        """
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.probabilities = probabilities
        self.dataset_names = list(dataset_sizes.keys())

        # Validate probabilities
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

        # Create indices for each dataset
        self.indices = {
                name: list(range(size))
                for name, size in dataset_sizes.items()
        }

        # Shuffle indices
        self.rng = np.random.RandomState(seed)
        for indices in self.indices.values():
            self.rng.shuffle(indices)

        # Track current position in each dataset
        self.current_pos = {name: 0 for name in self.dataset_names}

        # Calculate total number of batches
        self.num_batches = sum(dataset_sizes.values()) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []

            # Sample batch_size items according to probabilities
            for _ in range(self.batch_size):
                # Choose dataset according to probability
                dataset_name = self.rng.choice(
                        self.dataset_names,
                        p=[self.probabilities[name] for name in self.dataset_names]
                )

                # Get next index from chosen dataset (with cycling)
                idx = self.indices[dataset_name][self.current_pos[dataset_name]]
                self.current_pos[dataset_name] = (self.current_pos[dataset_name] + 1) % self.dataset_sizes[dataset_name]

                # If we've cycled through a dataset, reshuffle it
                if self.current_pos[dataset_name] == 0:
                    self.rng.shuffle(self.indices[dataset_name])

                # Add (dataset_index, sample_index) to batch
                dataset_idx = self.dataset_names.index(dataset_name)
                batch.append((dataset_idx, idx))

            yield batch

    def __len__(self):
        return self.num_batches


class CombinedAVDataset(Dataset):
    """
    Wrapper that combines all three datasets and handles mixed sampling
    """

    def __init__(self, dataset_1s, dataset_2s_clean, dataset_2s_noise):
        self.datasets = [dataset_1s, dataset_2s_clean, dataset_2s_noise]
        self.dataset_names = ['1s_noise', '2s_clean', '2s_noise']

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        # idx is a tuple: (dataset_index, sample_index)
        dataset_idx, sample_idx = idx
        return self.datasets[dataset_idx][sample_idx]


def create_mixed_dataloader(
        dir_1s_noise,
        dir_2s_clean,
        dir_2s_noise,
        batch_size=32,
        probabilities=None,
        num_workers=4,
        phase='train',
        seed=42
):
    """
    Create a dataloader that mixes batches from three different datasets
    """

    if probabilities is None:
        probabilities = {'1s_noise': 0.10, '2s_clean': 0.45, '2s_noise': 0.45}

    # Create individual datasets
    dataset_1s = AVSpeechDataset1SNoise(dir_1s_noise)
    dataset_2s_clean = AVSpeechDataset2SClean(dir_2s_clean)
    dataset_2s_noise = AVSpeechDataset2SNoise(dir_2s_noise)

    # Create combined dataset
    combined_dataset = CombinedAVDataset(dataset_1s, dataset_2s_clean, dataset_2s_noise)

    # Create sampler
    dataset_sizes = {
            '1s_noise': len(dataset_1s),
            '2s_clean': len(dataset_2s_clean),
            '2s_noise': len(dataset_2s_noise)
    }

    batch_sampler = MixedBatchSampler(
            dataset_sizes=dataset_sizes,
            batch_size=batch_size,
            probabilities=probabilities,
            seed=seed if phase == 'train' else seed + 1000  # Different seed for val
    )

    # Custom collate function to handle the mixed batch
    def mixed_collate_fn(batch):
        # batch is a list of samples from potentially different datasets
        # Stack them normally but preserve the mix_type information
        mixture = torch.stack([item['mixture'] for item in batch])
        clean = torch.stack([item['clean'] for item in batch])
        face = torch.stack([item['face'] for item in batch])

        return {
                'mixture'  : mixture,
                'clean'    : clean,
                'face'     : face,
                'sample_id': [item['sample_id'] for item in batch],
                'mix_type' : [item['mix_type'] for item in batch]
        }

    # Create dataloader with batch sampler
    dataloader = DataLoader(
            combined_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=mixed_collate_fn,
            pin_memory=torch.cuda.is_available()
    )

    return dataloader, dataset_sizes