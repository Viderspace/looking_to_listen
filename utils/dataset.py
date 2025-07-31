# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import platform



class AVSpeechDataset(Dataset):

    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to directory containing sample folders
        """
        self.root_dir = Path(root_dir)

        # Get all sample directories
        self.sample_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.sample_dirs.sort()  # For reproducibility

        print(f"Found {len(self.sample_dirs)} samples in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load only what we need
        mixture_audio = torch.load(sample_dir / 'audio' / 'mixture_embs.pt')
        clean_audio = torch.load(sample_dir / 'audio' / 'clean_embs.pt')
        face_embs = torch.load(sample_dir / 'face' / 'face_embs.pt')

        return {
                'mixture'  : mixture_audio,  # [257, 298, 2]
                'clean'    : clean_audio,  # [257, 298, 2]
                'face'     : face_embs,  # [75, 512]
                'sample_id': sample_dir.name
        }


# def create_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
#     """
#     Create a dataloader for training
#     """
#     dataset = AVSpeechDataset(root_dir)
#
#     # Simple collate - PyTorch handles tensor stacking automatically
#     dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=True  # Faster GPU transfer
#     )
#
#     return dataloader
#
#
#

def create_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
    """
    Create a dataloader for training
    """
    dataset = AVSpeechDataset(root_dir)

    # Detect if we're on Apple Silicon
    is_mps = platform.system() == 'Darwin' and torch.backends.mps.is_available()

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=not is_mps  # Disable pin_memory on MPS
    )

    return dataloader