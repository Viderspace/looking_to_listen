# utils/gcs_dataset.py
import torch
from torch.utils.data import Dataset
import os
import random
from typing import Dict, List
import numpy as np


class GCSAVSpeechDataset(Dataset):
    """Dataset class that loads AV Speech data from Google Cloud Storage"""

    def __init__(self, sample_list: List[str], gcs_helper, cache_size: int = 1000):
        """
        Args:
            sample_list: List of sample IDs in GCS
            gcs_helper: GCSDataHelper instance for downloading files
            cache_size: Number of samples to keep in memory cache
        """
        self.sample_list = sample_list
        self.gcs_helper = gcs_helper
        self.cache_size = cache_size
        self.cache = {}  # In-memory cache for loaded tensors

        print(f"Initialized GCS dataset with {len(sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_list[idx]

        # Check in-memory cache first
        if sample_id in self.cache:
            return self.cache[sample_id]

        try:
            # Download if not cached locally
            sample_path = self.gcs_helper.download_sample(sample_id)

            # Load tensors
            clean_audio = torch.load(
                    os.path.join(sample_path, 'audio', 'clean_embs.pt'),
                    map_location='cpu'
            )
            mixture_audio = torch.load(
                    os.path.join(sample_path, 'audio', 'mixture_embs.pt'),
                    map_location='cpu'
            )
            face_embs = torch.load(
                    os.path.join(sample_path, 'face', 'face_embs.pt'),
                    map_location='cpu'
            )

            # Validate shapes
            # Expected shapes based on your preprocessing:
            # Audio: [257, 298, 2] for 3-second chunks
            # Face: [75, 512] for 75 frames at 25fps

            if len(clean_audio.shape) == 3 and clean_audio.shape[0] == 257:
                # Single chunk - add batch dimension
                clean_audio = clean_audio.unsqueeze(0)
                mixture_audio = mixture_audio.unsqueeze(0)
                face_embs = face_embs.unsqueeze(0)

            # Random selection if multiple chunks
            if clean_audio.shape[0] > 1:
                chunk_idx = random.randint(0, clean_audio.shape[0] - 1)
                clean_audio = clean_audio[chunk_idx]
                mixture_audio = mixture_audio[chunk_idx]
                face_embs = face_embs[chunk_idx]
            else:
                clean_audio = clean_audio.squeeze(0)
                mixture_audio = mixture_audio.squeeze(0)
                face_embs = face_embs.squeeze(0)

            sample_data = {
                    'clean'    : clean_audio,
                    'mixture'  : mixture_audio,
                    'face'     : face_embs,
                    'sample_id': sample_id
            }

            # Update cache (with size limit)
            if len(self.cache) >= self.cache_size:
                # Remove random item to make space
                remove_key = random.choice(list(self.cache.keys()))
                del self.cache[remove_key]

            self.cache[sample_id] = sample_data

            return sample_data

        except Exception as e:
            print(f"Error loading sample {sample_id}: {e}")
            # Return a dummy sample to continue training
            return self._get_dummy_sample(sample_id)

    def _get_dummy_sample(self, sample_id: str) -> Dict[str, torch.Tensor]:
        """Create a dummy sample for error cases"""
        return {
                'clean'    : torch.zeros(257, 298, 2),
                'mixture'  : torch.zeros(257, 298, 2),
                'face'     : torch.zeros(75, 512),
                'sample_id': f"dummy_{sample_id}"
        }


class GCSDataHelper:
    """Helper class for downloading samples from GCS"""

    def __init__(self, bucket, prefix: str, cache_dir: str):
        self.bucket = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.download_stats = {'success': 0, 'failed': 0, 'cached': 0}

    def download_sample(self, sample_id: str) -> str:
        """Download all files for a sample from GCS to local cache"""
        sample_cache_dir = os.path.join(self.cache_dir, sample_id)

        # Check if already cached
        if self._is_cached(sample_cache_dir):
            self.download_stats['cached'] += 1
            return sample_cache_dir

        # Create directories
        os.makedirs(os.path.join(sample_cache_dir, 'audio'), exist_ok=True)
        os.makedirs(os.path.join(sample_cache_dir, 'face'), exist_ok=True)

        # Download files
        files_to_download = [
                ('audio/clean_embs.pt', 'audio/clean_embs.pt'),
                ('audio/mixture_embs.pt', 'audio/mixture_embs.pt'),
                ('face/face_embs.pt', 'face/face_embs.pt')
        ]

        try:
            for gcs_path, local_path in files_to_download:
                blob_name = f"{self.prefix}{sample_id}/{gcs_path}"
                blob = self.bucket.blob(blob_name)
                local_file = os.path.join(sample_cache_dir, local_path)

                # Download with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        blob.download_to_filename(local_file)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"Retry {attempt + 1} for {blob_name}")

            self.download_stats['success'] += 1
            return sample_cache_dir

        except Exception as e:
            print(f"Failed to download {sample_id}: {e}")
            self.download_stats['failed'] += 1
            # Clean up partial download
            import shutil

            if os.path.exists(sample_cache_dir):
                shutil.rmtree(sample_cache_dir)
            raise e

    def _is_cached(self, sample_cache_dir: str) -> bool:
        """Check if sample is fully cached"""
        if not os.path.exists(sample_cache_dir):
            return False

        required_files = [
                'audio/clean_embs.pt',
                'audio/mixture_embs.pt',
                'face/face_embs.pt'
        ]

        for file_path in required_files:
            full_path = os.path.join(sample_cache_dir, file_path)
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                return False

        return True

    def get_stats(self):
        """Get download statistics"""
        return self.download_stats