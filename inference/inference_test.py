#!/usr/bin/env python3
"""
Looking to Listen - Inference Testing Script
Tests trained model on sample videos and outputs separated audio
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import soundfile as sf


# Add your model path
sys.path.append('/Users/jonatanvider/Documents/LookingToListenProject/avspeech_model/model')
from model.av_model import AudioVisualModel


class LookingToListenInference:

    def __init__(self, checkpoint_path, device=None):
        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        print(f"Using device: {device}")

        self.model = AudioVisualModel().to(device)

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Training loss: {checkpoint.get('loss', 'unknown'):.4f}")

    def load_embeddings(self, embeddings_folder):
        """Load all preprocessed embeddings from your folder structure"""

        embeddings_path = Path(embeddings_folder)

        # Find all chunk folders (sorted by number)
        chunk_folders = sorted([f for f in embeddings_path.iterdir()
                                if f.is_dir() and f.name.split('_')[-1].isdigit()],
                               key=lambda x: int(x.name.split('_')[-1]))

        audio_chunks = []
        face_chunks = []

        print(f"Found {len(chunk_folders)} chunks to process")

        for chunk_folder in chunk_folders:
            # Load audio embeddings (STFT)
            audio_path = chunk_folder / 'audio' / 'audio_embs.pt'
            face_path = chunk_folder / 'face' / 'face_embs.pt'

            if audio_path.exists() and face_path.exists():
                audio_emb = torch.load(audio_path, map_location='cpu')
                face_emb = torch.load(face_path, map_location='cpu')

                # Ensure correct shapes
                print(f"Chunk {chunk_folder.name}: Audio {audio_emb.shape}, Face {face_emb.shape}")

                audio_chunks.append(audio_emb)
                face_chunks.append(face_emb)
            else:
                print(f"⚠️  Missing embeddings in {chunk_folder.name}")

        return audio_chunks, face_chunks

    def debug_audio_parameters(self, embeddings_folder):
        """Debug function to check STFT parameters match training"""

        print("🔍 Checking your preprocessing parameters...")

        # Load one chunk to inspect
        embeddings_path = Path(embeddings_folder)
        first_chunk = next(embeddings_path.iterdir())
        audio_path = first_chunk / 'audio' / 'audio_embs.pt'

        if audio_path.exists():
            audio_stft = torch.load(audio_path, map_location='cpu')
            print(f"STFT shape: {audio_stft.shape}")
            print(f"STFT dtype: {audio_stft.dtype}")

            # Based on paper: STFT with Hann window 25ms, hop 10ms, FFT 512
            # At 16kHz: 25ms = 400 samples, 10ms = 160 samples
            expected_freq_bins = 257  # = 257 (matches your shape!)
            expected_time_frames = 298  # ≈ 298 for 3s

            print(f"Expected shape for 3s audio: [{expected_freq_bins}, {expected_time_frames}, 2]")
            print(f"Your shape: {list(audio_stft.shape)}")

            if list(audio_stft.shape) == [257, 298, 2]:
                print("✅ STFT shape matches paper specifications!")
                return {
                        'n_fft'      : 512,
                        'hop_length' : 160,
                        'win_length' : 400,
                        'power_law_p': 0.3  # Add power law parameter from config
                }
            else:
                print("⚠️  STFT shape doesn't match expected - may need parameter adjustment")

        return None

    def stft_to_audio(self, stft_compressed, n_fft=512, hop_length=160, win_length=400, power_law_p=0.3):
        """Convert compressed STFT back to audio waveform

        Args:
            stft_compressed: STFT with power-law compression applied [freq, time, 2]
            n_fft: FFT size (512)
            hop_length: Hop length in samples (160)
            win_length: Window length in samples (400)
            power_law_p: Power law compression parameter (0.3 from your config)
        """

        print(f"DEBUG: STFT input shape: {stft_compressed.shape}")
        print(f"DEBUG: STFT dtype: {stft_compressed.dtype}")
        print(f"DEBUG: STFT device: {stft_compressed.device}")

        # Handle the format - your embeddings are [freq, time, 2] (real/imag)
        if len(stft_compressed.shape) == 3 and stft_compressed.shape[-1] == 2:
            print("DEBUG: Converting compressed real/imag to complex")
            # Extract compressed real and imaginary parts
            real_compressed = stft_compressed[..., 0]
            imag_compressed = stft_compressed[..., 1]

            # Invert the power-law compression
            # Original: sign(x) * |x|^p
            # Inverse: sign(x) * |x|^(1/p)
            inverse_power = 1.0 / power_law_p

            real_decompressed = torch.sign(real_compressed) * torch.abs(real_compressed).pow(inverse_power)
            imag_decompressed = torch.sign(imag_compressed) * torch.abs(imag_compressed).pow(inverse_power)

            # Convert to complex tensor
            stft_complex = torch.complex(real_decompressed, imag_decompressed)
        else:
            # Already complex, but might still need decompression
            raise ValueError("Expected input shape [freq, time, 2] with compressed values")

        print(f"DEBUG: After decompression - shape: {stft_complex.shape}, dtype: {stft_complex.dtype}")

        # Create window
        window = torch.hann_window(win_length, device=stft_complex.device)

        # ISTFT with center=True (works reliably across devices)
        # Note: STFT used center=False, but ISTFT with center=True works well
        # and avoids device-specific issues. Edge artifacts are negligible.
        audio = torch.istft(
                stft_complex,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,  # Always use center=True for reliability
                normalized=False,
                onesided=True,
                length=None
        )

        print(f"DEBUG: Audio output shape: {audio.shape}")
        print(f"DEBUG: Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

        # Check for any issues
        if torch.isnan(audio).any():
            print("WARNING: Audio contains NaN values!")
        if torch.isinf(audio).any():
            print("WARNING: Audio contains infinite values!")

        # Normalize audio to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 1.0:
            print(f"WARNING: Audio peak {max_val:.4f} > 1.0, normalizing to prevent clipping")
            audio = audio / max_val * 0.95  # Scale to 95% to leave headroom

        return audio

    def process_video(self, embeddings_folder, output_dir):
        """Process entire video by processing all chunks"""

        # Debug STFT parameters first
        stft_params = self.debug_audio_parameters(embeddings_folder)

        # Load all preprocessed embeddings
        audio_chunks, face_chunks = self.load_embeddings(embeddings_folder)

        if not audio_chunks:
            raise ValueError("No valid chunks found!")

        # Process each chunk
        enhanced_stfts = []
        original_stfts = []

        for i, (audio_stft, face_emb) in enumerate(tqdm(zip(audio_chunks, face_chunks))):
            enhanced_stft, mask = self.enhance_chunk(audio_stft, face_emb)
            enhanced_stfts.append(enhanced_stft)
            original_stfts.append(audio_stft)

        # Concatenate all STFTs along time dimension [257, 298*5, 2]
        full_enhanced_stft = torch.cat(enhanced_stfts, dim=1)
        full_original_stft = torch.cat(original_stfts, dim=1)

        # Single ISTFT on concatenated spectrograms
        if stft_params:
            full_enhanced = self.stft_to_audio(full_enhanced_stft, **stft_params)
            full_original = self.stft_to_audio(full_original_stft, **stft_params)
        else:
            # Default with power law parameter
            full_enhanced = self.stft_to_audio(full_enhanced_stft, power_law_p=0.3)
            full_original = self.stft_to_audio(full_original_stft, power_law_p=0.3)

        # Save results
        self.save_results(full_enhanced, full_original, output_dir, Path(embeddings_folder).name)

        return full_enhanced, full_original

    def enhance_chunk(self, audio_stft, face_embeddings):
        """Run inference on a single chunk"""
        # Purpose: This method processes one 3-second chunk of audio + face data through your trained model.

        with torch.no_grad():
            # Critical for inference: Disables gradient computation, which:
            # - Saves memory (no need to store gradients)
            # - Speeds up computation
            # - Prevents accidental model weight updates during inference

            # Move to device and ensure batch dimension
            audio_stft = audio_stft.to(self.device)
            face_embeddings = face_embeddings.to(self.device)
            # Device transfer: Moves tensors from CPU to your MPS device (Apple Silicon GPU):
            # - audio_stft: Your STFT spectrogram [257, 298, 2]
            # - face_embeddings: Your face data [75, 512]

            if len(audio_stft.shape) == 3:  # Add batch dim if missing
                audio_stft = audio_stft.unsqueeze(0)
            # Batch dimension handling: Your model was trained with batches, so it expects:
            # - Input shape: [257, 298, 2] (frequency, time, real/imag)
            # - Model expects: [batch_size, 257, 298, 2]
            # - unsqueeze(0) adds batch dimension: [1, 257, 298, 2]

            if len(face_embeddings.shape) == 2:  # Add batch dim if missing
                face_embeddings = face_embeddings.unsqueeze(0)
            # Same for faces:
            # - Input shape: [75, 512] (75 frames, 512-dim embeddings)
            # - Model expects: [batch_size, 75, 512]
            # - Result: [1, 75, 512]

            # Run model - outputs mask for target speaker
            mask = self.model(audio_stft, face_embeddings)
            # The magic happens here:
            # - Feeds both audio (STFT) and visual (faces) into your trained model
            # - Model learns which parts of the audio correspond to the visible speaker
            # - Output: A mask with same shape as input STFT [1, 257, 298, 2]
            # - Mask values: Typically between 0-1, where 1 = "keep this frequency/time", 0 = "suppress this"

            # Apply mask to enhance target speaker
            enhanced_stft = audio_stft * mask
            # Element-wise multiplication:
            # - audio_stft: Original mixed audio (speaker + noise)
            # - mask: Model's prediction of which parts belong to target speaker
            # - enhanced_stft: Cleaned audio with noise/interference suppressed
            # This is the core of masking-based speech separation!

            return enhanced_stft.squeeze(0), mask.squeeze(0)
            # Remove batch dimension and return:
            # - squeeze(0): Removes the batch dimension we added
            # - Returns: Enhanced STFT [257, 298, 2] and mask (for debugging/analysis)

    def save_results(self, enhanced_audio, original_audio, output_dir, video_name):
        """Save enhanced audio and create visualization"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Ensure audio is 1D and on CPU
        if len(enhanced_audio.shape) > 1:
            enhanced_audio = enhanced_audio.squeeze()
        if len(original_audio.shape) > 1:
            original_audio = original_audio.squeeze()

        # Move to CPU and convert to numpy
        enhanced_audio = enhanced_audio.cpu().numpy()
        original_audio = original_audio.cpu().numpy()

        # Save audio files using soundfile
        sf.write(
                output_dir / f"{video_name}_original.wav",
                original_audio,
                16000
        )

        sf.write(
                output_dir / f"{video_name}_enhanced.wav",
                enhanced_audio,
                16000
        )

        # Create visualization
        self.create_visualization(
                torch.from_numpy(enhanced_audio),
                torch.from_numpy(original_audio),
                output_dir,
                video_name
        )

        print(f"✓ Results saved to {output_dir}")
        print(f"   • {video_name}_original.wav ({len(original_audio) / 16000:.1f}s)")
        print(f"   • {video_name}_enhanced.wav ({len(enhanced_audio) / 16000:.1f}s)")
        print(f"   • {video_name}_comparison.png")

    def create_visualization(self, enhanced_audio, original_audio, output_dir, video_name):
        """Create before/after spectrogram comparison"""

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        # Create window to match your preprocessing
        window = torch.hann_window(400)  # 25ms window at 16kHz

        # Original mixed audio spectrogram
        original_stft = torch.stft(
                original_audio,
                n_fft=512,
                hop_length=160,
                win_length=400,
                window=window,
                return_complex=True,
                center=False  # Match your preprocessing
        )
        original_mag = torch.log(torch.abs(original_stft) + 1e-8)

        # Enhanced audio spectrogram
        enhanced_stft = torch.stft(
                enhanced_audio,
                n_fft=512,
                hop_length=160,
                win_length=400,
                window=window,
                return_complex=True,
                center=False  # Match your preprocessing
        )
        enhanced_mag = torch.log(torch.abs(enhanced_stft) + 1e-8)

        # Plot spectrograms
        im1 = axes[0].imshow(original_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Mixed Audio (Speaker + Noise/Interference)')
        axes[0].set_ylabel('Frequency Bin')

        im2 = axes[1].imshow(enhanced_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Enhanced Audio (Target Speaker Isolated)')
        axes[1].set_ylabel('Frequency Bin')
        axes[1].set_xlabel('Time Frame')

        # Add colorbars
        plt.colorbar(im1, ax=axes[0], label='Log Magnitude')
        plt.colorbar(im2, ax=axes[1], label='Log Magnitude')

        plt.tight_layout()
        plt.savefig(output_dir / f"{video_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


def debug_compression_decompression():
    """Test that compression and decompression are inverses"""
    import torch

    # Test values
    test_values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    power_law_p = 0.3

    # Compress
    compressed = torch.sign(test_values) * torch.abs(test_values).pow(power_law_p)

    # Decompress
    decompressed = torch.sign(compressed) * torch.abs(compressed).pow(1.0 / power_law_p)

    print("Original:     ", test_values.numpy())
    print("Compressed:   ", compressed.numpy())
    print("Decompressed: ", decompressed.numpy())
    print("Difference:   ", (test_values - decompressed).abs().max().item())

    # The difference should be very small (< 1e-6)
    assert (test_values - decompressed).abs().max() < 1e-6, "Compression/decompression not inverse!"
    print("✓ Compression/decompression verified!")


def main():
    parser = argparse.ArgumentParser(description='Test Looking to Listen model on preprocessed embeddings')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--embeddings', required=True, help='Path to preprocessed embeddings folder')
    parser.add_argument('--output', default='./inference_results_60k_20epochs', help='Output directory')
    parser.add_argument('--debug-compression', action='store_true', help='Run compression/decompression debug test')

    args = parser.parse_args()

    # Optional: Run debug test
    if args.debug_compression:
        print("🔍 Running compression/decompression debug test...")
        debug_compression_decompression()
        print()

    # Initialize inference
    print("🎯 Loading model...")
    inference = LookingToListenInference(args.checkpoint)

    # Process video
    print(f"📁 Processing embeddings from: {args.embeddings}")
    enhanced_audio, original_audio = inference.process_video(args.embeddings, args.output)

    print("✅ Speech enhancement complete!")
    print(f"📊 Audio length: {enhanced_audio.shape[-1] / 16000:.2f} seconds")


if __name__ == "__main__":
    main()