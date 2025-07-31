# # test_complete_model.py (updated)
# import torch
# from av_model import AudioVisualModel
#
#
# def test_complete_forward():
#     # Create model
#     model = AudioVisualModel()
#
#     # Create dummy inputs
#     batch_size = 2
#     audio_input = torch.randn(batch_size, 257, 298, 2)
#     visual_input = torch.randn(batch_size, 75, 512)
#
#     print("Input shapes:")
#     print(f"  Audio: {audio_input.shape}")
#     print(f"  Visual: {visual_input.shape}")
#
#     # Forward pass
#     masks = model(audio_input, visual_input)
#
#     print(f"\nModel output shape: {masks.shape}")
#     print(f"Expected: [batch={batch_size}, freq=257, time=298, complex=2]")
#
#     # Check that output has same shape as input audio
#     assert masks.shape == audio_input.shape, \
#         f"Output shape {masks.shape} doesn't match input shape {audio_input.shape}"
#
#     print("\n✓ Output shape matches input!")
#
#     # Check mask statistics (after sigmoid)
#     print(f"\nMask statistics (after sigmoid):")
#     print(f"  Min: {masks.min().item():.3f}")
#     print(f"  Max: {masks.max().item():.3f}")
#     print(f"  Mean: {masks.mean().item():.3f}")
#     print(f"  Std: {masks.std().item():.3f}")
#
#     # Check distribution
#     print(f"\nMask distribution:")
#     print(f"  < 0.1: {(masks < 0.1).float().mean().item():.1%}")
#     print(f"  0.1-0.5: {((masks >= 0.1) & (masks < 0.5)).float().mean().item():.1%}")
#     print(f"  0.5-0.9: {((masks >= 0.5) & (masks < 0.9)).float().mean().item():.1%}")
#     print(f"  > 0.9: {(masks > 0.9).float().mean().item():.1%}")
#
#     # Test applying mask to input
#     separated = audio_input * masks
#     print(f"\nSeparated audio shape: {separated.shape}")
#
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nModel parameters:")
#     print(f"  Total: {total_params:,}")
#     print(f"  Trainable: {trainable_params:,}")
#
#
# if __name__ == "__main__":
#     test_complete_forward()


# test_loss.py
import torch
from model.loss import ComplexCompressedLoss
from model.av_model import AudioVisualModel


def test_loss_function():
    # Create model and loss
    model = AudioVisualModel()
    criterion = ComplexCompressedLoss()

    # Create dummy data
    batch_size = 2
    noisy_audio = torch.randn(batch_size, 257, 298, 2)
    clean_audio = torch.randn(batch_size, 257, 298, 2)  # Target
    visual_input = torch.randn(batch_size, 75, 512)

    # Forward pass
    masks = model(noisy_audio, visual_input)

    # Apply masks to get separated audio
    separated_audio = noisy_audio * masks

    # Compute loss
    loss = criterion(separated_audio, clean_audio)

    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")  # Should be scalar

    # Test gradient flow
    loss.backward()

    # Check if gradients are computed
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_grad}")

    # Test with identical inputs (loss should be 0)
    zero_loss = criterion(clean_audio, clean_audio)
    print(f"\nLoss with identical inputs: {zero_loss.item():.6f}")

    # Test loss magnitude with different scales
    small_diff = clean_audio + 0.1 * torch.randn_like(clean_audio)
    small_loss = criterion(small_diff, clean_audio)

    large_diff = clean_audio + 1.0 * torch.randn_like(clean_audio)
    large_loss = criterion(large_diff, clean_audio)

    print(f"\nLoss with small difference: {small_loss.item():.4f}")
    print(f"Loss with large difference: {large_loss.item():.4f}")


if __name__ == "__main__":
    test_loss_function()
