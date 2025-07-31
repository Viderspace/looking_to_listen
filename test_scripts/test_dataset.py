# test_dataset.py
from utils.dataset import create_dataloader


def test_dataset(root_dir):
    # Create dataloader
    dataloader = create_dataloader(root_dir, batch_size=4, num_workers=0)

    # Get one batch
    batch = next(iter(dataloader))

    print("Batch contents:")
    print(f"  mixture: {batch['mixture'].shape}")  # Should be [4, 257, 298, 2]
    print(f"  clean: {batch['clean'].shape}")  # Should be [4, 257, 298, 2]
    print(f"  face: {batch['face'].shape}")  # Should be [4, 75, 512]
    print(f"  sample_ids: {batch['sample_id'][:2]}...")  # First 2 IDs

    # Test compatibility with model
    print("\nTesting with model:")
    from model.av_model import AudioVisualModel

    model = AudioVisualModel()

    # Forward pass
    masks = model(batch['mixture'], batch['face'])
    print(f"Model output shape: {masks.shape}")


if __name__ == "__main__":
    root_dir = "/Users/jonatanvider/Documents/LookingToListenProject/avspeech_prepro/processed_xaa"
    test_dataset(root_dir)

