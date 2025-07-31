# # test_training.py (optimized version)
# from train import Trainer
#
#
# def test_training():
#     # Optimized config for testing
#     root_dir = "/Users/jonatanvider/Documents/LookingToListenProject/avspeech_prepro/processed_xaa"
#     config = {
#             'train_dir'     : root_dir,
#             'batch_size'    : 16,  # Larger batch size (MPS can handle it)
#             'learning_rate' : 3e-4,
#             'num_epochs'    : 1,  # Just 1 epoch for testing
#             'num_workers'   : 0,  # Keep at 0 for MPS
#             'log_interval'  : 50,  # Log less frequently
#             'save_interval' : 1,
#             'log_dir'       : 'test_runs',
#             'checkpoint_dir': 'test_checkpoints'
#     }
#
#     trainer = Trainer(config)
#     trainer.test_step()
#
# if __name__ == "__main__":
#     test_training()


# # profiling_test.py
# import torch
# import time
# from av_model import AudioVisualModel
#
#
# def profile_model():
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     model = AudioVisualModel().to(device)
#     model.eval()
#
#     # Create dummy data
#     batch_size = 16
#     audio = torch.randn(batch_size, 257, 298, 2).to(device)
#     visual = torch.randn(batch_size, 75, 512).to(device)
#
#     # Warmup
#     with torch.no_grad():
#         _ = model(audio, visual)
#
#     # Profile each component
#     times = []
#
#     # Audio CNN
#     start = time.time()
#     with torch.no_grad():
#         audio_features = model.audio_cnn(audio)
#     times.append(('Audio CNN', time.time() - start))
#
#     # Visual CNN
#     start = time.time()
#     with torch.no_grad():
#         visual_features = model.visual_cnn(visual)
#     times.append(('Visual CNN', time.time() - start))
#
#     # Full forward pass
#     start = time.time()
#     with torch.no_grad():
#         _ = model(audio, visual)
#     times.append(('Full Model', time.time() - start))
#
#     print(f"Profiling on {device}:")
#     for name, t in times:
#         print(f"  {name}: {t:.3f}s")
#
#     # Calculate BiLSTM+FC time
#     blstm_fc_time = times[2][1] - times[0][1] - times[1][1]
#     print(f"  BiLSTM+FC (approx): {blstm_fc_time:.3f}s")
#
#
# if __name__ == "__main__":
#     profile_model()

# debug_fusion.py
import torch
import time
# from av_model import AudioVisualModel
#
#
# def debug_fusion():
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     model = AudioVisualModel().to(device)
#     model.eval()
#
#     batch_size = 16
#
#     with torch.no_grad():
#         # Get features before fusion
#         audio = torch.randn(batch_size, 257, 298, 2).to(device)
#         visual = torch.randn(batch_size, 75, 512).to(device)
#
#         audio_features = model.audio_cnn(audio)  # [16, 8, 257, 298]
#         visual_features = model.visual_cnn(visual)  # [16, 256, 75]
#
#         # Time each fusion step
#         start = time.time()
#         from visual_cnn import upsample_visual_features
#
#         visual_features = upsample_visual_features(visual_features)
#         print(f"Upsampling: {time.time() - start:.3f}s")
#
#         start = time.time()
#         # Reshape audio
#         audio_features = audio_features.permute(0, 3, 1, 2)
#         audio_features = audio_features.reshape(batch_size, 298, -1)
#         # Reshape visual
#         visual_features = visual_features.transpose(1, 2)
#         # Concatenate
#         fused = torch.cat([audio_features, visual_features], dim=2)
#         print(f"Reshape & concat: {time.time() - start:.3f}s")
#
#         # Time BiLSTM alone
#         start = time.time()
#         lstm_out, _ = model.blstm(fused)
#         print(f"BiLSTM: {time.time() - start:.3f}s")
#
#         # Time FC layers
#         start = time.time()
#         fc_input = lstm_out.reshape(-1, 800)
#         x = model.fc1(fc_input)
#         x = model.bn1(x)
#         x = torch.relu(x)
#         x = model.fc2(x)
#         x = model.bn2(x)
#         x = torch.relu(x)
#         x = model.fc3(x)
#         print(f"FC layers: {time.time() - start:.3f}s")
#
#
# if __name__ == "__main__":
#     debug_fusion()



# # test_upsampling.py
# import torch
# import time
# from visual_cnn import upsample_visual_features, upsample_visual_features_fast
#
# device = torch.device('mps')
# visual_features = torch.randn(16, 256, 75).to(device)
#
# # Original
# start = time.time()
# up1 = upsample_visual_features(visual_features)
# print(f"Original: {time.time() - start:.3f}s")
#
# # Fast version
# start = time.time()
# up2 = upsample_visual_features_fast(visual_features)
# print(f"Fast: {time.time() - start:.3f}s")
#
# print(f"Output shapes match: {up1.shape == up2.shape}")

#
# # detailed_profile.py
# import torch
# import time
# from av_model import AudioVisualModel
# from visual_cnn import upsample_visual_features
#
#
# def detailed_profile():
#     device = torch.device('mps')
#     model = AudioVisualModel().to(device)
#     model.eval()
#
#     batch_size = 16
#     audio = torch.randn(batch_size, 257, 298, 2).to(device)
#     visual = torch.randn(batch_size, 75, 512).to(device)
#
#     with torch.no_grad():
#         # Warm up
#         _ = model(audio, visual)
#
#         # Now profile step by step INSIDE the forward pass
#         print("Profiling inside forward pass:")
#
#         # Modified forward pass with timing
#         start_total = time.time()
#
#         # Audio CNN
#         t0 = time.time()
#         audio_features = model.audio_cnn(audio)
#         t1 = time.time()
#         print(f"  Audio CNN: {t1 - t0:.3f}s")
#
#         # Visual CNN
#         visual_features = model.visual_cnn(visual)
#         t2 = time.time()
#         print(f"  Visual CNN: {t2 - t1:.3f}s")
#
#         # Upsample
#         visual_features = upsample_visual_features(visual_features)
#         t3 = time.time()
#         print(f"  Upsample: {t3 - t2:.3f}s")
#
#         # Reshape audio
#         audio_features = audio_features.permute(0, 3, 1, 2)
#         audio_features = audio_features.reshape(batch_size, 298, -1)
#         t4 = time.time()
#         print(f"  Audio reshape: {t4 - t3:.3f}s")
#
#         # Reshape visual
#         visual_features = visual_features.transpose(1, 2)
#         t5 = time.time()
#         print(f"  Visual reshape: {t5 - t4:.3f}s")
#
#         # Concatenate
#         fused_features = torch.cat([audio_features, visual_features], dim=2)
#         t6 = time.time()
#         print(f"  Concatenate: {t6 - t5:.3f}s")
#
#         # BiLSTM
#         lstm_out, _ = model.blstm(fused_features)
#         t7 = time.time()
#         print(f"  BiLSTM: {t7 - t6:.3f}s")
#
#         # FC layers reshaping
#         fc_input = lstm_out.reshape(-1, 800)
#         t8 = time.time()
#         print(f"  FC reshape: {t8 - t7:.3f}s")
#
#         # FC1
#         fc1_out = model.fc1(fc_input)
#         fc1_out = model.bn1(fc1_out)
#         fc1_out = torch.relu(fc1_out)
#         t9 = time.time()
#         print(f"  FC1+BN+ReLU: {t9 - t8:.3f}s")
#
#         # FC2
#         fc2_out = model.fc2(fc1_out)
#         fc2_out = model.bn2(fc2_out)
#         fc2_out = torch.relu(fc2_out)
#         t10 = time.time()
#         print(f"  FC2+BN+ReLU: {t10 - t9:.3f}s")
#
#         # FC3
#         fc3_out = model.fc3(fc2_out)
#         t11 = time.time()
#         print(f"  FC3: {t11 - t10:.3f}s")
#
#         # Final reshape
#         output = fc3_out.reshape(batch_size, 298, 257, 2)
#         output = output.permute(0, 2, 1, 3)
#         masks = torch.sigmoid(output)
#         t12 = time.time()
#         print(f"  Final reshape+sigmoid: {t12 - t11:.3f}s")
#
#         print(f"\nTotal: {t12 - start_total:.3f}s")
#         print(
#             f"Sum of parts: {sum([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t9 - t8, t10 - t9, t11 - t10, t12 - t11]):.3f}s")
#
#
# if __name__ == "__main__":
#     detailed_profile()
#
# test_upsampling_fix.py
import torch
import torch.nn.functional as F


def debug_upsampling():
    # Small example to understand the mapping
    device = torch.device('mps')

    # Create a simple pattern to track
    visual = torch.arange(75).float().reshape(1, 1, 75).to(device)
    print(f"Source (75 frames): {visual[0, 0, :10]}...")  # First 10 values

    # F.interpolate version
    visual_4d = visual.unsqueeze(-1)
    up1 = F.interpolate(visual_4d, size=(298, 1), mode='nearest').squeeze(-1)
    print(f"\nF.interpolate result: {up1[0, 0, :10]}...")

    # Let's understand the mapping
    # 75 frames -> 298 frames
    # Each source frame should be repeated ~3.97 times
    scale = 298 / 75  # 3.9733...

    # Correct nearest neighbor mapping
    indices = torch.arange(298).float() / scale  # Scale down to source space
    indices = torch.round(indices).long().clamp(0, 74).to(device)
    up2 = visual[:, :, indices]
    print(f"Direct indexing result: {up2[0, 0, :10]}...")

    # Check specific mappings
    print(f"\nIndex mapping (first 20):")
    print(f"Target indices: {list(range(20))}")
    print(f"Source indices: {indices[:20].tolist()}")

    # Full test
    visual_full = torch.randn(16, 256, 75).to(device)


    visual_4d = visual_full.unsqueeze(-1)
    up1_full = F.interpolate(visual_4d, size=(298, 1), mode='nearest').squeeze(-1)

    # Direct indexing with correct mapping
    indices = (torch.arange(298).float() / (298 / 75)).round().long().clamp(0, 74).to(device)
    up2_full = visual_full[:, :, indices]

    print(f"\nFull tensor test:")
    print(f"Max difference: {(up1_full - up2_full).abs().max().item()}")
    print(f"Are they close? {torch.allclose(up1_full, up2_full, atol=1e-5)}")


if __name__ == "__main__":
    debug_upsampling()