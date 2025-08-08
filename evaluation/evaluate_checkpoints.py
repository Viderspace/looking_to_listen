# evaluate_checkpoints.py
"""
a script to evaluate multiple checkpoints (trained models, at different stages) of the Looking to Listen model
on two validation sets:

  1) 7k_noises: Test set with noise types seen during training
  2) 7k_speech_dubs: Robustness test with overlapping speech (unseen during training)
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import random
import numpy as np

from model.av_model import AudioVisualModel
import metrics
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for no display
import matplotlib.pyplot as plt


def plot_metric_progression(summary: Dict, output_path: Path):
    """Plot metric progression across checkpoints"""

    checkpoints = sorted(summary.keys(),
                         key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metric Progression Across Checkpoints', fontsize=14)

    metrics = ['avg_sdr', 'avg_sdr_improvement', 'avg_pesq']

    for row, val_set in enumerate(['7k_noises', '7k_speech_dubs']):
        for col, metric in enumerate(metrics):
            values = [summary[cp][val_set].get(metric, 0) for cp in checkpoints]

            ax = axes[row, col]
            ax.plot(range(len(checkpoints)), values, 'o-', linewidth=2, markersize=8)
            ax.set_xticks(range(len(checkpoints)))
            ax.set_xticklabels([cp.split('_')[-1] for cp in checkpoints], rotation=45)
            ax.set_title(f"{val_set}: {metric.replace('avg_', '').upper()}")
            ax.set_ylabel('dB' if 'sdr' in metric else 'Score')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'metric_progression.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path / 'metric_progression.png'}")
    plt.show()  # Comment out if issues persist



def stft_to_audio(stft_compressed: torch.Tensor, power_law_p: float = 0.3) -> torch.Tensor:
    """Convert compressed STFT back to audio waveform"""
    # Extract compressed real and imaginary parts
    real_compressed = stft_compressed[..., 0]
    imag_compressed = stft_compressed[..., 1]

    # Invert the power-law compression
    inverse_power = 1.0 / power_law_p
    real_decompressed = torch.sign(real_compressed) * torch.abs(real_compressed).pow(inverse_power)
    imag_decompressed = torch.sign(imag_compressed) * torch.abs(imag_compressed).pow(inverse_power)

    # Convert to complex tensor
    stft_complex = torch.complex(real_decompressed, imag_decompressed)

    # Create window
    window = torch.hann_window(400, device=stft_complex.device)

    # ISTFT
    audio = torch.istft(
            stft_complex,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            center=True,
            normalized=False,
            onesided=True
    )

    return audio


def find_checkpoints(root_path: Path) -> List[Tuple[str, Path]]:
    """
    Recursively find all .pt checkpoint files
    returning a pair of (name, path) for each checkpoint. ordered by name (ascending).
    """

    checkpoints = []
    for path in root_path.rglob('*.pt'):
        if path.is_file():
            name = path.stem  # Use filename without extension as name
            checkpoints.append((name, path))

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {root_path}")

    # Sort checkpoints by name
    return sorted(checkpoints, key=lambda x: x[0])


def load_validation_sets(validation_root: Path, test_limit: int) -> Tuple[List[Path], List[Path]]:
    """Load and randomly select samples from validation sets

    Returns:
        Tuple of (noises_samples, speech_dubs_samples)
    """
    noises_path = validation_root / '7k_noises'
    speech_dubs_path = validation_root / '7k_speech_dubs'

    # Load noises samples
    if not noises_path.exists():
        raise ValueError(f"Validation set 7k_noises not found at {noises_path}")

    all_noises = sorted([f for f in noises_path.iterdir() if f.is_dir()])
    num_noises = min(test_limit, len(all_noises))
    selected_noises = random.sample(all_noises, num_noises)

    # Load speech_dubs samples
    if not speech_dubs_path.exists():
        raise ValueError(f"Validation set 7k_speech_dubs not found at {speech_dubs_path}")

    all_speech_dubs = sorted([f for f in speech_dubs_path.iterdir() if f.is_dir()])
    num_speech_dubs = min(test_limit, len(all_speech_dubs))
    selected_speech_dubs = random.sample(all_speech_dubs, num_speech_dubs)

    return selected_noises, selected_speech_dubs


def load_checkpoint(checkpoint_path: Path, device: str) -> AudioVisualModel:
    """Load model from checkpoint"""
    model = AudioVisualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_single_sample(model: AudioVisualModel,
                           sample_path: Path,
                           device: str) -> Dict[str, float]:
    """Evaluate one sample and return metrics"""
    # Load embeddings
    clean = torch.load(sample_path / 'audio' / 'clean_embs.pt')
    mixture = torch.load(sample_path / 'audio' / 'mixture_embs.pt')
    face = torch.load(sample_path / 'face' / 'face_embs.pt')

    # Add batch dimension and move to device
    clean = clean.unsqueeze(0).to(device)
    mixture = mixture.unsqueeze(0).to(device)
    face = face.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        masks = model(mixture, face)
        separated = mixture * masks

    # Convert STFT to audio
    separated_audio = stft_to_audio(separated.squeeze(0))
    clean_audio = stft_to_audio(clean.squeeze(0))
    mixture_audio = stft_to_audio(mixture.squeeze(0))

    # Compute metrics
    return {
            'sdr'            : metrics.sdr(separated_audio, clean_audio),
            'sdr_improvement': metrics.sdr_improvement(separated_audio, clean_audio, mixture_audio),
            'si_sdr'         : metrics.si_sdr(separated_audio, clean_audio),
            'pesq'           : metrics.pesq(separated_audio, clean_audio),
            'stoi'           : metrics.stoi(separated_audio, clean_audio)
    }


def evaluate_validation_set(model: AudioVisualModel,
                            samples: List[Path],
                            set_name: str,
                            device: str) -> List[Dict[str, float]]:
    """Evaluate model on a validation set"""
    results = []
    for sample_path in tqdm(samples, desc=f"Evaluating {set_name}"):
        try:
            sample_metrics = evaluate_single_sample(model, sample_path, device)
            results.append(sample_metrics)
        except Exception as e:
            print(f"Error processing {sample_path.name}: {e}")
            continue
    return results


def compute_average_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute average of each metric across results"""
    if not results:
        return {}

    avg_metrics = {}
    failed_counts = {}

    for metric_name in results[0].keys():
        values = [r[metric_name] for r in results if r[metric_name] is not None]
        failed = len(results) - len(values)

        if values:
            avg_metrics[f'avg_{metric_name}'] = sum(values) / len(values)

        if failed > 0:
            failed_counts[metric_name] = failed

    # Report failures if any
    if failed_counts:
        metrics_str = ', '.join([f"{k}: {v}/{len(results)}" for k, v in failed_counts.items()])
        print(f"  Warning: Failed computations - {metrics_str}")

    return avg_metrics


def create_summary_table(results: Dict) -> Dict:
    """Create a summary table with all checkpoints and their average metrics"""
    summary = {}

    for checkpoint_name, checkpoint_data in results.items():
        # Skip special keys like '_summary', '_best_worst', etc.
        if checkpoint_name.startswith('_'):
            continue

        summary[checkpoint_name] = {}

        # Process each validation set
        for val_set_name, val_set_results in checkpoint_data.items():
            if val_set_results:
                avg_metrics = compute_average_metrics(val_set_results)
                summary[checkpoint_name][val_set_name] = avg_metrics

    return summary


def find_best_worst_checkpoints(summary: Dict) -> Dict:
    """Find best and worst performing checkpoints for each metric"""
    best_worst = {}

    # Collect all metrics across all validation sets
    all_metrics = set()
    for checkpoint_data in summary.values():
        for val_set_data in checkpoint_data.values():
            all_metrics.update(val_set_data.keys())

    # For each validation set and metric, find best and worst
    val_sets = list(next(iter(summary.values())).keys())

    for val_set in val_sets:
        best_worst[val_set] = {}

        for metric in all_metrics:
            values = []
            for checkpoint_name, checkpoint_data in summary.items():
                if val_set in checkpoint_data and metric in checkpoint_data[val_set]:
                    values.append((checkpoint_name, checkpoint_data[val_set][metric]))

            if values:
                # Sort by value
                sorted_values = sorted(values, key=lambda x: x[1])

                # Determine if higher is better (for most metrics except maybe some specific ones)
                # Generally, higher is better for SDR, SI-SDR, PESQ, STOI
                best_worst[val_set][metric] = {
                        'best'      : sorted_values[-1],  # (checkpoint_name, value)
                        'worst'     : sorted_values[0],
                        'all_values': sorted_values
                }

    return best_worst


def print_summary_table(summary: Dict):
    """Print a formatted summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - Average Metrics per Checkpoint")
    print("=" * 80)

    # Get validation sets and metrics
    val_sets = list(next(iter(summary.values())).keys())

    for val_set in val_sets:
        print(f"\n{val_set}:")
        print("-" * 60)

        # Collect all metrics for this validation set
        metrics = set()
        for checkpoint_data in summary.values():
            if val_set in checkpoint_data:
                metrics.update(checkpoint_data[val_set].keys())
        metrics = sorted(list(metrics))

        # Print header
        header = f"{'Checkpoint':<30}"
        for metric in metrics:
            # Shorten metric names for display
            metric_short = metric.replace('avg_', '').upper()[:8]
            header += f"{metric_short:>12}"
        print(header)
        print("-" * (30 + 12 * len(metrics)))

        # Print values for each checkpoint
        for checkpoint_name, checkpoint_data in summary.items():
            row = f"{checkpoint_name[:29]:<30}"
            if val_set in checkpoint_data:
                for metric in metrics:
                    if metric in checkpoint_data[val_set]:
                        value = checkpoint_data[val_set][metric]
                        row += f"{value:>12.3f}"
                    else:
                        row += f"{'N/A':>12}"
            print(row)


def print_best_worst_summary(best_worst: Dict):
    """Print best and worst performing checkpoints"""
    print("\n" + "=" * 80)
    print("BEST AND WORST PERFORMING CHECKPOINTS")
    print("=" * 80)

    for val_set, metrics_data in best_worst.items():
        print(f"\n{val_set}:")
        print("-" * 60)

        for metric, data in metrics_data.items():
            metric_display = metric.replace('avg_', '').upper()
            best_name, best_value = data['best']
            worst_name, worst_value = data['worst']

            print(f"\n  {metric_display}:")
            print(f"    Best:  {best_name:<30} ({best_value:.3f})")
            print(f"    Worst: {worst_name:<30} ({worst_value:.3f})")
            print(f"    Range: {best_value - worst_value:.3f}")


def compute_checkpoint_statistics(summary: Dict) -> Dict:
    """Compute statistics across all checkpoints for each metric"""
    stats = {}

    # Get validation sets
    val_sets = list(next(iter(summary.values())).keys())

    for val_set in val_sets:
        stats[val_set] = {}

        # Collect all metrics
        metrics = set()
        for checkpoint_data in summary.values():
            if val_set in checkpoint_data:
                metrics.update(checkpoint_data[val_set].keys())

        for metric in metrics:
            values = []
            for checkpoint_data in summary.values():
                if val_set in checkpoint_data and metric in checkpoint_data[val_set]:
                    values.append(checkpoint_data[val_set][metric])

            if values:
                stats[val_set][metric] = {
                        'mean'  : np.mean(values),
                        'std'   : np.std(values),
                        'min'   : np.min(values),
                        'max'   : np.max(values),
                        'median': np.median(values)
                }

    return stats


def print_overall_statistics(stats: Dict):
    """Print overall statistics across checkpoints"""
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS ACROSS ALL CHECKPOINTS")
    print("=" * 80)

    for val_set, metrics_data in stats.items():
        print(f"\n{val_set}:")
        print("-" * 60)

        for metric, metric_stats in metrics_data.items():
            metric_display = metric.replace('avg_', '').upper()
            print(f"\n  {metric_display}:")
            print(f"    Mean:   {metric_stats['mean']:.3f} (± {metric_stats['std']:.3f})")
            print(f"    Median: {metric_stats['median']:.3f}")
            print(f"    Min:    {metric_stats['min']:.3f}")
            print(f"    Max:    {metric_stats['max']:.3f}")


def save_results(results: Dict, output_path: Path = Path('evaluation_results.json')):
    """Save evaluation results to JSON file"""

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(v) for v in obj)
        return obj

    results_converted = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)


def print_header(text: str):
    """Print formatted section header"""
    print(f"\n{'=' * 50}")
    print(f"{text}")
    print(f"{'=' * 50}")


def print_summary(set_name: str, results: List[Dict[str, float]]):
    """Print evaluation summary for a validation set"""
    if results:
        avg_metrics = compute_average_metrics(results)
        if 'avg_sdr_improvement' in avg_metrics:
            print(f"{set_name} - Average SDR improvement: {avg_metrics['avg_sdr_improvement']:.2f} dB")


def get_commandline_args(print_info: bool = True) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Looking to Listen checkpoints')
    parser.add_argument('--checkpoints_root', type=str, required=True,
                        help='Path to root folder containing checkpoints')
    parser.add_argument('--validation_root', type=str, required=True,
                        help='Path to folder containing both 7k validation sets')
    parser.add_argument('--test_limit', type=int, default=2000,
                        help='Max samples to evaluate per validation set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file path')
    parser.add_argument('--summary_only', action='store_true',
                        help='Save only summary statistics, not individual results')

    args = parser.parse_args()

    if print_info:  # iterating all parameters and printing them
        print("Command line arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print("=" * 50)

    return args


def set_seed(value: int):
    # Set random seed for reproducibility
    random.seed(value)
    torch.manual_seed(value)
    print(f"Random seed: {value}")


def get_device() -> str:
    """Get the best available device for PyTorch"""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device


def check_dependencies():
    """Check if optional dependencies are available"""
    missing = []
    try:
        import pesq
    except ImportError:
        missing.append('pesq')

    try:
        import pystoi
    except ImportError:
        missing.append('pystoi')

    if missing:
        print(f"Warning: Optional dependencies missing: {', '.join(missing)}")
        print("  Install with: pip install " + ' '.join(missing))
        print("  SDR metrics will still be computed.\n")


def main():
    args = get_commandline_args()
    set_seed(args.seed)
    device = get_device()
    check_dependencies()

    # Find all checkpoints
    checkpoints_root = Path(args.checkpoints_root)
    checkpoints = find_checkpoints(checkpoints_root)
    print(f"Found {len(checkpoints)} checkpoints")
    for name, _ in checkpoints:
        print(f"  - {name}")

    # Load and select validation samples
    validation_root = Path(args.validation_root)
    noises_samples, speech_dubs_samples = load_validation_sets(validation_root, args.test_limit)
    print(f"Selected {len(noises_samples)} samples from 7k_noises")
    print(f"Selected {len(speech_dubs_samples)} samples from 7k_speech_dubs")

    # Main evaluation loop
    results = {}

    for checkpoint_name, checkpoint_path in checkpoints:
        print_header(f"Evaluating: {checkpoint_name}")

        # Load model
        model = load_checkpoint(checkpoint_path, device)
        results[checkpoint_name] = {}

        # Evaluate on noises set
        print(f"\nValidation set: 7k_noises")
        noises_results = evaluate_validation_set(model, noises_samples, '7k_noises', device)
        results[checkpoint_name]['7k_noises'] = noises_results
        print_summary('7k_noises', noises_results)

        # Evaluate on speech_dubs set
        print(f"\nValidation set: 7k_speech_dubs")
        speech_dubs_results = evaluate_validation_set(model, speech_dubs_samples, '7k_speech_dubs', device)
        results[checkpoint_name]['7k_speech_dubs'] = speech_dubs_results
        print_summary('7k_speech_dubs', speech_dubs_results)

    # Generate and print summary statistics
    summary = create_summary_table(results)
    best_worst = find_best_worst_checkpoints(summary)
    stats = compute_checkpoint_statistics(summary)

    print_summary_table(summary)
    print_best_worst_summary(best_worst)
    print_overall_statistics(stats)
    plot_metric_progression(summary, Path(args.output).parent)

    # Prepare data to save based on summary_only flag
    if args.summary_only:
        save_data = {
                'summary'   : summary,
                'best_worst': best_worst,
                'statistics': stats
        }
    else:
        # Add summaries to full results
        results['_summary'] = summary
        results['_best_worst'] = best_worst
        results['_statistics'] = stats
        save_data = results

    # Save results
    output_path = Path(args.output)
    save_results(save_data, output_path)
    print(f"\nResults saved to {output_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()