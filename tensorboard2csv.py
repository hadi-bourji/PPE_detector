import os
import argparse
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_best_metrics_wide(logdir):
    """
    Scans each subdirectory under logdir for TensorBoard logs, extracts the best value (or min for loss)
    and the corresponding epoch for each metric, and pivots results to one row per run with one column
    per metric.
    """
    run_dicts = []
    for run_name in sorted(os.listdir(logdir)):
        run_path = os.path.join(logdir, run_name)
        if not os.path.isdir(run_path):
            continue

        ea = EventAccumulator(run_path)
        ea.Reload()

        tags = ea.Tags().get('scalars', [])
        run_entry = {"run": run_name}
        all_steps = []

        for tag in tags:
            events = ea.Scalars(tag)
            if not events:
                continue

            # Track all steps to compute total epochs
            all_steps.extend(e.step for e in events)

            # Choose min for loss metrics, max otherwise
            if 'loss' in tag.lower():
                best_event = min(events, key=lambda e: e.value)
            else:
                best_event = max(events, key=lambda e: e.value)

            # Sanitize tag for column name
            col_base = tag.replace('/', '_').replace(' ', '_')

            run_entry[f"{col_base}_best_value"] = best_event.value
            run_entry[f"{col_base}_best_epoch"] = best_event.step

        # Total epochs is the max step seen across all metrics
        run_entry["total_epochs"] = max(all_steps) if all_steps else 0
        run_dicts.append(run_entry)

    # Create DataFrame with one row per run
    df = pd.DataFrame(run_dicts)
    # Optional: sort columns so 'run' and 'total_epochs' are first
    cols = ["run", "total_epochs"] + [c for c in df.columns if c not in ("run", "total_epochs")]
    return df[cols]

def main():
    parser = argparse.ArgumentParser(description="Extract best values per metric (wide format) from TensorBoard logs.")
    parser.add_argument('--logdir', required=True, help="Directory containing run subdirectories with TensorBoard logs.")
    parser.add_argument('--out_csv', required=True, help="Output CSV file path.")
    args = parser.parse_args()

    df = extract_best_metrics_wide(args.logdir)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved wide-format best metrics summary to {args.out_csv}")

if __name__ == "__main__":
    main()
