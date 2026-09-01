import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results():
    experiments_dir = 'experiments'
    tradernet_dir = os.path.join(experiments_dir, 'tradernet')
    integrated_dir = os.path.join(experiments_dir, 'integrated')
    canonical_suffix = '_Portfolio-Simulator_eval_cumul_pnls.csv'

    data = []
    seen_baseline_datasets = set()

    # 1. TraderNet Results
    if os.path.exists(tradernet_dir):
        for agent_name in sorted(os.listdir(tradernet_dir)):
            agent_dir = os.path.join(tradernet_dir, agent_name)
            if not os.path.isdir(agent_dir):
                continue

            for filename in sorted(os.listdir(agent_dir)):
                if filename.endswith(canonical_suffix):
                    filepath = os.path.join(agent_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                    except Exception:
                        continue

                    if 'cumulative_pnl' in df.columns:
                        for i, val in enumerate(df['cumulative_pnl']):
                            data.append({
                                'Step': i,
                                'Cumulative PnL': val,
                                'Agent': agent_name,
                                'Type': 'Standard'
                            })

                    dataset_name = filename[:-len(canonical_suffix)]
                    if dataset_name not in seen_baseline_datasets and 'buy_and_hold_cumulative_pnl' in df.columns:
                        seen_baseline_datasets.add(dataset_name)
                        for i, val in enumerate(df['buy_and_hold_cumulative_pnl']):
                            data.append({
                                'Step': i,
                                'Cumulative PnL': val,
                                'Agent': f"Buy & Hold ({dataset_name})" if dataset_name else "Buy & Hold",
                                'Type': 'Baseline'
                            })

    # 2. Integrated Results
    if os.path.exists(integrated_dir):
        for agent_name in sorted(os.listdir(integrated_dir)):
            agent_dir = os.path.join(integrated_dir, agent_name)
            if not os.path.isdir(agent_dir):
                continue

            for filename in sorted(os.listdir(agent_dir)):
                if filename.endswith(canonical_suffix):
                    filepath = os.path.join(agent_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                    except Exception:
                        continue

                    if 'cumulative_pnl' not in df.columns:
                        continue

                    for i, val in enumerate(df['cumulative_pnl']):
                        data.append({
                            'Step': i,
                            'Cumulative PnL': val,
                            'Agent': f"{agent_name} (Integrated)",
                            'Type': 'Integrated'
                        })
    if not data:
        print("No result files found in experiments/")
        return

    df_all = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_all, x='Step', y='Cumulative PnL', hue='Agent', style='Type')
    plt.title('Cumulative PnL Comparison (Evaluation)')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)

    output_path = os.path.join(experiments_dir, 'comparison_plot.png')
    os.makedirs(experiments_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_results()
