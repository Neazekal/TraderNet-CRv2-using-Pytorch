#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from database.datasets.utils import prepare_eval_dataset
from environments.factory import build_trading_environment
from environments.actions import Action
from evaluation import run_episode

def build_eval_env(
    dataset_path: str,
    timeframe_size: int,
    num_eval_samples: int,
    initial_equity: float,
    fee_rate: float,
    slippage_rate: float,
    position_size: float = 1.0,
    leverage: float = 1.0,
    n_consecutive_window: int | None = None,
    **kwargs
) -> DummyVecEnv:
    eval_data = prepare_eval_dataset(
        dataset_path=dataset_path,
        feature_columns=config.regression_features,
        timeframe_size=timeframe_size,
        num_eval_samples=num_eval_samples,
    )
    def make_env():
        env = build_trading_environment(
            market_data=eval_data,
            episode_steps=len(eval_data.execution_opens),
            initial_equity=initial_equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
            leverage=leverage,
            n_consecutive_window=n_consecutive_window,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    return env


def load_agent(agent_class, checkpoint_filepath, env, device: str = "cpu"):
    model_path = os.path.join(checkpoint_filepath, "model")
    if not os.path.exists(model_path + ".zip"):
        print(f"Warning: Model not found at {model_path}.zip")
        return None

    try:
        agent = agent_class.load(model_path, env=env, device=device)
        return agent
    except Exception as e:
        print(f"Error loading agent from {model_path}: {e}")
        return None


def eval_tradernet(agent, env) -> tuple[float, list[dict]]:
    def select_action(obs: np.ndarray) -> np.ndarray:
        action, _ = agent.predict(obs, deterministic=True)
        return action

    result = run_episode(env, select_action)
    return result.total_reward, list(result.steps)


def eval_buy_and_hold(env) -> tuple[float, list[dict]]:
    is_first_step = True

    def select_action(obs: np.ndarray) -> np.ndarray:
        nonlocal is_first_step
        if is_first_step:
            is_first_step = False
            return np.array([Action.BUY.value])
        return np.array([Action.HOLD.value])

    result = run_episode(env, select_action)
    return result.total_reward, list(result.steps)


if __name__ == "__main__":
    scenario_name = "Portfolio-Simulator"

    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            print(f"Evaluating {agent_name} on {dataset_name} with {scenario_name}...")

            full_dataset_path = config.dataset_save_filepath.format(dataset_filepath)
            env = build_eval_env(
                dataset_path=full_dataset_path,
                **config.env_config
            )

            checkpoint_path = (
                f'database/storage/checkpoints/experiments/tradernet/{agent_name}/{dataset_name}/{scenario_name}/'
            )
            agent = load_agent(
                agent_class=agent_params['agent_class'],
                checkpoint_filepath=checkpoint_path,
                env=env
            )

            if agent is None:
                print(f"Skipping {agent_name} on {dataset_name} due to missing model at {checkpoint_path}")
                continue

            average_returns, step_records = eval_tradernet(
                agent=agent.model,
                env=env
            )

            bh_env_config = {**config.env_config, 'n_consecutive_window': None}
            bh_env = build_eval_env(
                dataset_path=full_dataset_path,
                **bh_env_config
            )
            bh_rewards, bh_step_records = eval_buy_and_hold(
                env=bh_env
            )

            if len(step_records) != len(bh_step_records):
                raise ValueError(
                    f"Record length mismatch: learned has {len(step_records)} records, "
                    f"buy-and-hold has {len(bh_step_records)} records."
                )

            for idx, (lr_rec, bh_rec) in enumerate(zip(step_records, bh_step_records)):
                if lr_rec['step_index'] != bh_rec['step_index'] or lr_rec['timestamp'] != bh_rec['timestamp']:
                    raise ValueError(
                        f"Alignment mismatch at step {idx}: "
                        f"learned step_index={lr_rec.get('step_index')}, timestamp={lr_rec.get('timestamp')} vs "
                        f"buy-and-hold step_index={bh_rec.get('step_index')}, timestamp={bh_rec.get('timestamp')}"
                    )

            base_env = env.envs[0].unwrapped
            episode_metrics = base_env.get_metrics()

            final_equity = step_records[-1]['equity'] if step_records else config.env_config['initial_equity']
            cumulative_return = step_records[-1]['cumulative_return'] if step_records else 0.0

            bh_final_equity = bh_step_records[-1]['equity'] if bh_step_records else config.env_config['initial_equity']
            bh_cumulative_return = bh_step_records[-1]['cumulative_return'] if bh_step_records else 0.0
            excess_cumulative_return = cumulative_return - bh_cumulative_return

            metrics = {
                'average_returns': [average_returns],
                'final_equity': [final_equity],
                'cumulative_return': [cumulative_return],
                'buy_and_hold_final_equity': [bh_final_equity],
                'buy_and_hold_cumulative_return': [bh_cumulative_return],
                'excess_cumulative_return': [excess_cumulative_return],
                **{
                    metric.name: [float(np.mean(metric.episode_metrics))] if len(metric.episode_metrics) > 0 else [float(metric.result())]
                    for metric in episode_metrics
                }
            }
            results_df = pd.DataFrame(metrics)

            output_metrics_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{scenario_name}_metrics.csv'
            os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
            results_df.to_csv(output_metrics_path, index=False)

            print(results_df, '\n')

            episode_pnls_df = pd.DataFrame(step_records)
            episode_pnls_df['buy_and_hold_position'] = [r['position'] for r in bh_step_records]
            episode_pnls_df['buy_and_hold_equity'] = [r['equity'] for r in bh_step_records]
            episode_pnls_df['buy_and_hold_cumulative_pnl'] = [r['cumulative_pnl'] for r in bh_step_records]
            episode_pnls_df['buy_and_hold_cumulative_return'] = [r['cumulative_return'] for r in bh_step_records]

            output_pnls_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{scenario_name}_eval_cumul_pnls.csv'
            episode_pnls_df.to_csv(output_pnls_path, index=False)

            print(episode_pnls_df[['step_index', 'position', 'equity', 'cumulative_pnl', 'cumulative_return', 'buy_and_hold_equity', 'buy_and_hold_cumulative_pnl']].tail(5))
