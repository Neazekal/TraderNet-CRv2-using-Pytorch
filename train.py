#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from database.datasets.utils import prepare_train_eval_dataset
from environments.factory import build_trading_environment


def train(
    dataset_path: str,
    timeframe_size: int,
    num_eval_samples: int,
    initial_equity: float,
    fee_rate: float,
    slippage_rate: float,
    position_size: float,
    leverage: float,
    agent_class,
    checkpoint_filepath: str,
    train_episode_steps: int,
    train_iterations: int,
    eval_episodes: int,
    n_consecutive_window: int | None = 3,
    **kwargs
):
    train_data, eval_data = prepare_train_eval_dataset(
        dataset_path=dataset_path,
        feature_columns=config.regression_features,
        timeframe_size=timeframe_size,
        num_eval_samples=num_eval_samples,
    )

    # Create training environment wrapped with Monitor and DummyVecEnv for SB3
    def make_train_env():
        env = build_trading_environment(
            market_data=train_data,
            episode_steps=train_episode_steps,
            initial_equity=initial_equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
            leverage=leverage,
            n_consecutive_window=n_consecutive_window,
        )
        return Monitor(env)

    def make_eval_env():
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

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # Initialize agent with SB3 wrapper
    agent = agent_class(
        env=train_env,
        tensorboard_log=checkpoint_filepath,
        verbose=0,
        **kwargs
    )

    # Train the agent
    agent.train(total_timesteps=train_iterations, progress_bar=True)

    # Save the model
    agent.save(os.path.join(checkpoint_filepath, "model"))

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        agent.model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    # Get metrics from the underlying environment
    eval_base_env = eval_env.envs[0].unwrapped
    eval_metrics = eval_base_env.get_metrics()

    return [mean_reward], eval_metrics


def run_experiments(experiment_name: str):
    scenario_name = "Portfolio-Simulator"
    results = {
        agent_name: {dataset_name: {} for dataset_name in config.datasets_dict.keys()}
        for agent_name in config.agent_config.keys()
    }

    os.makedirs(f'experiments/{experiment_name}', exist_ok=True)

    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            print(f"Training {experiment_name} {agent_name} on {dataset_name} with {scenario_name}...")
            torch.manual_seed(0)
            np.random.seed(0)

            checkpoint_filepath = (
                f'database/storage/checkpoints/experiments/{experiment_name}/{agent_name}/{dataset_name}/{scenario_name}/'
            )
            os.makedirs(checkpoint_filepath, exist_ok=True)

            full_dataset_path = config.dataset_save_filepath.format(dataset_filepath)
            train_params = {
                'dataset_path': full_dataset_path,
                'checkpoint_filepath': checkpoint_filepath,
                **config.env_config,
                **agent_params,
            }

            eval_avg_returns, eval_metrics = train(**train_params)
            results[agent_name][dataset_name][scenario_name] = (eval_avg_returns, eval_metrics)

            metric_values = {
                metric.name: [float(np.mean(metric.episode_metrics))] if len(metric.episode_metrics) > 0 else [0.0]
                for metric in eval_metrics
            }
            cumulative_return_values = next(
                (metric.episode_metrics for metric in eval_metrics if metric.name == 'Cumulative Return'),
                [],
            )
            mean_cumulative_return = (
                float(np.mean(cumulative_return_values))
                if len(cumulative_return_values) > 0
                else 0.0
            )
            metrics_dict = {
                'steps': [10000 * i for i in range(len(eval_avg_returns))],
                'average_returns': eval_avg_returns,
                'final_equity': [float(train_params['initial_equity'] * (1.0 + mean_cumulative_return))],
                'cumulative_return': [mean_cumulative_return],
                **metric_values,
            }
            metrics_df = pd.DataFrame(metrics_dict)
            output_csv_path = f'experiments/{experiment_name}/{agent_name}/{dataset_name}_{scenario_name}.csv'
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            metrics_df.to_csv(output_csv_path, index=False)
            print(f"Saved results to {output_csv_path}")


if __name__ == "__main__":
    run_experiments('tradernet')
