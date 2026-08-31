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
        dataset_filepath,
        timeframe_size,
        target_horizon_len,
        num_eval_samples,
        fees,
        reward_fn_instance,
        agent_class,
        checkpoint_filepath,
        train_episode_steps,
        train_iterations,
        eval_episodes,
        position_size=1.0,
        leverage=1.0,
        n_consecutive_window=3,
        reward_wrapper=None,
        **kwargs
):
    x_train, train_reward_fn, x_eval, eval_reward_fn = prepare_train_eval_dataset(
        dataset_path=config.dataset_save_filepath.format(dataset_filepath),
        feature_columns=config.regression_features,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        reward_fn_factory=reward_fn_instance,
        position_size=position_size,
        leverage=leverage,
        reward_wrapper=reward_wrapper
    )

    # Create training environment wrapped with Monitor and DummyVecEnv for SB3
    def make_train_env():
        env = build_trading_environment(
            states=x_train,
            reward_fn=train_reward_fn,
            episode_steps=train_episode_steps,
            n_consecutive_window=n_consecutive_window
        )
        return Monitor(env)

    def make_eval_env():
        env = build_trading_environment(
            states=x_eval,
            reward_fn=eval_reward_fn,
            episode_steps=x_eval.shape[0] - 1,
            n_consecutive_window=n_consecutive_window
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
    agent.save(checkpoint_filepath + 'model')

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        agent.model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    # Get metrics from the underlying environment
    eval_base_env = eval_env.envs[0].unwrapped
    eval_metrics = eval_base_env.get_metrics()

    return [mean_reward], eval_metrics


def run_experiments(experiment_name: str, reward_wrapper=None):
    results = {
        agent_name: {dataset_name: {} for dataset_name in config.datasets_dict.keys()}
        for agent_name in config.agent_config.keys()
    }

    os.makedirs(f'experiments/{experiment_name}', exist_ok=True)

    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            for reward_fn_name, reward_fn_instance in config.reward_config.items():
                print(f"Training {experiment_name} {agent_name} on {dataset_name} with {reward_fn_name}...")
                torch.manual_seed(0)
                np.random.seed(0)

                checkpoint_filepath = f'database/storage/checkpoints/experiments/{experiment_name}/{agent_name}/{dataset_name}/{reward_fn_name}/'
                os.makedirs(checkpoint_filepath, exist_ok=True)

                train_params = {
                    'dataset_filepath': dataset_filepath,
                    'reward_fn_instance': reward_fn_instance,
                    'checkpoint_filepath': checkpoint_filepath,
                    'reward_wrapper': reward_wrapper,
                    **config.env_config,
                    **agent_params
                }

                eval_avg_returns, eval_metrics = train(**train_params)

                results[agent_name][dataset_name][reward_fn_name] = (eval_avg_returns, eval_metrics)

            for reward_fn_name, reward_fn_results in results[agent_name][dataset_name].items():
                eval_avg_returns, eval_metrics = reward_fn_results

                metrics_dict = {
                    'steps': [10000 * i for i in range(len(eval_avg_returns))],
                    'average_returns': eval_avg_returns,
                    **{
                        metric.name: [float(np.mean(metric.episode_metrics))] if len(metric.episode_metrics) > 0 else [0.0]
                        for metric in eval_metrics
                    }
                }
                metrics_df = pd.DataFrame(metrics_dict)
                output_csv_path = f'experiments/{experiment_name}/{agent_name}/{dataset_name}_{reward_fn_name}.csv'
                os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                metrics_df.to_csv(output_csv_path, index=False)
                print(f"Saved results to {output_csv_path}")


if __name__ == "__main__":
    run_experiments('tradernet')
