from database.entities.crypto import Crypto

# --- Database ---
ohlcv_dataset_period_id = '1h'
ohlcv_history_filepath = 'database/storage/downloads/ohlcv/{}.csv'
dataset_save_filepath = 'database/storage/datasets/{}.csv'

regression_features = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs', 'obv_diffs'
]

# --- Model ---
checkpoint_dir = 'database/storage/checkpoints/'

# --- Training Configuration ---
import torch
from agents.torch.ppo_agent import PPOAgent
from agents.torch.dqn_agent import DQNAgent

# Datasets to train on
datasets_dict = {'DOGEUSDT': 'DOGEUSDT'}

# Environment parameters
env_config = {
    'timeframe_size': 12,
    'num_eval_samples': 2250,
    'initial_equity': 10000.0,
    'fee_rate': 0.007,
    'slippage_rate': 0.0005,
    'position_size': 1.0,       # Percentage of capital per trade (0.0 to 1.0)
    'leverage': 1.0,            # Leverage multiplier (e.g., 1.0, 5.0, 10.0)
    'train_episode_steps': 100, # Steps per episode during training
    'eval_episodes': 1,         # Number of episodes to evaluate
    'n_consecutive_window': 3
}

# Agent parameters
agent_config = {
    'PPO': {
        'agent_class': PPOAgent,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'train_iterations': 100000,
        'device': 'cpu' # Force CPU to avoid SB3 warning for MlpPolicy
    },
    'DDQN': {
        'agent_class': DQNAgent,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'train_iterations': 100000,
        'device': 'cpu'
    }
}
