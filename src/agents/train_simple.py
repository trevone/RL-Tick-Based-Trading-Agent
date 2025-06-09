# src/agents/train_simple.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.environments.env_loader import load_environments

def train_simple_task():
    print("--- Starting Simplified State Machine Test ---")
    
    # Load the environment
    available_envs = load_environments()
    env_class = available_envs.get("state_machine_test_env")
    if not env_class:
        print("ERROR: state_machine_test_env.py not found in experiments folder.")
        return
        
    # Create a vectorized environment
    # The lambda function passes dummy data (None) as this env doesn't use it
    vec_env = make_vec_env(lambda: env_class(None, None, None), n_envs=4)

    # Use a standard PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1, ent_coef=0.02, n_steps=256)

    # Train for a short period. This task should be learned very quickly.
    model.learn(total_timesteps=50000)

    print("\n--- Testing Trained Agent ---")
    obs = vec_env.reset()
    for _ in range(20): # Run for 20 steps to see the cycle
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        
        # Print status of the first environment
        pos_status = "OPEN" if obs[0][0] == 1.0 else "FLAT"
        action_map = {0: "Hold", 1: "Buy ", 2: "Sell"}
        print(f"Action: {action_map[action[0]]} -> New State: {pos_status}, Reward: {rewards[0]:.2f}")

    vec_env.close()

if __name__ == "__main__":
    train_simple_task()