# custom_wrappers.py
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np

class FlattenAction(gym.ActionWrapper):
    """
    A custom action wrapper to flatten a Tuple action space
    (Discrete, Box) into a single Box action space.

    The new Box space will combine the discrete choices with
    the continuous parameters.

    Example:
        Original action space: Tuple(Discrete(3), Box(low=0.001, high=0.05, shape=(1,), float32))
        Flattened action space: Box(low=[-1.0, 0.001], high=[1.0, 0.05], shape=(2,), float32)
            Where the first dim [-1, 1] represents Discrete(3) (e.g., -1=Hold, 0=Buy, 1=Sell)
            and the second dim [0.001, 0.05] is the original Box action.
            The discrete value is then mapped back.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Check if the environment's action space is a Tuple
        if not isinstance(env.action_space, Tuple):
            raise ValueError(f"Action space must be a Tuple, but got {type(env.action_space)}")

        # Assuming Tuple(Discrete, Box) structure
        self.discrete_space = env.action_space.spaces[0]
        self.box_space = env.action_space.spaces[1]

        if not isinstance(self.discrete_space, Discrete):
            raise ValueError(f"First element of Tuple must be Discrete, but got {type(self.discrete_space)}")
        if not isinstance(self.box_space, Box):
            raise ValueError(f"Second element of Tuple must be Box, but got {type(self.box_space)}")

        # Define the new flattened action space (Box)
        # We need to map discrete actions (0, 1, 2) to a continuous range, e.g., 0 to num_discrete-1
        # and then combine with the Box space.
        
        # Action mapping:
        # 0: Hold --> -1.0
        # 1: Buy  -->  0.0
        # 2: Sell -->  1.0
        # This mapping can be adjusted. The agent will learn what values to output.
        self.num_discrete_actions = self.discrete_space.n
        
        # The new low will be the minimum of our discrete mapping and the Box low
        # The new high will be the maximum of our discrete mapping and the Box high
        
        # For simplicity, we can create a continuous action where the first element
        # represents the discrete choice, and the second element is the continuous parameter.
        # The agent will output continuous values for the first element, which we will discretize.

        # Example:
        # Agent outputs a 2-element array: [action_choice_continuous, profit_target_value]
        # action_choice_continuous will be mapped to 0, 1, or 2 (Hold, Buy, Sell)
        
        # Define the bounds of the new Box action space
        # First dimension: discrete action (mapped to a continuous range, e.g., 0 to num_discrete-1)
        # Second dimension: continuous profit target
        
        low_bounds = np.array([0.0] + self.box_space.low.tolist(), dtype=np.float32)
        high_bounds = np.array([self.num_discrete_actions - 1.0] + self.box_space.high.tolist(), dtype=np.float32)

        self.action_space = Box(low=low_bounds, high=high_bounds, shape=(1 + self.box_space.shape[0],), dtype=np.float32)
        
        print(f"Original action space: {env.action_space}")
        print(f"Flattened action space: {self.action_space}")

    def action(self, action: np.ndarray) -> tuple:
        """
        Converts the flattened action from the agent back to the environment's Tuple action.
        """
        # The agent outputs a continuous array (e.g., [action_choice_continuous, profit_target_value])
        # We need to discretize the first element back to 0, 1, or 2.
        
        # Discretize the first element (action choice)
        discrete_action_raw = action[0]
        # Clamp to bounds [0, num_discrete_actions - 1]
        discrete_action_clamped = np.clip(discrete_action_raw, 0, self.num_discrete_actions - 1)
        # Round to nearest integer (0, 1, or 2)
        discrete_action_int = int(np.round(discrete_action_clamped))

        # The second element is the continuous profit target, clamped to its original bounds
        profit_target_param = np.array([np.clip(action[1], self.box_space.low[0], self.box_space.high[0])], dtype=np.float32)
        
        # Return the action in the environment's expected Tuple format
        return (discrete_action_int, profit_target_param)

    def reverse_action(self, action: tuple) -> np.ndarray:
        """
        Not typically used for training but here for completeness if needed by some tools.
        Converts environment's Tuple action back to flattened action for the agent.
        """
        discrete_part, box_part = action
        
        # Map discrete_part (0, 1, 2) back to a continuous range, e.g., 0, 1, 2
        # (This is implicitly handled by `action`'s mapping)
        
        # Concatenate
        return np.array([float(discrete_part)] + box_part.tolist(), dtype=np.float32)