# tests/environments/test_custom_wrappers.py
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np

# Import the wrapper from the new path
from src.environments.custom_wrappers import FlattenAction

# Apply pytest-order marker
pytestmark = pytest.mark.order(4) #


# Mock a simple environment with a Tuple action space
class MockEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Example Tuple action space: (Discrete for action choice, Box for a continuous parameter)
        self.action_space = Tuple((
            Discrete(3),  # 0: Hold, 1: Buy, 2: Sell
            Box(low=np.array([0.001], dtype=np.float32), high=np.array([0.05], dtype=np.float32), shape=(1,))
        ))
        self.observation_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)

    def step(self, action):
        # In a real env, this would process the action tuple
        assert isinstance(action, tuple)
        assert isinstance(action[0], int)
        assert isinstance(action[1], np.ndarray)
        assert action[1].shape == (1,)
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

def test_flatten_action_wrapper_init():
    """Test initialization of FlattenAction wrapper."""
    mock_env = MockEnv()
    wrapped_env = FlattenAction(mock_env)

    # Expected flattened action space
    expected_low = np.array([0.0, 0.001], dtype=np.float32)
    expected_high = np.array([2.0, 0.05], dtype=np.float32) # Discrete(3) maps to 0, 1, 2
    expected_shape = (2,) # 1 discrete + 1 continuous param

    assert isinstance(wrapped_env.action_space, Box)
    assert np.allclose(wrapped_env.action_space.low, expected_low)
    assert np.allclose(wrapped_env.action_space.high, expected_high)
    assert wrapped_env.action_space.shape == expected_shape
    assert wrapped_env.action_space.dtype == np.float32

def test_flatten_action_wrapper_action_conversion():
    """Test conversion from flattened Box action to original Tuple action."""
    mock_env = MockEnv()
    wrapped_env = FlattenAction(mock_env)

    # Test 'Hold' action
    flattened_action_hold = np.array([0.0, 0.02], dtype=np.float32) # 0.0 should map to discrete 0
    converted_action_hold = wrapped_env.action(flattened_action_hold)
    assert converted_action_hold == (0, np.array([0.02], dtype=np.float32))

    # Test 'Buy' action
    flattened_action_buy = np.array([1.0, 0.005], dtype=np.float32) # 1.0 should map to discrete 1
    converted_action_buy = wrapped_env.action(flattened_action_buy)
    assert converted_action_buy == (1, np.array([0.005], dtype=np.float32))

    # Test 'Sell' action
    flattened_action_sell = np.array([2.0, 0.04], dtype=np.float32) # 2.0 should map to discrete 2
    converted_action_sell = wrapped_env.action(flattened_action_sell)
    assert converted_action_sell == (2, np.array([0.04], dtype=np.float32))

    # Test values between discrete actions (should round)
    flattened_action_mid = np.array([0.6, 0.03], dtype=np.float32) # Should round to 1
    converted_action_mid = wrapped_env.action(flattened_action_mid)
    assert converted_action_mid == (1, np.array([0.03], dtype=np.float32))

    # Test values beyond discrete action range (should clamp and round)
    flattened_action_clamped_low = np.array([-0.5, 0.01], dtype=np.float32) # Should clamp to 0 and round to 0
    converted_action_clamped_low = wrapped_env.action(flattened_action_clamped_low)
    assert converted_action_clamped_low == (0, np.array([0.01], dtype=np.float32))

    flattened_action_clamped_high = np.array([2.8, 0.045], dtype=np.float32) # Should clamp to 2 and round to 2
    converted_action_clamped_high = wrapped_env.action(flattened_action_clamped_high)
    assert converted_action_clamped_high == (2, np.array([0.045], dtype=np.float32))

    # Test continuous parameter clipping
    flattened_action_cont_clip_low = np.array([1.0, -0.001], dtype=np.float32) # Should clip to 0.001
    converted_action_cont_clip_low = wrapped_env.action(flattened_action_cont_clip_low)
    assert np.isclose(converted_action_cont_clip_low[1][0], 0.001)

    flattened_action_cont_clip_high = np.array([1.0, 0.1], dtype=np.float32) # Should clip to 0.05
    converted_action_cont_clip_high = wrapped_env.action(flattened_action_cont_clip_high)
    assert np.isclose(converted_action_cont_clip_high[1][0], 0.05)


def test_flatten_action_wrapper_step_integration():
    """Test that the wrapped environment's step method receives the correct action tuple."""
    mock_env = MockEnv()
    wrapped_env = FlattenAction(mock_env)
    wrapped_env.reset()

    # Define a flattened action
    test_flattened_action = np.array([1.2, 0.025], dtype=np.float32) # Should map to (1, 0.025)

    # Call the wrapped environment's step method
    obs, reward, terminated, truncated, info = wrapped_env.step(test_flattened_action)

    # The mock_env.step includes assertions that check the type and content of the action tuple
    # If those assertions pass, it means the wrapper successfully converted the action.
    assert obs is not None
    assert reward == 0.0
    assert not terminated
    assert not truncated