'''__init__ to register the gym env'''
from env.single_agent_env import SingleAgentEnv
from gymnasium.envs.registration import register

register(
     id='SingleAgentEnv-v0',
     entry_point='env.single_agent_env:SingleAgentEnv',
     max_episode_steps=1000,
)
