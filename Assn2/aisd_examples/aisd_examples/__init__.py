from gym.envs.registration import register

register(
    id='CreateRedBall-v0',
    entry_point='aisd_examples.envs:CreateRedBallEnv',
)
