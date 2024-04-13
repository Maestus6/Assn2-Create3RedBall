from gym.envs.registration import register

register(
    id='CreateRedBall',
    entry_point='aisd_examples.envs.create_red_ball:CreateRedBallEnv',
)