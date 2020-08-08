from gym.envs.registration import register


register(
    id='Ant-RM1-v0',
    entry_point='envs.ant.ant_environment:MyAntEnvRM1',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)



register(
    id='Ant-RM4-v0',
    entry_point='envs.ant.ant_environment:MyAntEnvRM4',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


register(
    id='Ant-RM5-v0',
    entry_point='envs.ant.ant_environment:MyAntEnvRM5',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

