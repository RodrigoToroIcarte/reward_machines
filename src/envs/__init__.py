from gym.envs.registration import register

# ----------------------------------------- ANT

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

# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

# ----------------------------------------- CRAFT
for i in range(1):
    w_id = 'Craft-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )