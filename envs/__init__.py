from gym.envs.registration import register

register(
    id="rnaenv-v0",
    entry_point="envs.gym_envs:Rnaenv_v0",
    kwargs={'dataset': 'datasets/expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
)

