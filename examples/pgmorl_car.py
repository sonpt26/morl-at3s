import numpy as np

from morl_baselines.common.evaluation import eval_mo
# from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.single_policy.ser.mo_ppo import make_env


if __name__ == "__main__":
    env_id = "mo-mountaincarcontinuous-v0"
    algo = CAPQL(env=env_id)
    algo.train(
        total_timesteps=int(1e3),
        eval_env=make_env(env_id, 42, 0, "CAPQL_eval_env", gamma=0.995)(),
        ref_point=np.array([-100.0, -100.0]),
        known_pareto_front=None,
    )
    env = make_env(env_id, 422, 1, "CAPQL_test", gamma=0.995)()  # idx != 0 to avoid taking videos

    # Execution of trained policies
    for a in algo.archive.individuals:
        scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
            agent=a, env=env, w=np.array([1.0, 1.0]), render=True
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")
