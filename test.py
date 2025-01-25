import numpy as np
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.utils.hypervolume import HyperVolume

def run_capql_mountaincar():
    # 1) Create the multi-objective Continuous Mountain Car environment
    env_id = "mo-MountainCarContinuous-v0"
    env = mo_gym.make(env_id)

    # 2) Define a reference point for hypervolume
    #    Continuous Mountain Car typically yields negative returns,
    #    so pick something more negative than you'd expect in practice.
    ref_point = np.array([-200.0, -200.0])

    # 3) Initialize the CAPQL agent
    #    'pop_size' is the number of policies in the Pareto population
    agent = CAPQL(
        env=env,
        ref_point=ref_point,
        pop_size=4,         # Key parameter (does exist in Lucas Alegre’s CAPQL)
        gamma=0.99,
        net_arch=[64, 64],
        alpha=3e-4,         # LR
        tau=0.005,
        policy="Gaussian",
        buffer_size=100000,
        exploration_steps=1000,
        verbose=0
    )

    # 4) Create a HyperVolume instance
    hv_metric = HyperVolume(ref_point=ref_point)

    # 5) Define a callback to evaluate hypervolume across the entire population
    def hv_callback(_locals, _globals):
        """
        This callback is called after each environment step by default.
        We'll evaluate each policy in the population, compute hypervolume,
        and print it out.
        """
        # 'self' is the CAPQL agent, or we can just use 'agent' if in scope
        a = _locals["self"]

        # Evaluate each policy in the population
        pop_returns = []
        n_eval_episodes = 3
        for i in range(a.pop_size):
            # returns shape: (n_eval_episodes, n_objectives)
            returns = a.eval(env, n_episodes=n_eval_episodes, policy_idx=i, render=False)
            mean_return = np.mean(returns, axis=0)
            pop_returns.append(mean_return)

        # Compute hypervolume of the set of average returns
        pop_returns = np.array(pop_returns)
        hv_value = hv_metric.compute(pop_returns)
        step = a._total_steps
        print(f"[Step {step}] Hypervolume = {hv_value:.3f}")

        # Return False to keep training, or True to interrupt training
        return False

    # 6) Train CAPQL and track hypervolume
    total_timesteps = 10_000
    agent.learn(total_timesteps=total_timesteps, callback=hv_callback)
    print("Training complete.")

    # 7) Final evaluation after training (optional)
    print("Evaluating final population:")
    final_pop_returns = []
    for i in range(agent.pop_size):
        rets = agent.eval(env, n_episodes=5, policy_idx=i)
        final_pop_returns.append(np.mean(rets, axis=0))
    final_pop_returns = np.array(final_pop_returns)
    final_hv = hv_metric.compute(final_pop_returns)
    print(f"Final Population Hypervolume: {final_hv:.3f}")
    print("Population average returns:")
    for i, r in enumerate(final_pop_returns):
        print(f" Policy {i} average returns: {r}")

if __name__ == "__main__":
    run_capql_mountaincar()
