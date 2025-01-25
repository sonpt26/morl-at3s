import matplotlib.pyplot as plt

class PerformanceEvaluator:
    """
    Chạy nhiều episode cho mỗi policy,
    thống kê average reward vector (hoặc metric) để so sánh.
    """

    def __init__(self, env, num_episodes=10, max_steps=100):
        """
        env: Môi trường multi-objective (hoặc single-objective).
        num_episodes: số episode để đánh giá.
        max_steps: giới hạn step mỗi episode (nếu environment chưa done).
        """
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def evaluate_policy(self, policy):
        """
        Chạy 'num_episodes' episode với 'policy',
        trả về trung bình reward (hoặc vector reward) qua các episode.
        Ở đây, ta giả định reward là vector [revenue, data_drop].
        """

        # Lưu sum of reward (2D) để tính trung bình
        sum_reward = np.zeros(2, dtype=np.float64)  # [sum_revenue, sum_drop]
        # Hoặc cũng có thể lưu list => phân phối

        for ep in range(self.num_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            ep_reward = np.zeros(2, dtype=np.float64)

            step_count = 0
            while not (done or truncated):
                action = policy.get_action(obs)
                obs, reward_vec, done, truncated, info = self.env.step(action)
                
                # reward_vec = [revenue_step, drop_step]
                # tuỳ logic: ta cộng dồn
                ep_reward += reward_vec

                step_count += 1
                if step_count >= self.max_steps:
                    truncated = True

            # Cộng dồn reward episode này vào sum
            sum_reward += ep_reward

        avg_reward = sum_reward / self.num_episodes
        return avg_reward  # [avg_revenue, avg_drop]

    def compare_policies(self, policies):
        """
        policies: dict { 'policy_name': policy_object }
        => chạy evaluate_policy cho từng policy,
           in/return kết quả so sánh
        """
        results = {}
        for name, pol in policies.items():
            avg_r = self.evaluate_policy(pol)
            results[name] = avg_r
        
        # In ra so sánh
        print("=== Comparison of Policies ===")
        for name, val in results.items():
            print(f"Policy: {name}, Avg Revenue={val[0]:.4f}, Avg Drop={val[1]:.4f}")

        return results

def plot_revenue(results):
    """
    results: dict { policy_name: [avg_revenue, avg_drop], ... }
    Vẽ biểu đồ cột so sánh REVENUE của mỗi policy trên một figure riêng.
    """
    policy_names = list(results.keys())
    revenue_vals = [results[name][0] for name in policy_names]

    plt.figure(figsize=(6, 4))
    plt.bar(policy_names, revenue_vals, color='skyblue')
    plt.title("Average Revenue by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()


def plot_drop(results):
    """
    results: dict { policy_name: [avg_revenue, avg_drop], ... }
    Vẽ biểu đồ cột so sánh DROP của mỗi policy trên một figure riêng.
    """
    policy_names = list(results.keys())
    drop_vals = [results[name][1] for name in policy_names]

    plt.figure(figsize=(6, 4))
    plt.bar(policy_names, drop_vals, color='salmon')
    plt.title("Average Drop by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Drop")
    plt.tight_layout()
    plt.show()

def plot_pareto_front(morl_solutions, ws_solution):
    """
    morl_solutions: list of (drop, revenue) -> các nghiệm Pareto do MORL tìm.
    ws_solution: (drop_ws, revenue_ws) -> lời giải single do WeightedSum.
    
    Vẽ scatter các điểm MORL, nối lại để thấy 'Pareto front', 
    rồi đánh dấu WS solution khác màu.
    """

    # Nếu muốn vẽ đường kết nối front, 
    # ta sort theo drop (hoặc revenue) để mô tả 'biên Pareto' rõ ràng.
    morl_solutions_sorted = sorted(morl_solutions, key=lambda x: x[0])  
    # x[0] là drop

    # Tách list x, y
    drop_vals = [pt[0] for pt in morl_solutions_sorted]
    rev_vals  = [pt[1] for pt in morl_solutions_sorted]

    # Tạo figure
    plt.figure(figsize=(6, 5))

    # Vẽ scatter cho các điểm Pareto
    plt.scatter(drop_vals, rev_vals, color='blue', label='MORL Solutions')
    
    # Nối chúng để thấy 'Pareto front' (tùy ý, không bắt buộc)
    plt.plot(drop_vals, rev_vals, color='blue', alpha=0.5)

    # Đánh dấu Weighted Sum solution
    ws_drop, ws_rev = ws_solution
    plt.scatter([ws_drop], [ws_rev], color='red', marker='X', s=100, label='Weighted Sum')

    # Trang trí
    plt.title("Pareto Front (Drop vs Revenue)")
    plt.xlabel("Data Drop (lower is better)")
    plt.ylabel("Revenue (higher is better)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) Khởi tạo environment multi-objective (VD: MultiObjectiveQueueEnv).
    from your_env_module import MultiObjectiveQueueEnv

    env = MultiObjectiveQueueEnv(
        n_flows=3,
        max_steps=50,
        ...
    )

    # 2) Tạo 3 policy: Round-Robin, Weighted-Sum, MORL
    rr_policy = RoundRobinPolicy(n_flows=3)
    ws_policy = WeightedSumPolicy(weights=[1.0, 2.0, 1.0])  # tuỳ
    # morl_policy = YourMORLPolicy(...)

    # 3) Tạo evaluator
    evaluator = PerformanceEvaluator(env, num_episodes=5, max_steps=50)

    # 4) So sánh
    policies_dict = {
        "RoundRobin": rr_policy,
        "WeightedSum": ws_policy,
        # "MORL": morl_policy
    }

    results = evaluator.compare_policies(policies_dict)
    # => In ra avg revenue & drop
