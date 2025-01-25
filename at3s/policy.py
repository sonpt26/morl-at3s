import numpy as np

class RoundRobinPolicy:
    def __init__(self, n_flows):
        self.n_flows = n_flows
        self.current_index = 0

    def get_action(self, obs=None):
        # One-hot action
        action = np.zeros(self.n_flows, dtype=np.float32)
        action[self.current_index] = 1.0
        self.current_index = (self.current_index + 1) % self.n_flows
        return action

class WeightedSumPolicy:
    """
    Ví dụ: Mỗi flow i có weight_i,
    action[i] = weight_i / sum(weight).
    """
    def __init__(self, weights):
        self.weights = np.array(weights, dtype=np.float32)
        self.n_flows = len(weights)

    def get_action(self, obs=None):
        sum_w = np.sum(self.weights)
        if sum_w < 1e-9:
            # Chia đều nếu tất cả =0
            return np.ones(self.n_flows, dtype=np.float32)/self.n_flows
        else:
            return self.weights / sum_w

# Giả định MORLPolicy do bạn xây dựng, với hàm get_action(obs).
# class MORLPolicy:
#     def get_action(self, obs):
#         ...
