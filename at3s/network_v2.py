import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiObjectiveQueueEnv(gym.Env):
    """
    Môi trường multi-objective:
      - Ta có 2 "tiêu chí" (objectives):
         1) revenue (càng cao càng tốt)
         2) data_drop (càng thấp càng tốt)

    Thay vì trả về 1 scalar reward, ta trả về 1 vector: [revenue, data_drop].
    """

    metadata = {"render_modes":["human"]}

    def __init__(self,
                 n_flows=3,
                 fiveqi_list=None,
                 max_latency_map=None, # dict 5QI->deadline
                 rate_revenue=None,    # list: revenue per bit
                 max_buffer=5e6,
                 max_arrival=2e6,
                 capacity_nr=1e6,
                 max_steps=10,
                 render_mode=None):
        super().__init__()

        self.n_flows = n_flows
        self.max_buffer = max_buffer
        self.max_arrival = max_arrival
        self.capacity_nr = capacity_nr
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 5QI -> max_latency
        if max_latency_map is None:
            # default: 5QI=x => x steps
            self.max_latency_map = {i:i for i in range(1,10)}
        else:
            self.max_latency_map = max_latency_map

        if fiveqi_list is None:
            self.fiveqi_list = np.random.randint(1,10,size=n_flows)
        else:
            self.fiveqi_list = np.array(fiveqi_list,dtype=int)
        
        self.flow_latency = np.array([
            self.max_latency_map.get(qi,9) for qi in self.fiveqi_list
        ], dtype=int)

        if rate_revenue is None:
            # default: 1e-9 ~ 1 USD / Gb
            self.rate_revenue = np.array([1e-9]*n_flows, dtype=float)
        else:
            self.rate_revenue = np.array(rate_revenue,dtype=float)

        # Quan sát = [arrivals..., queue..., time_in_queue...]
        obs_low = np.zeros(3*n_flows, dtype=np.float32)
        obs_high = []
        # arrivals <= max_arrival
        obs_high.extend([max_arrival]*n_flows)
        # queue <= max_buffer
        obs_high.extend([max_buffer]*n_flows)
        # time_in_queue <= max_steps*10 (đoán)
        obs_high.extend([max_steps*10]*n_flows)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=np.array(obs_high,dtype=np.float32),
            shape=(3*n_flows,),
            dtype=np.float32
        )

        # Action = n_flows giá trị [0..1], sum=1 => phân chia capacity NR
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_flows,),
            dtype=np.float32
        )

        # Ta có 2 mục tiêu: [revenue, data_drop].
        # Thường data_drop >=0, revenue>=0, ta cứ để high=inf
        self.reward_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.arrivals = None
        self.queue = None
        self.time_in_queue = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.arrivals = np.random.uniform(0,self.max_arrival,size=self.n_flows)
        self.queue = np.zeros(self.n_flows,dtype=np.float32)
        self.time_in_queue = np.zeros(self.n_flows,dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        # 1) Thêm arrivals
        for i in range(self.n_flows):
            self.queue[i]+= self.arrivals[i]

        data_drop=0.0
        revenue=0.0

        # 2) buffer overflow => drop flow 5QI cao
        sum_q = np.sum(self.queue)
        if sum_q>self.max_buffer:
            overflow = sum_q - self.max_buffer
            dropped = self._drop_high_5qi(overflow)
            data_drop += dropped

        # 3) time_in_queue++
        for i in range(self.n_flows):
            if self.queue[i]>0:
                self.time_in_queue[i]+=1

        # 4) deadline => drop
        for i in range(self.n_flows):
            if self.time_in_queue[i]>self.flow_latency[i]:
                data_drop+= self.queue[i]
                self.queue[i]=0
                self.time_in_queue[i]=0

        # 5) action => chia capacity NR
        sum_a = np.sum(action)
        if sum_a<1e-9:
            alloc = np.ones(self.n_flows)/self.n_flows
        else:
            alloc = action/sum_a

        for i in range(self.n_flows):
            cap_i = alloc[i]*self.capacity_nr
            send_i = min(self.queue[i], cap_i)
            # update queue
            self.queue[i]-= send_i
            if self.queue[i]<=0:
                self.time_in_queue[i]=0
            # tính revenue
            revenue+= send_i*self.rate_revenue[i]

        # 6) reward = vector: [revenue, data_drop]
        # data_drop => "càng nhỏ càng tốt"
        # => tuỳ multi-objective algo mà handle
        reward_vec = np.array([revenue, data_drop], dtype=np.float32)

        # 7) random arrivals next step
        variation = np.random.uniform(-0.2,0.2,size=self.n_flows)
        self.arrivals = np.clip(self.arrivals*(1+variation),0,self.max_arrival)

        self.step_count+=1
        truncated = (self.step_count>=self.max_steps)
        terminated = False

        obs = self._get_obs()
        info = {
            "revenue": revenue,
            "data_drop": data_drop,
            "queue": self.queue.copy(),
            "time_in_queue": self.time_in_queue.copy()
        }
        return obs, reward_vec, terminated, truncated, info

    def _drop_high_5qi(self, overflow):
        remain=overflow
        dropped=0.0
        order=np.argsort(-self.fiveqi_list) # 5QI giảm dần => drop flow priority thấp
        for idx in order:
            if remain<=1e-9:
                break
            if self.queue[idx]>=remain:
                self.queue[idx]-= remain
                dropped+= remain
                self.time_in_queue[idx]=0
                remain=0
            else:
                dropped+= self.queue[idx]
                remain-= self.queue[idx]
                self.queue[idx]=0
                self.time_in_queue[idx]=0
        return dropped

    def render(self):
        if self.render_mode=="human":
            print(f"Step={self.step_count}, queue={self.queue}, TQ={self.time_in_queue}, arrivals={self.arrivals}")

    def _get_obs(self):
        return np.concatenate([
            self.arrivals,
            self.queue,
            self.time_in_queue
        ]).astype(np.float32)


if __name__=="__main__":
    env=MultiObjectiveQueueEnv(
        n_flows=3,
        fiveqi_list=[1,5,9],
        max_latency_map={1:2,5:5,9:9},
        rate_revenue=[1e-9,2e-9,5e-9],
        max_buffer=5e6,
        max_arrival=2e6,
        capacity_nr=1e6,
        max_steps=8,
        render_mode="human"
    )

    obs,_=env.reset()
    done=False
    truncated=False
    total_vec = np.array([0.0,0.0])
    while not(done or truncated):
        action=env.action_space.sample()
        obs,reward,done,truncated,info=env.step(action)
        env.render()
        print(f"Action={action}, RewardVec={reward}, Info={info}")
        total_vec+=reward
    print("Episode end. total multi-obj reward=", total_vec)
    env.close()
