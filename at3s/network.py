from itertools import count
import queue
import gym
from gym import spaces
import numpy as np
import threading
import time
from atomiclong import AtomicLong
import timeit
from queue import Empty
import logging
import yaml
import json

logger = logging.getLogger("5GC")


DEFAULT_GENERATOR_SETTING = {
    "TF1": {
        "num_thread": 2,
        "packet_size": 512,
        "rate": 5,
        "price": 10,
        "qos_latency_ms": 4,
    },
    "TF2": {
        "num_thread": 2,
        "packet_size": 1024,
        "rate": 6,
        "price": 10,
        "qos_latency_ms": 10,
    },
    "TF3": {
        "num_thread": 5,
        "packet_size": 1500,
        "rate": 15,
        "price": 30,
        "qos_latency_ms": 20,
    },
}

DEFAULT_PROCESSOR_SETTING = {
    "NR": {
        "num_thread": 2,
        "limit": 500,
        "rate": 200,
        "revenue_factor": 0.9,
    },
    "WF": {
        "num_thread": 2,
        "limit": 500,
        "rate": 100,
        "revenue_factor": 0.1,
    },
}

class Packet:
    def __init__(self, time, traffic_class) -> None:
        self.start = time
        self.traffic_class = traffic_class        
        pass

    def get_start(self):
        return self.start

    def get_traffic_class(self):
        return self.traffic_class    


class NetworkEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, generator_file=None, processor_file=None, clear_queue_step=False, step_second = 20):
        super(NetworkEnv, self).__init__()

        # Generator and processor setting
        if generator_file is not None:
            with open(generator_file, "r") as f:
                self.generator_setting = yaml.safe_load(f.read())
        else:
            self.generator_setting = DEFAULT_GENERATOR_SETTING

        if processor_file is not None:
            with open(processor_file, "r") as f:
                self.processor_setting = yaml.safe_load(f.read())
        else:
            self.processor_setting = DEFAULT_PROCESSOR_SETTING

        logger.info(
            "Create environment. \nGenerator setting \n%s. \nProcessor setting \n%s",
            json.dumps(self.generator_setting, indent=4),
            json.dumps(self.processor_setting, indent=4),
        )
        
        # Parameters
        self.queue_max_utilization = 0.1
        self.scale_factor = 1
        self.reward_factor = {"qos": 1, "revenue": 1, "queue": 0.3}
        self.clear_queue_at_step = clear_queue_step
        self.timeout_processor = 0.5
        self.total_simulation_time = step_second  # seconds
        self.stat_interval = 2
        self.sigmoid_state = False
        self.is_drop = False        

        # Setup
        self.traffic_classes = list(self.generator_setting.keys())
        self.choices = list(self.processor_setting.keys())                
        self.action_space = spaces.Box(low=0, high=len(self.traffic_classes), shape=(1,), dtype=np.float32)                        
        self.init_queue()        
        # [qos_level[class_i], throughput_level[class_i], traffic_allocation_level[class_i][traffic_j]]        
        self.state_shape = (len(self.generator_setting)* 2 + len(self.processor_setting)*len(self.generator_setting),)
        # min(qos_level, throughput_level, traffic_throughput) = 0
        # max(throughput_level) = 1        
        # max(traffic_allocation_level) = 1        
        # max(qos_level) = max(latency) / min(qos_latency) = max(packet_size)*8 / (min(qos_latency_ms)*min(bandwidth)*1e3)
        max_packet_size_key = max(self.generator_setting, key=lambda k: self.generator_setting[k]['packet_size'])
        max_packet_size = self.generator_setting[max_packet_size_key]["packet_size"]        
        min_bandwidth_key = min(self.processor_setting, key=lambda k: self.processor_setting[k]['rate'])
        min_bandwidth = self.processor_setting[min_bandwidth_key]["rate"]        
        min_qos_latency_key = min(self.generator_setting, key=lambda k: self.generator_setting[k]['qos_latency_ms'])            
        min_qos_latency = self.generator_setting[min_qos_latency_key]["qos_latency_ms"]        
        max_qos_level = max_packet_size * 8 / (min_qos_latency * min_bandwidth * 1e3)        
        max_state = 1
        if max_qos_level > 1:
           max_state = max_qos_level

        self.observation_space = spaces.Box(low=0, high=max_state, dtype=np.float32, shape=self.state_shape)                
        self.start_interval = time.time()
        sorted_traffic_class = []
        for tf, value in self.generator_setting.items():
            sorted_traffic_class.append({
                "key": tf,
                "value": value["price"]
            })
        self.sorted_traffic_class = sorted(sorted_traffic_class, key=lambda x: x['value'])
        self.max_prce = self.sorted_traffic_class[len(self.sorted_traffic_class)-1]["value"]

        #Start
        self.stop = False
        self.pause = False
        # self.pause_generate = False
        self.init_state_snap()
        self.init_accum()
        self.init_queue() 
        self.set_action(0.0)       
        self.init_thread()

    def init_thread(self):
        self.list_generator_threads = []
        self.list_processor_threads = []
        for tech, value in self.processor_setting.items():
            for i in range(value["num_thread"]):
                processor = threading.Thread(
                    target=self.packet_processor,
                    args=(tech,),
                )
                self.list_processor_threads.append(processor)          
        
        for tc, value in self.generator_setting.items():
            for i in range(value["num_thread"]):
                packet_generator_thread = threading.Thread(
                    target=self.packet_generator,
                    args=(tc,),
                )
                self.list_generator_threads.append(packet_generator_thread)

        for t in self.list_processor_threads:
            t.start()

        for t in self.list_generator_threads:
            t.start()

        self.log_thread = threading.Thread(target=self.perf_monitor)
        self.log_thread.start()

    def init_queue(self):
        self.queue = {}
        for key, value in self.processor_setting.items():
            if self.is_drop:
                self.queue[key] = queue.Queue(value["limit"] * value["num_thread"])
            else:
                self.queue[key] = queue.Queue()

    def packet_generator(self, traffic_class):        
        setting = self.generator_setting[traffic_class]
        packet_size_bytes = setting["packet_size"]
        target_throughput_mbps = setting["rate"]
        time_to_wait = packet_size_bytes * 8 / (target_throughput_mbps * 1e6)
        start_time = time.time()        
        while True:
            if self.stop:
                logger.info("Generator finish %s", traffic_class)
                # self.generators_finish += 1
                break
            if self.pause:                
                time.sleep(0.1)
                continue
            self.accumulators[traffic_class]["total"] += 1
            weight = self.get_weights(traffic_class)            
            choice = np.random.choice(a=self.choices, p=weight)
            queue = self.queue[choice]          
            # print("choice", choice)  
            if queue.full():
                self.accumulators[traffic_class]["drop"] += 1
                self.stat[choice][traffic_class]["loss"] += 1
            else:
                packet = Packet(time.time_ns(), traffic_class)
                queue.put(packet)
                self.accumulators[traffic_class][choice] += 1
            timeit.time.sleep(time_to_wait)

    def get_weights(self, traffic_class):                
        proprotion_wifi = self.split_dict[traffic_class]
        weights = [1 - proprotion_wifi, proprotion_wifi]        
        return weights

    def spinwait_nano(delay):
        target = perf_counter_ns() + delay * 1000
        while perf_counter_ns() < target:
            pass    

    def packet_processor(self, tech):
        start = time.time()
        rate = self.processor_setting[tech]["rate"]
        logger.info("Processor %s %s mbps", tech, rate)
        while True:
            if self.stop:
                logger.info("Processor finish %s", tech)
                break
            if self.pause:
                time.sleep(0.1)
                continue
            try:
                item = self.queue[tech].get(timeout=self.timeout_processor)
                if item is None:
                    time.sleep(0.0001)
                    continue
                else:
                    traffic_class = item.get_traffic_class()
                    packet_size = self.generator_setting[traffic_class]["packet_size"]
                    process_time = packet_size * 1.0 * 8 / (rate * 1e6)
                    timeit.time.sleep(process_time)
                    latency = time.time_ns() - item.get_start()
                    if latency <= 0:
                        logger.error("Negative time %s", latency)
                    else:
                        self.accumulators[traffic_class]["latency"].append(latency)
                        self.accumulators[traffic_class]["process"] += 1
                        self.stat[tech][traffic_class]["revenue"] += 1
                        self.stat[tech][traffic_class]["packet_count"] += 1
                        self.stat[tech][traffic_class]["latency"] += latency
                    # queue.task_done()
            except Exception as error:
                if type(error) is Empty:
                    if self.stop:
                        logger.info("Processor finish %s", tech)
                        # self.processors_finish += 1
                        break
                    continue
                logger.error(error)

    def perf_monitor(self):
        while True:
            if self.stop:
                logger.info("Finish monitor")
                return

            if self.pause or time.time() - self.start_interval < self.stat_interval:
                time.sleep(0.1)
                continue

            logger.info("Action %s. Offload setting %s", self.action, self.split_dict)
            longest = 0
            for tc, value in self.accumulators.items():
                log_str = tc
                for k, v in value.items():
                    if k == "latency":
                        latency = np.mean(v) / 1e6
                        log_str += ". latency: " + str(round(latency, 2)) + " ms"
                        self.state_snapshot[tc]["latency"].append(latency)
                        value[k] = []
                    else:                                             
                        total_processed = self.scale_factor * (
                            v.value
                            * self.generator_setting[tc]["packet_size"]
                            * 8
                            / 1e6
                        )
                        throughput = total_processed / self.stat_interval
                        self.state_snapshot[tc]["rate"].append(throughput)
                        log_str += ". " + k + ": " + str(round(throughput, 2)) + " mbps"
                        v.value = 0
                logger.info(log_str)
                longest = max(longest, len(log_str))
            
            rev_loss = 0
            rev_gain = 0
            for tech, value in self.stat.items():
                log_str = tech
                total_revenue = 0
                total_loss = 0
                total_data = 0
                rev_factor = self.processor_setting[tech]["revenue_factor"]                                               
                for tc, v in value.items():
                    price = self.generator_setting[tc]["price"]
                    tf_amount = (
                        self.generator_setting[tc]["packet_size"]
                        * v["revenue"].value
                        / 8
                        / 1e6
                    )
                    tf_rev = tf_amount * price
                    if price == 0:
                       rev_loss += tf_amount * self.max_price

                    tf_loss = (
                        self.generator_setting[tc]["price"]
                        * self.generator_setting[tc]["packet_size"]
                        * v["loss"].value
                        / 8
                        / 1e6
                    )
                    total_revenue += self.scale_factor * tf_rev
                    total_loss += self.scale_factor * tf_loss
                    total_data += self.scale_factor * (
                        v["packet_count"].value
                        * self.generator_setting[tc]["packet_size"]
                        * 8
                    )
                    throughput = self.scale_factor * (
                        v["packet_count"].value
                        * self.generator_setting[tc]["packet_size"]
                        * 8
                        / (self.stat_interval * 1e6)
                    )                    
                    latency = 0
                    if v["packet_count"].value > 0:
                        latency = v["latency"].value / (v["packet_count"].value * 1e6)                    
                        
                    self.state_snapshot[tc]["throughput"][tech].append(throughput)                                        
                    log_str += (
                        "| "
                        + tc
                        + ". R: "
                        + str(round(tf_rev, 2))
                        + "$. L: "
                        + str(round(tf_loss, 2))
                        + "$. T: "
                        + str(round(throughput, 2))
                        + "mbps. D: "
                        + str(round(latency, 2))
                        + " ms"
                    )
                    v["packet_count"].value = 0
                    v["latency"].value = 0
                log_str += (
                    "|All. R: "
                    + str(round(total_revenue * rev_factor, 2))
                    + "$. L: "
                    + str(round(total_loss * rev_factor, 2))
                    + "$. T: "
                    + str(round(total_data / (self.stat_interval * 1e6), 2))
                    + " mbps"
                )
                rev_gain += total_revenue * rev_factor
                logger.info(log_str)
                longest = max(longest, len(log_str))

            self.state_snapshot["rev_loss"] = rev_gain
            self.state_snapshot["rev_gain"] = rev_loss

            separator = "=" * longest            
            logger.info("Queue. %s", self.get_queue_status())
            logger.info(separator)
            self.start_interval = time.time()

    def get_queue_status(self):
        result = ""
        for tech, value in self.queue.items():
            max_queue_size = (
                self.processor_setting[tech]["limit"]
                * self.processor_setting[tech]["num_thread"]
            )
            percent = value.qsize() / max_queue_size
            result += tech + ": " + str(round(percent, 2)) + ". "
            for tc, val in self.generator_setting.items():
                self.state_snapshot[tc]["queue"][tech].append(percent)
        return result

    def init_state_snap(self):
        self.state_snapshot = {
            "rev_gain": 0,
            "rev_loss": 0
        }
        for tc, setting in self.generator_setting.items():
            self.state_snapshot[tc] = {"latency": [], "throughput": {}, "queue": {}, "rate": []}            
            for tech, v in self.processor_setting.items():
                self.state_snapshot[tc]["throughput"][tech] = []
                self.state_snapshot[tc]["queue"][tech] = []

    def init_accum(self):        
        self.stat = {}
        self.accumulators = {}

        for key, value in self.processor_setting.items():
            self.stat[key] = {}
            for tc in self.generator_setting.keys():
                self.stat[key][tc] = {
                    "revenue": AtomicLong(0),
                    "packet_count": AtomicLong(0),
                    "loss": AtomicLong(0),
                    "latency": AtomicLong(0),                    
                }

        for key, value in self.generator_setting.items():
            self.accumulators[key] = {}
            self.accumulators[key]["total"] = AtomicLong(0)
            self.accumulators[key]["drop"] = AtomicLong(0)
            self.accumulators[key]["process"] = AtomicLong(0)
            self.accumulators[key]["latency"] = []
            for val in self.choices:
                self.accumulators[key][val] = AtomicLong(0)

    def set_action(self, action):
        # remap action   
        # 1.5 => [1.0, 0.5, 0]     
        self.action = action
        quotion, remainder = divmod(action, 1)
        # print(action, quotion, remainder)
        idx = 0
        for i in range(int(quotion)):
            self.sorted_traffic_class[i]["offload"] = 1.0
            idx += 1        
                
        if remainder > 0 and idx < len(self.sorted_traffic_class):
           self.sorted_traffic_class[idx]["offload"] = remainder        
        # print("sorted_traffic_class", self.sorted_traffic_class)
        self.split_dict = {}
        for item in self.sorted_traffic_class:
            name = item["key"]            
            percent = 0.0
            if "offload" in item: 
                percent = item["offload"]
            if name not in self.split_dict:
                self.split_dict[name] = percent
        
        # print("split_dict",  self.split_dict)

    def step(self, action):
        if self.clear_queue_at_step:
            self.clear_queue()
        
        self.init_state_snap()                
        self.set_action(action)                        
        start_step = time.time()
        time.sleep(self.total_simulation_time)
        self.pause_generate = False
        self.start_interval = time.time()        
        logger.info(
            "Finish step. Queue %s. Total time: %s s",
            self.get_queue_status(),
            str(round(time.time() - start_step, 2)),
        )
        logger.info("state_snapshot %s", self.state_snapshot)
        # state, reward, done, terminated = self.get_current_state_and_reward()
        # return state, reward, done, terminated, {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_current_state_and_reward(self):
        state_arr = []
        reward_qos = []
        reward_revenue = 0
        mean_queue = 0
        count_queue = 0            
        # return final_state, self.last_retained_revenue, done, terminal

    def get_last_step_latency(self):
        return self.last_latency

    def get_last_step_revenue(self):
        return self.last_revenue

    def get_last_step_throughput(self):
        return self.last_throughtput

    def get_last_step_queue_load(self):
        return self.last_queue_load

    def reset(self):
        logger.info("Reset env")
        self.pause = True
        self.clear_queue()
        return np.zeros(self.state_shape), {}

    def clear_queue(self):
        for tech, q in self.queue.items():
            while not q.empty():
                q.get()
            logger.info("Queue %s cleared", tech)

    def render(self, mode="human"):
        print("Render not implemented")
        pass

    def get_action_shape(self):
        return self.action_space.sample().shape

    def get_state_shape(self):
        return self.observation_space.sample().shape

    def close(self):
        logger.info("Close env. Stop all thread")
        self.stop = True

    def get_traffic_classes(self):
        return self.traffic_classes  