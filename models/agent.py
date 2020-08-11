import shutil
import math
import numpy as np
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import torch

from utils.utils import OUNoise, make_gif
from utils.logger import Logger
from env.utils import create_env_wrapper


class Agent(object):

    def __init__(self, config, policy, global_episode, n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.use_global_episode = True
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.env_wrapper.env.set_agent(self.n_agent)
        print ("set agent {}".format(n_agent))
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])
        self.ou_noise.reset()

        self.actor = policy

        # Logger
        log_path = f"{log_dir}/agent-{n_agent}"
        self.logger = None
        if not config["test"]:
            run_name = config["run_name"]
            self.logger = Logger(log_path, name = f"{run_name}/agent-{n_agent}", project_name=config["project_name"])


    def update_actor_learner(self, learner_w_queue):
        """Update local actor to the actor from learner. """
        if learner_w_queue.empty():
            return
        source = learner_w_queue.get()
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        all_test_done = False
        if self.config["test"]:
            self.env_wrapper.env.use_test_setting()
        log_path = f"{self.log_dir}/agent-{self.n_agent}"
        while training_on.value:
            if all_test_done:
                break
            elif self.config["test"]:
                # if self.logger is not None:
                #     self.logger.close()
                self.logger = Logger(log_path, name = f"{self.config['run_name']}_path_{self.env_wrapper.env.get_test_path_number()}", project_name="evaluate")
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()
            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            print("call reset on agent {}".format(self.n_agent))
            state = self.env_wrapper.reset()
            #self.env_wrapper.env.resume_simulator()
            print (state.shape)
            print("called reset on agent {}".format(self.n_agent))
            self.ou_noise.reset()
            done = False
            heading_avg = []
            reward_avg = []
            distance_avg = []
            skip_run = False
            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "supervisor":
                    action = self.env_wrapper.env.get_supervised_action()
                elif self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()
                next_state, reward, done = self.env_wrapper.step(action)
                if hasattr(self.env_wrapper.env, 'get_angle_person_robot'):
                    heading_avg.append(np.rad2deg(self.env_wrapper.env.get_angle_person_robot()))
                distance_avg.append(math.hypot(state[0]*6, state[1]*6))
                reward_avg.append(reward)
                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                if not self.config["test"]:
                    self.exp_buffer.append((state, action, reward))
                else:
                    self.logger.scalar_summary("reward", reward_avg[-1], len(reward_avg))
                    self.logger.scalar_summary("distance", distance_avg[-1], len(reward_avg))
                    self.logger.scalar_summary("heading", heading_avg[-1], len(reward_avg))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']
                    if not replay_queue.full():
                        replay_queue.put([state_0, action_0, discounted_reward, next_state, done, gamma])

                state = next_state

                if done or (not self.config["test"] and num_steps == self.max_steps):
                    if hasattr(self.env_wrapper.env, 'is_skip_run') and self.env_wrapper.env.is_skip_run():
                        print("skiping this run as it is not useful")
                        skip_run = True
                        break
                    print ("agent {} done steps: {}/{} episode reward: {}".format(self.n_agent, num_steps, self.max_steps, episode_reward))
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        #print("agent {} exp_buffer_len {}".format(self.n_agent, len(self.exp_buffer)))
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            #print("agent {} exp_buffer_len {}".format(self.n_agent, len(self.exp_buffer)))
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        replay_queue.put([state_0, action_0, discounted_reward, next_state, done, gamma])
                    break

                num_steps += 1

            #print("agent {} finished if".format(self.n_agent))
            # Log metrics
            if skip_run:
                continue
            step = update_step.value
            global_step = self.global_episode.value
            self.env_wrapper.env.set_mode_person_based_on_episode_number(global_step)
            observation_image = self.env_wrapper.env.get_current_observation_image()
            if self.use_global_episode:
                step= global_step
            if self.agent_type == "exploitation" or self.agent_type == "supervisor":
                pre_log = ""
                if self.config["test"]:
                    #self.logger.close()
                    self.logger = Logger(log_path, name = f"{self.config['run_name']}_p_{self.env_wrapper.env.get_test_path_number()}", project_name="evaluate")
                    step = 1
                    pre_log = f"path_{self.env_wrapper.env.get_test_path_number()}"
                if len(heading_avg)>0:
                    self.logger.scalar_summary(f"{pre_log}agent/heading_avg", np.mean(heading_avg), step)
                self.logger.scalar_summary(f"{pre_log}agent/reward_avg", np.mean(reward_avg), step)
                self.logger.scalar_summary(f"{pre_log}agent/distance_avg", np.mean(distance_avg), step)
                observation_image_type = f"{pre_log}agent/observation_error"
                if (hasattr(self.env_wrapper.env, 'is_successful') and self.env_wrapper.env.is_successful()) or \
                    (not hasattr(self.env_wrapper.env, 'is_successful') and num_steps == self.max_steps):
                    observation_image_type = f"{pre_log}agent/observation_end"
                self.logger.image_summar(observation_image_type, observation_image, step)
                print("-------------------->{} {} test: {} global: {} ,step: {}".format(np.mean(reward_avg), np.mean(heading_avg), self.config["test"], global_step, step))
            else:
                if num_steps == self.max_steps:
                    self.logger.image_summar("agent_{}/observation_end".format(self.n_agent), observation_image, step)
                else:
                    self.logger.image_summar("agent_{}/observation_error".format(self.n_agent), observation_image, step)

            self.logger.scalar_summary("agent/reward", episode_reward, step)
            self.logger.scalar_summary("agent/episode_step", num_steps, step)
            self.logger.scalar_summary("agent/episode_timing", time.time() - ep_start_time, step)

            if self.config["test"]:
                if not self.env_wrapper.env.is_finish():
                    self.env_wrapper.env.next_setting()
                else:
                    all_test_done = True

            # Saving agent
            if not self.config["test"] and (self.local_episode % self.num_episode_save == 0 or episode_reward > best_reward):
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")
                print("reward is: {} step: {} ".format(episode_reward, step))

            rewards.append(episode_reward)
            if (self.agent_type == "exploration" or self.agent_type == "supervisor") and self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_actor_learner(learner_w_queue)

        # while not replay_queue.empty():
        #     replay_queue.get()

        # Save replay from the first agent only
        # if self.n_agent == 0:
        #    self.save_replay_gif()

        #print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        last_path = f"{self.log_dir}"
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        if not os.path.exists(last_path):
            os.makedirs(last_path)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)
        model_fn = f"{last_path}/best.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self):
        dir_name = "replay_render"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        #self.env_wrapper.env.resume_simulator()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")
