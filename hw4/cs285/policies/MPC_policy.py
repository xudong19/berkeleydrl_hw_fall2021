import numpy as np
import random
from scipy.stats import truncnorm
from .base_policy import BasePolicy



def update_value(old_v, new_v, alpha):
    return alpha * new_v + (1 - alpha) * old_v


# def get_truncated_normal(mean, var, low, upp):
#     max_try_times = 10
#     for _ in range(max_try_times):
#         val = random.gauss(mean, np.sqrt(var))
#         if low <= val <= upp:
#             return val
#     return (low + upp) / 2
    
    # X = truncnorm(
    #     (low - mean) / np.sqrt(var), (upp - mean) / sd, loc=mean, scale=sd)
    # return X.rvs()


def get_truncated_normal_array(mean, var, low, upp):
    max_try_times = 10
    std = np.sqrt(var)
    shape = mean.shape
    valid_mask = np.full(shape, False)
    remaining_mask = np.full(shape, True)
    result = (low + upp) / 2
    for _ in range(max_try_times):
        new_result = np.random.normal(loc=mean, 
                                  scale=std)
        valid_mask_this_sample = np.logical_and(
                                       low <= new_result, new_result >= upp)
        to_be_added_sample_mask = np.logical_and(valid_mask_this_sample, 
                                                 remaining_mask)
        result[to_be_added_sample_mask] = new_result[to_be_added_sample_mask]
        valid_mask = np.logical_or(valid_mask, valid_mask_this_sample)
        remaining_mask = np.logical_not(valid_mask)
        if np.all(valid_mask):
            return result
    return result


def get_truncated_normal_array2(mean, var, low, upp):
    # Though slightly faster, but incorrect sampling.
    std = np.sqrt(var)
    result = np.random.normal(loc=mean, 
                                  scale=std)
    return np.minimum(np.maximum(result, low), upp)
        


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")
    
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            # candidate_action_sequences = np.zeros((num_sequences, 
            #                                        horizon, self.ac_dim))
            candidate_action_sequences = \
                np.random.uniform(low=np.tile(self.low[None, None, ...],
                                              (num_sequences, horizon, 1)),
                                  high=np.tile(self.high[None, None, ...],
                                               (num_sequences, horizon, 1)))
            assert candidate_action_sequences.shape == (num_sequences,
                                                        horizon, self.ac_dim), \
                                                            f"{candidate_action_sequences.shape}"
            # for i_seq in range(num_sequences):
            #     for i_step in range(horizon):
            #         action = np.random.uniform(low=self.low, high=self.high)
            #         candidate_action_sequences[i_seq, i_step, :] = action
            return candidate_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            
            # step 1, init with random distribution
            # mu = np.zeros((horizon, self.ac_dim))
            # var = 10* np.ones((horizon, self.ac_dim))    
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                if i == 0:
                    candidate_action_sequences = self.sample_action_sequences(
                        num_sequences, horizon, obs=None)
                else:
                    candidate_action_sequences = np.zeros((num_sequences, 
                                                   horizon, self.ac_dim))
                    # for i_seq in range(num_sequences):
                    #     for i_step in range(horizon):
                    #         for i_ac_dim in range(self.ac_dim):
                    #             ac_element = get_truncated_normal(
                    #                 mu[i_step, i_ac_dim], 
                    #                 var[i_step, i_ac_dim], 
                    #                 low=self.low[i_ac_dim], 
                    #                 upp=self.high[i_ac_dim])
                    #             candidate_action_sequences[
                    #                 i_seq, i_step, i_ac_dim] = ac_element
                    mu_broad = np.tile(mu[None, ...], (num_sequences, 1, 1))
                    var_broad = np.tile(var[ None, ...], ( num_sequences, 1, 1))
                    low_broad = np.tile(self.low[ None, None, ...], 
                                        (num_sequences, horizon, 1))
                    upp_broad = np.tile(self.high[None, None, ...], 
                                        ( num_sequences, horizon, 1))
                    candidate_action_sequences = get_truncated_normal_array(
                        mu_broad, var_broad, low_broad, upp_broad
                    )
                
                rewards_of_sequences = self.evaluate_candidate_sequences(
                    candidate_action_sequences, obs)
                elite_indes = np.argsort(rewards_of_sequences)[-self.cem_num_elites:]
                new_mu = candidate_action_sequences[elite_indes, ...].mean(
                    axis=0, keepdims=False)
                new_var = candidate_action_sequences[elite_indes, ...].var(
                    axis=0, keepdims=False)
                if i == 0:
                    mu = new_mu
                    var = new_var
                else:
                    mu = update_value(mu, new_mu, self.cem_alpha)
                    var = update_value(var, new_var, self.cem_alpha)

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            cem_action = mu

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        num_sequences, horizon, _ = candidate_action_sequences.shape
        n_models = len(self.dyn_models)
        rewards_across_models = np.zeros((n_models, num_sequences))
        
        for i_model, model in enumerate(self.dyn_models):
            rewards_across_models[i_model, :] = \
            self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
        rewards_mean_across_models = rewards_across_models.mean(axis=0, 
                                                                keepdims=False)
        return rewards_mean_across_models

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(
                candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = np.argmax(predicted_rewards)
            action_to_take = candidate_action_sequences[best_action_sequence, 0, :]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        num_sequences, horizon, _ = candidate_action_sequences.shape
        sum_of_rewards = np.zeros(num_sequences)
        # for i_seq in range(num_sequences):
        #         curr_obs = obs
        #         for i_step in range(horizon):
        #             acs = candidate_action_sequences[i_seq, i_step, :]
        #             reward = self.env.get_reward(curr_obs, acs)
        #             sum_of_rewards[i_seq] += reward
        #             curr_obs = model.get_prediction(curr_obs, acs, 
        #                                             self.data_statistics)
        curr_obs = np.tile(obs[None, :], (num_sequences, 1))
        episode_end = np.zeros(num_sequences)
        for i_step in range(horizon):
            acs = candidate_action_sequences[:, i_step, :]
            reward, done = self.env.get_reward(curr_obs, acs)
            assert reward.shape == (num_sequences, ), \
            f"{reward.shape}, {num_sequences}"
            assert done.shape == (num_sequences, ), \
            f"{done.shape}, {num_sequences}"
            # This seems to be the correct one.
            # sum_of_rewards += reward * (1 - episode_end)
            sum_of_rewards += reward
            episode_end = np.logical_or(episode_end, done)
            curr_obs = model.get_prediction(curr_obs, acs, 
                                            self.data_statistics)
        
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
