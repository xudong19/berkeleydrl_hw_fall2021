import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        n_batch = observation.shape[0]
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        qa_values = self.critic.qa_values(observation)
        # assume qa_values.shape: n_batch * n_action
        assert len(qa_values.shape) == 2, f"{qa_values.shape}"
        assert qa_values.shape[0] == n_batch, f"{qa_values.shape}"
        action = np.argmax(qa_values, axis=1)
        return action.squeeze()