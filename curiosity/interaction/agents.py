'''
Agents have access to the graph and make action choices each turn.
'''
import numpy as np
import random
import copy


class Agent(object):
    '''
    Abstract agent class

    '''
    def __init__(self,
            compute_settings,
            stored_params,
            **kwargs):
        raise NotImplementedError()


    def act(self,
            sess,
            history):
        raise NotImplementedError()


class TelekineticMagicianAgent(Agent):
    '''
    The baby.
    '''
    def __init__(self,
            compute_settings,
            stored_params,
            model_params,
            **kwargs):
        for k_ in ['model_params']:
            setattr(self, k_, eval(k_))
        self.network = model_params['func'](self.model_params)
        self.policy = self.network['policy']
        shape_needed = [1] + self.network['inputs']['action_sample'].get_shape().as_list()[1:]
        self.dummy_action = np.zeros(shape_needed, dtype = np.float32)
        self.num_needed = {k_ : self.network['holders'][k_].get_shape().as_list()[1] for k_ in ['images1', 'action', 'action_post']}

    def act(self, sess, history):
        _holders = self.network['holders']
        feed_dict = {_holders['train_indicator'] : 0., _holders['action_sample'] : self.dummy_action}
        if history is not None:
            recent_history = {k_ : history[k_][-self.num_needed[k_]:] for k_ in ['images1', 'action', 'action_post']}
            sufficient_inputs = not any([any([_input is None for _input in _v]) for _v in recent_history.values()])
            if sufficient_inputs:
                recent_history = {k_ : np.array([v_]) for k_, v_ in recent_history.items()}
                feed_dict.update({_holders[k_] : recent_history[k_] for k_ in recent_history})
                #for f_ in feed_dict.values():
                #    print(type(f_))
                policy_output = sess.run(self.network['policy'], feed_dict = feed_dict)
                action_chosen = policy_output.pop('action_chosen')
                return action_chosen, policy_output
        action_chosen = sess.run(self.network['policy']['random_alternative'], feed_dict = feed_dict)
        return action_chosen, {'reward' : 0.}

    def get_train_targets(self):
        return self.network['train_targets']

    def get_signals_and_variables(self):
        return self.network['signals_and_variables']

    def get_update_holders(self):
        return self.network['holders']

    def get_all_holders(self):
        return self.network['holders']

class RandomAgent(Agent):
    def __init__(self,
            compute_settings,
            stored_params,
            seed,
            action_dim,
            lower_bound,
            upper_bound,
            **kwargs):
        self.rng = np.random.RandsomState(seed = self.seed)

    def act(self, sess, history):
        return self.rng.uniform(self.lower_bound, self.upper_bound, \
                self.action_dim), {}
