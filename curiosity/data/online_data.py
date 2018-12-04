'''
A data provider for online training
'''

import copy, threading
import environment
import six.moves.queue as queue
import numpy as np

class History:
    '''
        Maintains a history of data.
        Pads with Nones when requested.
    '''
    def __init__(self, 
            memory_len,
            none_pad_len,
        ):
        for k_ in ['memory_len', 'none_pad_len']:
            setattr(self, k_, eval(k_))
        self._history = None
        self._add_calls = 0

    def add(self, dat):
        if self._history is None:
            self._history = {k_ : [None for _ in range(self.memory_len)] for k_ in dat}
        self._add_calls += 1
        for k_, mem_ in self._history.items():
            mem_.pop(0)
            mem_.append(dat[k_])
        return self._history, self._add_calls

    def pad_with_nones(self):
        dat = {k_ : None for k_ in self._history}
        for _ in range(self.none_pad_len):
            self.add(dat)
        return self._history



class InteractiveProviderBase(threading.Thread):
    def __init__(self, environment, capacity = 5):
        '''
            A base for interactive data gathering. Runs a loop on a separate thread.
            Children must implement
                -run_env that actually does the data collection loop
                -_batching_fn that takes the yield of run_env and produces a batch
        '''
        threading.Thread.__init__(self)
        self.env = environment
        self.queue = queue.Queue(capacity)
        self.daemon = True
        self.sess = None

    def start_runner(self, sess, agent):
        self.sess = sess
        self.agent = agent
        self.start()

    def run(self):
        print('run started')
        with self.sess.as_default():
            self._run()
            
    def _run(self):
        yielded = self.run_env()
        while True:
            history, add_calls = next(yielded)
            batch = self.batching_fn(history)
            self.queue.put(batch, timeout = 10000.0)

    def dequeue_batch(self):
        return self.queue.get(timeout = 10000.0)

    def run_env(self):
        raise NotImplementedError()

    def _batching_fn(self):
        raise NotImplementedError()


class IntrinsicRewardDataProvider(InteractiveProviderBase):
    '''
    Runs data collection loop, with the agent returning the intrinsic reward signal.
    '''
    def __init__(self, 
                 memory_len,
                 gather_at_beginning,
                 gather_per_batch,
                 batches_per_gather,
                 scene_params,
                 batching_seed,
                 batching_params,
                 scene_length_bounds,
                 *args,
                 **kwargs):
        super(IntrinsicRewardDataProvider, self).__init__(* args, **kwargs)
        self.history = History(memory_len, none_pad_len = 0)
        for k_ in ['gather_at_beginning', 'gather_per_batch', 'batches_per_gather', 'scene_params', 'batching_params', 'scene_length_bounds']:
            setattr(self, k_, eval(k_))
        self.rng = np.random.RandomState(batching_seed)

    def batching_fn(self, history):
        raise NotImplementedError()

    def run_env(self):
        num_this_scene = 0
        total_gathered = 0
        termination_signal = False
        hist = None
        scene_len = -1

        while True:
            num_this_yield = 0

            while num_this_yield < self.gather_per_batch or total_gathered < self.gather_at_beginning\
                             or hist is None:
                #print('data step')
                if termination_signal or num_this_scene >= scene_len:
                    try:
                        obs = self.env.reset(\
                               *self.scene_params)
                    except IOError:
                        continue
                    to_add = {
                        'action' : None,
                        'reward' : None,
                    }
                    to_add.update(obs)
                    scene_len = self.rng.randint(self.scene_length_bounds[0], self.scene_length_bounds[1])
                    num_this_scene = 0
                    termination_signal = False
                    action_next = None
                try:
                    #TODO: replace with real action provided by agent
                    action, other_act_returns = self.agent.act(self.sess, hist)
                    obs, termination_signal = self.env.step(action)
                    to_add = {'action' : action}
                    to_add.update(other_act_returns)
                    to_add.update(obs)

                    hist, num_add_calls = self.history.add(to_add)
                except IOError:
                    termination_signal = True
                    continue
                num_this_yield += 1
                num_this_scene += 1
                total_gathered += 1
            for _ in range(self.batches_per_gather):
                yield hist, num_add_calls

def check_none_are_none(history, idx, grab_indices):
    for k_, (grab_start_, grab_end_) in grab_indices.items():
        dat = history[k_][grab_start_ - idx : ]\
                if grab_end_ == idx\
                else history[k_][grab_start_ - idx: grab_end_ - idx]
        if any([d_ is None for d_ in dat]):
            return False
    #TODO: make more shift-invariant
    if history['action'][-idx] is None:
        return False
    return True
    


class BufferSampleDataProvider(IntrinsicRewardDataProvider):
    def batching_fn(self, history):
        chosen = []
        (min_idx, max_idx) = self.batching_params['valid_index_range']
        while len(chosen) < self.batching_params['batch_size']:
            proposed_idx = self.rng.randint(min_idx, max_idx)
            if proposed_idx not in chosen and check_none_are_none(history, proposed_idx, self.batching_params['grab_indices']):
                chosen.append(proposed_idx)
    
        #grab data
        batch = {}
        for k_, (grab_start_, grab_end_) in self.batching_params['grab_indices'].items():
            dat = [history[k_][grab_start_ - idx : ]\
                    if grab_end_ == idx\
                    else history[k_][grab_start_ - idx: grab_end_ - idx]\
                    for idx in chosen]
            dat = [[self.batching_params['none_replacement'][k_] if step_dat_ is None else step_dat_\
                      for step_dat_ in ex_dat_] for ex_dat_ in dat]
            batch[k_] = np.array(dat)

        #grab some recent data
        batch['recent'] = {k_ : h_[-self.batching_params['recent_history_length'] : ] for k_, h_ in history.items()}
        #grab action sample data
        dat = [history['action'][-idx] for idx in chosen]
        #dat = [act_ if act_ is not None else np.zeros(batch['action'][0, -1].shape, dtype = batch['action'].dtype) for act_ in dat]
        batch['action_sample'] = np.array(dat)
        return batch 
        
class MultiEnvironmentDataProvider(object):
    def __init__(self, environments, data_provider_constructor, **provider_params):
        provider_params_copy = copy.copy(provider_params)
        self.envs_ = environments
        self.dps_ = []
        assert provider_params['batching_params']['batch_size'] % len(environments) == 0, 'Number of environments must divide batch size'
        for _k, _env in enumerate(environments):
            _params_copy = copy.copy(provider_params)
            _params_copy['batching_params'] = copy.copy(provider_params['batching_params'])
            _params_copy['batching_seed'] = provider_params['batching_seed'] + _k
            _params_copy['batching_params']['batch_size'] = provider_params['batching_params']['batch_size'] / len(environments)
            self.dps_.append(data_provider_constructor(environment = _env, ** _params_copy))

    def start_runner(self, sess, agents):
        for _dp in self.dps_:
            _dp.start_runner(sess, agents)

    def dequeue_batch(self):
        batches = [_dp.dequeue_batch() for _dp in self.dps_]
        batches = {k_ : [b_[k_] for b_ in batches] for k_ in batches[0]}
        #for k_, bpc_ in batches.items():
        #    print(k_)
        #    print(bpc_[0].shape)
        return {k_ : np.concatenate(v_) if k_ not in ['recent'] else v_ for k_, v_ in batches.items()}


