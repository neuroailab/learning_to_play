
import numpy as np
import tensorflow as tf
import copy, tqdm
import random
import time
from tfutils.base import get_learning_rate
from optimizer import get_curious_optimizer
from pdb import set_trace
import time
from data import environment, online_data

def run_targets(sess,
        dbinterface,
        target,
        data_provider,
        placeholders,
        target_name,
        num_steps,
        online_agg_func,
        agg_func,
        save_intermediate_freq=None,
        validation_only=False):
    '''Helper, adapted from tfutils.'''
    agg_res = None

    if save_intermediate_freq is not None:
        n0 = len(dbinterface.outrecs)

    for _step in tqdm.trange(num_steps, desc=target_name):
        batch = data_provider.dequeue_batch()
        batch['train_indicator'] = 0.
        feed_dict = {placeholders[k_] : batch[k_] for k_ in placeholders}
        res = sess.run(target, feed_dict = feed_dict)
        res['batch'] = batch
        assert hasattr(res, 'keys'), 'result must be a dictionary'
        if save_intermediate_freq is not None and _step % save_intermediate_freq == 0:
            dbinterface.save(valid_res={target_name: res},
                    step=_step,
                    validation_only=validation_only)
        agg_res = online_agg_func(agg_res, res, _step)

    result = agg_func(agg_res)

    if save_intermediate_freq is not None:
        dbinterface.sync_with_host()
        n1 = len(dbinterface.outrecs)
        result['intermediate_steps'] = dbinterface.outrecs[n0: n1]

    return result

class Updater(object):
    '''
    Abstract base class
    '''
    def __init__(self, resource_res, agents, params):
        with tf.variable_scope(tf.get_variable_scope(),  reuse = tf.AUTO_REUSE):
            self.global_step = tf.get_variable('global_step', [], tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64))


    def start(self, session):
        '''
        Starts loops not in sync with the main train loop
        e.g. data collection
        '''
        raise NotImplementedError()

    def update(self):
        '''
        The thing that the train loop calls.
        Results must be of the form
        results (to be put in dbinterface), step (distinguished value for dbinterface)
        '''
        raise NotImplementedError()


class SimpleUpdater(Updater):
    def __init__(self, resource_res, agents,
                 learning_rate_params,
                 optimizer_params,
                 data_params,
                 postprocess_func,
                 params, 
                 sess_res,
                 validation_params = None,
):
        super(SimpleUpdater, self).__init__(resource_res, agents, params)
        self.agents = agents
        #define optimization
        signals_and_vars = agents.get_signals_and_variables()
        sig_keys = signals_and_vars.keys()
        signal_steps = {k_ : tf.get_variable(k_ + '_step', [], tf.int64, initializer = tf.constant_initializer(0, dtype = tf.int64)) for k_ in sig_keys}
        lrs = {k_ + '_lr' : get_learning_rate(signal_steps[k_], **learning_rate_params[k_])[1] for k_ in sig_keys}
        optimizer_params = optimizer_params
        opts = {k_ + '_opt' : get_curious_optimizer(learning_rate = lrs[k_ + '_lr'],
                                              loss = signals_and_vars[k_][0],
                                              global_step = signal_steps[k_],
                                              optimizer_params = optimizer_params[k_],
                                              update_var_list = signals_and_vars[k_][1],
                                              compute_grad_var_list = signals_and_vars[k_][1])[1]
                            for k_ in sig_keys}
        #create train targets
        self.train_targets = {
            'global_step' : self.global_step,
            'global_increment' : tf.assign_add(self.global_step, 1),
        }
        self.train_targets.update(lrs)
        self.train_targets.update(opts)
        self.train_targets.update(agents.get_train_targets())


        #prepare update loop
        self.sess = sess_res['sess']
        self._initialize_data_provider(data_params)
        self.placeholders = agents.get_update_holders()
        self._postprocess_func_proto = postprocess_func

        #prepare validation
        self.valid_params = validation_params
        self.valid_freq = params['save_params']['save_valid_freq']
        self._initialize_validation(self.valid_params)

    def _initialize_data_provider(self, data_params):
        env_params = data_params['environment_params']
        assert len(env_params['random_seed']) == \
                len(env_params['unity_seed']) and \
               len(env_params['gpu_num']) == len(env_params['unity_seed'])

        envs = []

        for (rs_, us_, g_) in zip(env_params['random_seed'], env_params['unity_seed'], env_params['gpu_num']):
            params_copy = copy.copy(env_params)
            params_copy['random_seed'] = rs_
            params_copy['unity_seed'] = us_
            params_copy['gpu_num'] = g_
            params_copy.pop('func', None)
            #new_env = environment.GaussianFingersEnvironment(**params_copy)
            new_env = env_params.get('func', environment.TelekineticMagicianEnvironment)(**params_copy)
            envs.append(new_env)
        _dp_func = data_params.get('func', online_data.BufferSampleDataProvider)

        self.dp = online_data.MultiEnvironmentDataProvider(environments = envs,
            data_provider_constructor = _dp_func,
             ** data_params['provider_params'])

    def start(self, sess_res):
        session = sess_res['sess']
        self.dp.start_runner(session, self.agents)
        for k_, dp in self.valid_dps.items():
            dp.start_runner(session, self.valid_agents[k_])


    def _initialize_validation(self, validation_params):
        if validation_params is None:
            self.valid_dps, self.valid_targets, self.valid_holders = {}, {}, {}
            return
        dps = {}
        validation_targets = {}
        valid_holders = {}
        valid_agents = {}
        all_placeholders = self.agents.get_all_holders()
        for k_, v_params_ in validation_params.items():
            dps[k_] = v_params_['data_params']['func'](v_params_['data_params'])
            validation_targets[k_] = v_params_['get_targets'](self.agents, **v_params_['kwargs'])
            valid_holders[k_] = {k_ : all_placeholders[k_] 
                for k_ in v_params_['holder_keys'] + ['train_indicator']} if 'holder_keys' in v_params_\
                else self.placeholders
            valid_agents[k_] = v_params_['get_agent'](self.agents, v_params_['agent_params'])
        self.valid_dps = dps
        self.valid_targets = validation_targets
        self.valid_holders = valid_holders
        self.valid_agents = valid_agents

    def _postprocess_func(self, results, batch):
        #just calls input for now, but in general the idea is that you can maintai the state
        return self._postprocess_func_proto(results, batch)

    def update(self):
        batch = self.dp.dequeue_batch()
        batch['train_indicator'] = 1.
        #print('batch')
        #for k_, b_ in batch.items():
        #    print(k_)
        #    print(type(b_))
        #    try:
        #        print(b_.shape)
        #        print(b_.dtype)
        #    except:
        #        print('not an array')
        #    if k_ == 'reward':
        #        print(b_)
        #print('placeholders')
        #for k_, b_ in placeholders.items():
        #    print(k_)
        #    print(b_)
        feed_dict = {self.placeholders[k_] : batch[k_] for k_ in self.placeholders}
        res = {}      
        res['train_res'] = self.sess.run(self.train_targets, feed_dict = feed_dict)
        valid_res = {}
        if self.valid_params and res['train_res']['global_step'] % self.valid_freq == 0:
            for k_ in self.valid_params:
                num_steps = self.valid_params[k_]['num_steps']
                targets = self.valid_targets[k_]
                agg_func = self.valid_params[k_]['agg_func']
                online_agg_func = self.valid_params[k_]['online_agg_func']
                dp = self.valid_dps[k_]
                holders = self.valid_holders[k_]
                valid_res[k_] = run_targets(sess = self.sess,
                    dbinterface = None,
                    target = targets,
                    data_provider = dp,
                    placeholders = holders,
                    target_name = k_,
                    num_steps = num_steps,
                    online_agg_func = online_agg_func,
                    agg_func = agg_func,
                    save_intermediate_freq = None,
                    validation_only = False,
                    )
        res['valid_res'] = valid_res
        self.global_step_eval = res['train_res']['global_step']
        res = self._postprocess_func(res, batch)
        return res, res['train_res']['global_step']
