import os
import tensorflow as tf
import tfutils.base as base
import numpy as np
import time
import tqdm
import copy
from pdb import set_trace
import argparse


def copy_pop_apply(my_dict, default=None, func_key='func', other_kwargs={}):
    copy_dict = copy.copy(my_dict)
    # fails if default is not set and func not present
    func = copy_dict.pop(func_key) if default is None else copy_dict.pop(func_key, default)
    copy_dict.update(other_kwargs)
    return func(**copy_dict)


def get_empty(**kwargs):
    return {}


def train_from_params(
        resource_params,
        agent_params,
        updater_params,
        save_params,
        session_params,
        load_params=None,
        train_from_params_params = {},
):
    ''' Does training loop from params.
    Args:
        Each of these is a dict with a 'func' key and other keys that are args to the 'func' val.
        resource_params,
        agent_params,
        updater_params,
    '''
    # just for saving. could also input as all kwargs.
    params = {'resource_params': resource_params, 'agent_params': agent_params, 'updater_params': updater_params,
              'save_params': save_params, 'load_params': load_params, 'session_params': session_params, 
              'train_from_params_params' : train_from_params_params}

    # allocate resources. place to set up server, but in base case nothing
    resource_res = copy_pop_apply(resource_params, default=get_empty)  # Each worker/ps has a tf.train.Server object
    print("Got Worker Server")

    # get session
    sess_res = copy_pop_apply(session_params, other_kwargs={'resource_res': resource_res, 'params': params})
    print("Got Session")

    # define environments and things interacting with them (implicitly defines computational graph)
    agents = copy_pop_apply(agent_params, other_kwargs={'compute_settings': resource_res, 'stored_params': params})
    print("Defined Agents")  # AsyncSelfModelAgent class

    # set up updater
    updater = copy_pop_apply(updater_params, other_kwargs={'resource_res': resource_res, 'agents': agents, 'params': params, 'sess_res': sess_res})  # default get session
    print("Updater Initialized")

    db = base.DBInterface(sess=sess_res['sess'], global_step=updater.global_step, params=params, save_params=save_params, load_params=load_params)
    db.initialize()
    print("DB Initialized")

    # now that we've loaded variables, we can start things up
    updater.start(sess_res)

    local_step = -1
    always_train = 'train_steps' not in train_from_params_params

    # the loop
    while always_train or local_step < train_from_params_params['train_steps']:
        db.start_time_step = time.time()
        res, local_step = updater.update()  # res = {agent output, validation output}
        db.save(train_res=res['train_res'], valid_res=res.get('valid_res', {}), validation_only=False, step=local_step)  # change to task step
        db.sync_with_host()

    updater.end()


