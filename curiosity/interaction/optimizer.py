'''
Some optimization utils, meant to mirror tfutils.
'''
from tfutils.utils import frozendict
import tensorflow as tf
from pdb import set_trace

DEFAULT_OPTIMIZER_PARAMS = frozendict({'optimizer_class': tf.train.MomentumOptimizer,
                                                                              'momentum': 0.9})

def collect_dicts(my_list_of_dicts):
    '''Takes a list of dicts and makes a dict of lists!'''
    retval = {}
    for my_dict in my_list_of_dicts:
        for k, v in my_dict.items():
            if k in retval:
                retval[k].append(v)
            else:
                retval[k] = [v]
    return retval

class MultiLossOpt(object):
    def __init__(self, optimizer_class, clip = True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip

    def compute_gradients(self, loss_var_pairs, *args, **kwargs):
        lvp_gvs = [self._optimizer.compute_gradients(l, var_list = v)
                for l, v in loss_var_pairs]
        #I'm not exactly sure how tf does hashing, so...
        lvp_gvs_name_dicts = [{var.name : (grad, var) for (grad, var) in _gvs}
                for _gvs in lvp_gvs]
        collected_gvs = collect_dicts(lvp_gvs_name_dicts)
        for nm, lst in collected_gvs.items():
            print nm
            print len(lst)
        summed_gvs = []
        for _var_gvs in collected_gvs.values():
            collected_grads = [g for (g, v) in _var_gvs if g is not None]
            if len(collected_grads) == 0:
                continue
            var = _var_gvs[0][1]
            summed_gvs.append((sum(collected_grads), var))
        #summed_gvs = [(sum([g for (g, v) in _var_gvs if g is not None]), _var_gvs[0][1])
        #        for _var_gvs in collected_gvs.values()]
        if self.clip:
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                    for grad, var in summed_gvs if grad is not None]
        return gvs

    def minimize(self, loss, global_step, var_list = None):
        grad_and_vars = self.compute_gradients(loss_var_pairs = loss)
        return self._optimizer.apply_gradients(grad_and_vars,
                global_step = global_step)

class CuriousClipOptimizer(object):

    def __init__(self, optimizer_class, clip=True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip
        
    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        return gvs
    
    def minimize(self, loss, global_step, compute_grad_var_list = None, update_var_list = None):
        grads_and_vars = self.compute_gradients(loss, var_list = compute_grad_var_list)
        grads_and_vars = [(g_local, v_global) for ((g_local, v_local), v_global) in zip(grads_and_vars, update_var_list)]
        return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)

def get_curious_optimizer(learning_rate,
                  loss,
                  global_step,
                  optimizer_params,
                  update_var_list = None,
                  compute_grad_var_list = None,
                  default_optimizer_params = DEFAULT_OPTIMIZER_PARAMS,
                  default_optimizer_func = CuriousClipOptimizer):
    if optimizer_params is None:
        optimizer_params = dict(default_optimizer_params)
    func = optimizer_params.pop('func', default_optimizer_func)
    optimizer_base = func(learning_rate = learning_rate,
                          **optimizer_params)
    optimizer = optimizer_base.minimize(loss, global_step, compute_grad_var_list, update_var_list)
    optimizer_params['func'] = func
    return optimizer_params, optimizer

