'''
Computation grap-building components




'''

import numpy as np
import tensorflow as tf
import copy
import model_utils

def get_reward_holders(cfg):
    '''
    Takes cfg and makes placeholders for our feed dicts
    
    reward holders is the default, meant to also hold a generic reward that might have been computed at some previous time.
    '''
    images_holder = tf.placeholder(tf.uint8, [None] + cfg['states_shape'], name = 'images')
    action = tf.placeholder(tf.float32, [None] + cfg['action_shape'], name = 'action')
    action_post = tf.placeholder(tf.float32, [None] + cfg['action_shape'], name = 'action_post')
    reward = tf.placeholder(tf.float32, [None, cfg['grab_losses_end'] - cfg['grab_losses_start']], name = 'reward')
    objthere = tf.placeholder(tf.int32, [None, cfg['grab_objthere_end'] - cfg['grab_objthere_start']], name = 'obj_there')
    action_sample = tf.placeholder(tf.float32, [None] + cfg['action_shape'][1:], name = 'action_sample')
    return {
                'images1' : images_holder,
                'action' : action,
                'action_post' : action_post,
                'action_sample' : action_sample,
                'reward' : reward,
                'obj_there' : objthere,
                #'obj_in_frame' : tf.placeholder(tf.int32, [None, cfg['grab_objthere_end'] - cfg['grab_objthere_start']], name = 'obj_in_frame'),
                'train_indicator' : tf.placeholder(tf.float32, shape = (), name = 'train_indicator')
           }


def get_reward_inputs(holders, cfg):
    '''
    Performs postprocessing on holders for all the rest.
    Treat as a default.
    '''
    lm_begin_idx, lm_end_idx = cfg.get('lm_state_idxs', (3, 5))
    images_cast = tf.cast(holders['images1'], tf.float32)
    images = model_utils.postprocess_std(images_cast)
    lm_images = tf.concat([images[:, t_] for t_ in range(lm_begin_idx, lm_end_idx)], axis = 3)
    reward = holders['reward']
    compare_idx = cfg['reward_comparison_idx'] - cfg['grab_losses_start']
    compare_reward = reward[:, compare_idx : compare_idx + 1]
    lm_reward_start = cfg['lm_reward_start'] - cfg['grab_losses_start']
    wm_losses = reward[:, lm_reward_start : ]
    wm_losses = [wm_losses[:, t_ : t_ + 1] for t_ in range(wm_losses.get_shape().as_list()[1])]
    return {'images' : images, 'action' : holders['action'], 'action_post' : holders['action_post'],
            'action_sample' : holders['action_sample'], 'obj_there' : holders['obj_there'],
            'compare_act_loss' : compare_reward, 'wm_losses' : wm_losses, 'lm_images' : lm_images,
            #'obj_in_frame' : holders['obj_in_frame'], 
            'train_indicator' : holders['train_indicator']
}


def get_state_encodings(inputs, model_desc, scope_name, cfg):
    '''A generic encoding.
    cfg args:
    'state_dict':
        {name_of_state : [which images to grab]}
    '''
    state_dict = cfg[model_desc]['state_dict']
    s_first = state_dict.keys()[0]
    #check that all states are of the same size
    assert all([len(v) == len(state_dict[s_first]) for v in state_dict.values()])

    m = model_utils.ConvNetwithBypasses()
    outputs = {}
    with tf.variable_scope(scope_name):
        #supposed to start off with a clean slate
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)) == 0

        #collect states, concatenated along channel
        states_collected = {nm_ : tf.concat([inputs['images'][:, t_] for t_ in st_], axis = 3)
                for nm_, st_ in state_dict.items()}

        #encode stuff
        outputs[model_desc + '_encoding'] ={\
                nm_ : model_utils.feedforward_conv_loop(collected_state_, m,\
                                            cfg[model_desc]['encode'],
                                            desc = 'encode', bypass_nodes = None,
                                            reuse_weights = (nm_ != s_first),
                                            batch_normalize = False,
                                            no_nonlinearity_end = False)
                for nm_, collected_state_ in states_collected.items()
                }

        outputs[model_desc + '_encode_var_list'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        assert len(outputs[model_desc + '_encode_var_list']) > 0
    return outputs


def get_leaveout_mlp_heads_helper(inputs, encodings, other_givens, model_desc, cfg, scope_dict, reuse_all_vars = False):
    outputs = {'pred' : {}, 'tv' : {}, 'var_list' : {}}
    #here we cut out any skips, need to change for any skip using earlier encodings
    flat_encodings = {nm_ : model_utils.flatten(enc_[-1]) for nm_, enc_ in encodings[model_desc + '_encoding'].items()}
    s_first = flat_encodings.keys()[0]
    m = model_utils.ConvNetwithBypasses()

    #optional post-encoding state processing
    if 'mlp_before_concat' in cfg[model_desc]:
        with tf.variable_scope(scope_dict.get('before_concat', 'before_concat')):
            flat_encodings = {s : model_utils.hidden_loop_with_bypasses(enc_flat_, m, cfg[model_desc]['mlp_before_concat'], 
                        reuse_weights = (s != s_first) or reuse_all_vars, train_indicator = inputs['train_indicator'])
                    for s, enc_flat_ in flat_encodings.items()}
            outputs['before_concat_var_list'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    #make predictions
    reuse_for_shared = reuse_all_vars
    for nm_, (states_given_, acts_given_) in cfg[model_desc]['givens'].items():
        if nm_ == 'FAKE_ID':
            continue
        (states_tv_, acts_tv_) = cfg[model_desc]['tvs'][nm_]
        encoded_states_given = [flat_encodings[s_] for s_ in states_given_]
        encoded_states_tv = [flat_encodings[s_] for s_ in states_tv_]
        images_tv = [inputs['images'][:, s_] for s_ in states_tv_]
        action_given = [inputs['action'][:, a_] for a_ in acts_given_]
        action_tv = [inputs['action_post'][:, a_] for a_ in acts_tv_]
        to_concat = encoded_states_given + action_given
        if nm_ in other_givens:
            to_concat += other_givens[nm_]
        x = tf.concat(to_concat, axis = 1)
        if 'shared_mlp' in cfg[model_desc]:
            with tf.variable_scope(scope_dict.get('shared_mlp', 'shared_mlp')):
                x = model_utils.hidden_loop_with_bypasses(x, m, \
                        cfg[model_desc]['shared_mlp'], \
                        reuse_weights = reuse_for_shared, train_indicator = inputs['train_indicator'])
            reuse_for_shared = True
        with tf.variable_scope(scope_dict.get(nm_, nm_)):
            pred = x
            if 'mlp' in cfg[model_desc] and nm_ in cfg[model_desc]['mlp']:
                pred = model_utils.hidden_loop_with_bypasses(pred, m, \
                        cfg[model_desc]['mlp'][nm_], \
                        reuse_weights = reuse_all_vars, train_indicator = inputs['train_indicator'])
                outputs['tv'][nm_] = (encoded_states_tv, action_tv)
                outputs['pred'][nm_] = pred
            if 'deconv' in cfg[model_desc] and nm_ in cfg[model_desc]['deconv']:
                pred = tf.reshape(pred, [-1, 11, 8, 1])
                pred = model_utils.deconv_loop(pred, m, \
                        cfg[model_desc]['deconv'][nm_], desc='deconv', \
                        reuse_weights = reuse_all_vars)
                outputs['tv'][nm_] = (images_tv, action_tv)
                outputs['pred'][nm_] = tf.image.resize_image_with_crop_or_pad(
                        pred[-1], 128, 170)
            outputs['var_list'][nm_] = tf.get_collection( \
                    tf.GraphKeys.TRAINABLE_VARIABLES, \
                    tf.get_variable_scope().name)
    outputs['flat_encodings'] = flat_encodings
    return outputs


def get_leaveout_mlp_heads(inputs, encodings, other_givens, model_desc, cfg, scope_name = None, scope_dict = {}, reuse_all_vars = False):
    '''
    Computes readouts (with givens fed into model)
    and holds onto corresponding  true values for loss computation.
    Meant to support readouts feeding in any number of encoded states
    and actions, with actions concatenated in
    cfg args:
        'givens' : {prediction_name_0 : (state names given_0, action_number given_0),
                        prediction_name_1 : (state_names_given_1, action_num_given_1),
                        ...}
        'tvs' : {prediction_name_0 : (state_tv_0, act_tv_0)...}
    '''
    if scope_name is None:
        return get_leaveout_mlp_heads_helper(inputs, encodings, other_givens, model_desc, cfg, scope_dict = scope_dict, reuse_all_vars = reuse_all_vars)
    else:
        with tf.variable_scope(scope_name):
            #clean slate
            assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)) == 0 or reuse_all_vars           
            return get_leaveout_mlp_heads_helper(inputs, encodings, other_givens, model_desc, cfg, scope_dict = scope_dict, reuse_all_vars = reuse_all_vars)

def get_leaveout_losses(inputs, leaveout_outputs, model_desc, cfg, loss_kwargs = {}):
    '''
    Computes losses for the leaveout models above.
    cfg args:
        cfg[model_desc]['loss'][prediction_name]['func'] = loss_function(true_value, prediction, cfg)
    '''
    loss_per_example = {}
    losses = {}
    for nm_, tv_ in leaveout_outputs['tv'].items():
        pred_ = leaveout_outputs['pred'][nm_]
        lpe, loss = cfg[model_desc]['loss'][nm_]['func'](tv_, pred_, cfg[model_desc]['loss'][nm_], **loss_kwargs)
        loss_per_example[nm_] = lpe
        losses[nm_] = loss
    return {'loss_per_example' : loss_per_example, 'losses' : losses}




def get_default_lm_heads(inputs, encodings, cfg):
    n_timesteps = cfg['self_model']['n_timesteps_fwd']
    
    with tf.variable_scope('self_model'):
        outputs = {'estimated_world_loss' : []}

        x = model_utils.flatten(encodings['self_model']['self_model_encoding']['state'][-1])
        #after-encoding mlp, pre-actions (e.g. choke down)
        m = model_utils.ConvNetwithBypasses()
        with tf.variable_scope('before_action'):
            x = before_act = model_utils.hidden_loop_with_bypasses(x, m, 
                          cfg['self_model']['shared_mlp_before_action'], 
                          reuse_weights = False, train_indicator = inputs['train_indicator'])

        #option to concatenate other actions
        if 'concat_actions' in cfg['self_model']:
            x = tf.concat([x] + [inputs['lm_action'][:, t_] for t_ in cfg['self_model']['concat_actions']], axis = 1)

        #if we are computing the uncertainty map for many action samples, tile to use the same encoding for each action
        #x = tf.cond(tf.equal(tf.shape(inputs['action_sample'])[0], cfg['self_model']['n_action_samples']), lambda : tf.tile(x, [cfg['self_model']['n_action_samples'], 1]), lambda : x)

        #concatenate action
        action_sample = tf.random_uniform([cfg['policy']['n_action_samples'],
                cfg['policy']['action_dim']], minval = -1., maxval = 1., dtype = tf.float32)
        action_sample = tf.cond(tf.equal(inputs['train_indicator'], 0.), 
             lambda : action_sample,
             lambda : inputs['action_sample'])
        #if we are computing the uncertainty map for many action samples, tile to use the same encoding for each action
        x = tf.cond(tf.equal(tf.shape(action_sample)[0], cfg['self_model']['n_action_samples']), lambda : tf.tile(x, [cfg['self_model']['n_action_samples'], 1]), lambda : x)
        x = tf.concat([x, action_sample], 1)
        

        #shared mlp after action
        if 'shared_mlp' in cfg['self_model']:
            with tf.variable_scope('shared_mlp'):
                x = model_utils.hidden_loop_with_bypasses(x, m, cfg['self_model']['shared_mlp'], reuse_weights = False, train_indicator = inputs['train_indicator'])

        #making predictions
        for t in range(n_timesteps):
            with tf.variable_scope('split_mlp' + str(t)):
                outputs['estimated_world_loss'].append(model_utils.hidden_loop_with_bypasses(x, m,\
                            cfg['self_model']['mlp'][t], reuse_weights = False,\
                            train_indicator = inputs['train_indicator']))
        outputs['lm_var_list'] = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)]

    outputs['probs_per_timestep'] = [tf.nn.softmax(logits) for logits in outputs['estimated_world_loss']]
    outputs['before_action'] = before_act
    outputs['action_sample'] = action_sample
    return outputs



def get_feed_in_losses(inputs, lm_outputs, cfg):
    '''
    Compares lm predictions with wm losses observed.
    '''
    true_loss = inputs['wm_losses']
    assert len(true_loss) == len(lm_outputs['estimated_world_loss']), (len(true_loss), len(lm_outputs['estimated_world_loss']))
    lm_lpe, lm_loss_per_step, lm_loss = cfg['self_model']['loss_func'](true_loss,\
                   lm_outputs['estimated_world_loss'], cfg['self_model'])
    return {'lm_lpe' : lm_lpe, 'lm_loss_per_step' : lm_loss_per_step, 'lm_loss' : lm_loss}

def get_random_policy(lm_outputs, cfg):
    random_idx = tf.random_uniform(shape = [1], 
        maxval = cfg['policy']['n_action_samples'],
        dtype = tf.int32)
    action_chosen = lm_outputs['action_sample'][random_idx[0]]
    return {'action_chosen' : action_chosen,
            'random_alternative' : action_chosen,
            'reward' : losses['world_model']['losses']['ID'],
            'action_sample' : lm_outputs['action_sample'],
            'probs_per_timestep' : lm_outputs['probs_per_timestep'],
           }
            


def get_sum_sampling_policy(lm_outputs, losses, cfg):
    '''
    Computes policy sample from default lm heads.
    '''
    probs_per_timestep = lm_outputs['probs_per_timestep']
    n_classes = probs_per_timestep[0].get_shape().as_list()[-1]
    expected_class_per_timestep = [sum([probs[:, i:i+1] * float(i) for i in range(n_classes)]) for probs in probs_per_timestep]
    expected_tot_est = sum(expected_class_per_timestep)
    heat = tf.constant(cfg['policy']['heat'], dtype = tf.float32)
    x = tf.transpose(expected_tot_est) / heat
    sample = model_utils.categorical_sample(x, cfg['policy']['n_action_samples'], one_hot = False)
    random_idx = tf.random_uniform(shape = [1], 
        maxval = cfg['policy']['n_action_samples'],
        dtype = tf.int32)
    return {'action_chosen' : lm_outputs['action_sample'][sample[0]], 
            'action_sample' : lm_outputs['action_sample'], 
            'probs_per_timestep' : probs_per_timestep,
            'reward' : losses['world_model']['losses']['ID'],
            'random_alternative' : lm_outputs['action_sample'][random_idx[0]]
           }
    
POLICY_CONSTRUCTORS = {
    'thom_sampling' : get_sum_sampling_policy,
    'random' : get_random_policy,    

}


def get_default_signals_vars(inputs, encodings, wm_outputs, lm_outputs, losses):
    '''
    Gets signals and vars for a default world model and the default lm model.
    Assumes that the world_model encoding gets updated by all world_model loss signals.
    '''
    sum_losses = sum(losses['world_model']['losses'].values())
    all_wm_vars = [v_ for var_list_ in wm_outputs['var_list'].values() for v_ in var_list_]\
            + encodings['world_model']['world_model_encode_var_list']\
            + wm_outputs['before_concat_var_list']
    return {
                'world_model' : (sum_losses, all_wm_vars),
                'self_model' : (losses['self_model']['lm_loss'], encodings['self_model']['self_model_encode_var_list'] + lm_outputs['lm_var_list'])
            }


def get_default_training_targets(inputs, wm_outputs, lm_outputs, losses, cfg):
    readouts = {nm_ + '_loss' : l_ for nm_, l_ in losses['world_model']['losses'].items()}
    readouts['self_model_loss'] = losses['self_model']['lm_loss']

    #want to add in object there conditioning, but want to think through and simplify.
    #also TODO: collision conditioning, action stats conditioned on both of these
    #round version, separately
    return readouts


def get_indicator_conditions(indicators, values, cfg):
    '''
    Conditions values on indicators...you know, multiplies them, averaging over batch,
    tiling the indicators if necessary.
    Assumes that the values are a dict, each value 
    of shape [None (batch_size), something],
    Assumes that the indicators are a dict, each value 
    a list (meant to vary some time dependence) 
    of shapes [None (batch_size), 1].
    Tiles the indicators accordingly.
    '''
    to_return = {}
    for ind_nm_, indicator_ in indicators.items():
        for val_nm_, tens_ in values.items():
            tile_num = tens_.get_shape().as_list()[1]
            indicator_tiled_ = [tf.tile(ind_, [1, tile_num]) for ind_ in indicator_]
            to_return[ind_nm_ + '_' + val_nm_ + '_noprint'] =\
                [tf.reduce_sum(tens_ * ind_tiled_, axis = 0) / tf.reduce_sum(ind_)\
                     for (ind_, ind_tiled_) in zip(indicator_, indicator_tiled_)]
    return to_return


def get_paired_down_training_targets(inputs, wm_outputs, lm_outputs, losses, cfg):
    '''The basic training targets for replicating first paper.
    '''
    #where we put all indicators
    indicators = {}

    numobj_indicators = {}
    for num_there_ in range(cfg['train_targets']['indicator_num_objects'] + 1):
        are_there = tf.cast(tf.equal(inputs['obj_there'], num_there_), tf.int32)
        objthere_wm = [are_there[:, t_ : t_ + 1] for t_ in\
                range(cfg['train_targets']['obj_there_wm_start_idx'],\
                cfg['train_targets']['obj_there_wm_end_idx'])]

        numobj_indicators[str(num_there_) + 'there'] = [tf.cast(ind_, tf.float32) for ind_ in objthere_wm]
    indicators.update(numobj_indicators)
 
    acc_01 = tf.cast(model_utils.binned_01_accuracy_per_example(wm_outputs['tv']['ID'][1][0],\
            wm_outputs['pred']['ID'], {'thresholds' : [-.1, .1]}), tf.float32)
   
    wm_for_conditioning = {
            'act_loss' : losses['world_model']['loss_per_example']['ID'],
            'act_acc' : acc_01,
    }


    condition_readouts = get_indicator_conditions(indicators, wm_for_conditioning, cfg)

    objthere_lm = [inputs['obj_there'][:, t_ : t_ + 1] for t_ in\
            range(cfg['train_targets']['obj_there_lm_start_idx'],\
            cfg['train_targets']['obj_there_wm_end_idx'])]

    lm_for_conditioning = {
        'lmprob' + str(i_) : ppt_
        for (i_, ppt_) in enumerate(lm_outputs['probs_per_timestep']) if i_ in [0, 1]
    }
   
    
    indicators_lm = {
        str(num_there_) + 'there' : [tf.cast(tf.equal(ind_, num_there_), tf.float32) for ind_ in objthere_lm]
        for num_there_ in range(cfg['train_targets']['indicator_num_objects'] + 1)
    }

    lm_condition_readouts = get_indicator_conditions(indicators_lm, lm_for_conditioning, cfg)
    assert(len(set(condition_readouts.keys()).intersection(set(lm_condition_readouts.keys()))) == 0)
    
    for readouts in [condition_readouts, indicators_lm]:
        for nm in readouts:
            readouts[nm] = readouts[nm][2]

    training_targets = get_default_training_targets(inputs, wm_outputs, lm_outputs, losses, cfg)
    training_targets.update(condition_readouts)
    training_targets.update(lm_condition_readouts)
    return training_targets

def get_reward_graph(cfg):
    holders = get_reward_holders(cfg)
    inputs = get_reward_inputs(holders, cfg)
    wm_encodings = get_state_encodings(inputs, 'world_model', scope_name = 'wm_encoding', cfg = cfg)
    lm_encodings = get_state_encodings(inputs, 'self_model', scope_name = 'lm_encoding', cfg = cfg)
    encodings = {'world_model' : wm_encodings, 'self_model' : lm_encodings}
    wm_outputs = get_leaveout_mlp_heads(inputs, wm_encodings, other_givens = {}, model_desc = 'world_model', scope_name = 'wm_heads', cfg = cfg)
    lm_outputs = get_default_lm_heads(inputs, encodings, cfg)
    wm_losses = get_leaveout_losses(inputs, wm_outputs, model_desc = 'world_model', cfg = cfg)
    lm_losses = get_feed_in_losses(inputs, lm_outputs, cfg)
    losses = {'world_model' : wm_losses, 'self_model' : lm_losses}
    training_targets = get_paired_down_training_targets(inputs, wm_outputs, lm_outputs, losses, cfg)
    signals_and_variables = get_default_signals_vars(inputs, encodings, wm_outputs, lm_outputs, losses)
    policy = POLICY_CONSTRUCTORS[cfg['policy']['mode']](lm_outputs, losses, cfg)
    return {'holders' : holders, 'inputs' : inputs, 'encodings' : encodings, 'wm_outputs' : wm_outputs,
                 'signals_and_variables' : signals_and_variables,
                 'lm_outputs' : lm_outputs, 'losses' : losses, 'train_targets' : training_targets, 'policy' : policy}



