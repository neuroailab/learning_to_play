import tensorflow as tf



def get_session(resource_res, params, **session_params):
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    return {'sess' : tf.Session(config = config)}


def generate_mlp_architecture_cfg(num_features = [20, 1], dropout = [None, None], nonlinearities = ['relu', 'identity']):
	retval = {}
	assert len(num_features) == len(dropout) and len(dropout) == len(nonlinearities)
	retval['hidden_depth'] = len(num_features)
	retval['hidden'] = {}
	for i, (nf, drop, nl) in enumerate(zip(num_features, dropout, nonlinearities)):
		retval['hidden'][i + 1] = {'num_features' : nf, 'dropout' : drop, 'activation' : nl}
	return retval

def generate_conv_architecture_cfg(desc = 'encode', sizes = [3, 3, 3, 3, 3], strides = [2, 1, 2, 1, 2], num_filters = [20, 20, 20, 10, 4], 
                                    bypass = [None, None, None, None, None], nonlinearity = None,
                                    poolsize = None, poolstride = None, batch_normalize = None):
	retval = {}
	if nonlinearity is None:
		nonlinearity = ['relu' for _ in sizes]
        if poolsize is None:
            poolsize = [None for _ in sizes]
        if poolstride is None:
            poolstride = [None for _ in sizes]
        if batch_normalize is None:
            batch_normalize = [False for _ in sizes]
        else:
            assert len(batch_normalize) == len(sizes)
	assert len(sizes) ==  len(strides) and len(num_filters) == len(strides) and len(bypass) == len(strides)
	retval[desc + '_depth'] = len(sizes)
	retval[desc] = {}
	for i, (sz, stride, nf, byp, nl, psz, pstr, bn) in enumerate(zip(sizes, strides, num_filters, bypass,\
                            nonlinearity, poolsize, poolstride, batch_normalize)):
		retval[desc][i + 1] = {'conv' : {'filter_size' : sz, 'stride' : stride, 'num_filters' : nf, 'batch_normalize' : bn}, 'bypass' : byp, 'nonlinearity' : nl}
                if psz is not None:
                    retval[desc][i+1]['pool'] = {'size' : psz, 'stride' : pstr, 'type' : 'max'}
	return retval



#various encoding choices, selected via command line
encoding_choices = [


        {
            'sizes' : [3, 3, 3, 3, 3, 3, 3, 3],
            'strides' : [1, 1, 1, 1, 1, 1, 1, 1],
            'num_filters' : [64, 64, 64, 64, 64, 64, 64, 64],
            'poolsize' : [None, 3, None, 3, None, 3, None, 3],
            'poolstride' : [None, 2, None, 2, None, 2, None, 2],
            'bypass' : [None, None, None, None, None, None, None, None]
        },

        {
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]   
        },                                                                                  

        {       
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [64, 64, 64, 64, 128, 128, 128, 128, 192, 192, 192, 192],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]           
        },


        {
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [32 for _ in range(12)],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]   
        },                                                                                  


        {       
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [32, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 96],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]           
        },
                                                                                        
        {       
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]           
        },

        {       
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]           
        },

         {       
             'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             'strides' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'num_filters' : [128, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128],
             'bypass' : [None, None, None, None, None, None, None, None, None, None, None, None],
             'poolsize' : [None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
             'poolstride' : [None, 2, None, 2, None, 2, None, 2, None, 2, None, 2]           
        },
   


]


#fully-connected choices ending in the right number of readouts for the world model
def get_wm_mlp_choices(action_dim, n_classes_wm):
    return [

        {
                'num_features' : [512, action_dim * n_classes_wm],
                'nonlinearities' : ['relu', 'identity'],
                'dropout' : [.5, None]
        },


        {
                'num_features' : [192, action_dim * n_classes_wm],
                'nonlinearities' : ['relu', 'identity'],
                'dropout' : [.5, None]
        },


        {
                'num_features' : [512, 384, action_dim * n_classes_wm],
                'nonlinearities' : ['relu', 'identity'],
                'dropout' : [.5, None]
        },

        {
                'num_features' : [512, 384, action_dim * n_classes_wm],
                'nonlinearities' : ['relu', 'relu', 'identity'],
                'dropout' : [.5, .5, None]
        },

        {
                'num_features' : [50, action_dim * n_classes_wm],
                'nonlinearities' : ['relu', 'identity'],
                'dropout' : [.5, None]
        },]

#various fully-connected choices
mlp_choices = [



        {
        'num_features' : [512],
        'nonlinearities' : ['relu'],
        'dropout' : [.5]
        },

	{
	'num_features' : [192],
	'nonlinearities' : ['relu'],
	'dropout' : [.5]
        },

	{
	'num_features' : [384],
	'nonlinearities' : ['relu'],
	'dropout' : [.5]
        },

        {
        'num_features' : [512, 384],
        'nonlinearities' : ['relu', 'relu'],
        'dropout' : [.5, .5]
        },

        {
        'num_features' : [512, 512, 384],
        'nonlinearities' : ['relu', 'relu', 'relu'],
        'dropout' : [.5, .5, .5]
        },

        {
        'num_features' : [100],
        'nonlinearities' : ['relu'],
        'dropout' : [.5]
        },

]


def get_sm_readout_cfg(n_classes_sm, n_timesteps):
    separate_mlp_choices_proto = {
		'num_features' : [n_classes_sm],
		'nonlinearities' : ['identity'],
		'dropout' : [None]
	}
    return dict((t, separate_mlp_choices_proto) for t in range(n_timesteps))



