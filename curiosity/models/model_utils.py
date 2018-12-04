'''
Extending tfutils.model Convnet class to have some additional moves.
'''
import sys
sys.path.append('tfutils')
import tensorflow as tf
from collections import OrderedDict

import distutils.version
use_tf1 = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')
import numpy as np


def bin_values(values, thresholds):
	for i, th in enumerate(thresholds):
		if i == 0:
			lab = tf.cast(tf.greater(values, th), tf.int32)
		else:
			lab += tf.cast(tf.greater(values, th), tf.int32)
	return lab

def binned_softmax_loss(tv, prediction, cfg):
	thresholds = cfg['thresholds']
	tv = bin_values(tv, thresholds)
	tv = tf.squeeze(tv)
	loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tv, logits = prediction)
	loss = tf.reduce_mean(loss_per_example) * cfg.get('loss_factor', 1.)
	return loss_per_example, loss

def binned_01_accuracy_per_example(tv, prediction, cfg):
    thresholds = cfg['thresholds']
    n_classes = len(thresholds) + 1
    tv_shape = tv.get_shape().as_list()
    d = tv_shape[1]
    assert(len(tv_shape)) == 2
    tv = bin_values(tv, thresholds)
    prediction = tf.reshape(prediction, [-1, d, n_classes])
    hardmax = tf.cast(tf.argmax(prediction, axis = -1), tf.int32)
    correct_answers = tf.cast(tf.equal(hardmax, tv), tf.int32)
    return correct_answers


def binned_softmax_loss_per_example(tv, prediction, cfg):
	thresholds = cfg['thresholds']
	n_classes = len(thresholds) + 1
	tv_shape = tv.get_shape().as_list()
	d = tv_shape[1]
	assert len(tv_shape) == 2
	tv = bin_values(tv, thresholds)
	prediction = tf.reshape(prediction, [-1, d, n_classes])
	loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tv, logits = prediction) *  cfg.get('loss_factor', 1.)
	loss_per_example = tf.reduce_mean(loss_per_example, axis = 1, keep_dims = True)
	loss = tf.reduce_mean(loss_per_example)
	return loss_per_example, loss


def ms_sum_binned_softmax_loss(tv, prediction, cfg):
    assert len(tv) == len(prediction)
    loss_per_example_and_step = [binned_softmax_loss(y, p, cfg) for y, p in zip(tv, prediction)]
    loss_per_example = [lpe for lpe, lps in loss_per_example_and_step]
    loss_per_step = [lps for lpe, lps in loss_per_example_and_step]
    loss = tf.reduce_mean(loss_per_step)
    return loss_per_example, loss_per_step, loss


def tf_concat(list_of_tensors, axis = 0):
    if use_tf1:
        return tf.concat(list_of_tensors, axis)
    return tf.concat(axis, list_of_tensors)



from my_model import ConvNet

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def postprocess_std(in_node):
	in_node = tf.cast(in_node, tf.float32)
	in_node = in_node / 255.
	return in_node

def hidden_loop_with_bypasses(input_node, m, cfg, nodes_for_bypass = [], stddev = .01, reuse_weights = False, activation = 'relu', train_indicator = 0.):
        assert len(input_node.get_shape().as_list()) == 2, len(input_node.get_shape().as_list())
        hidden_depth = cfg['hidden_depth']
        m.output = input_node
        for i in range(1, hidden_depth + 1):
                with tf.variable_scope('hidden' + str(i)) as scope:
                        if reuse_weights:
                                scope.reuse_variables()
                        bypass = cfg['hidden'][i].get('bypass')
                        if bypass:
                                bypass_node = nodes_for_bypass[bypass]
                                m.add_bypass(bypass_node)
                        nf = cfg['hidden'][i]['num_features']
                        my_activation = cfg['hidden'][i].get('activation')
                        if my_activation is None:
                                my_activation = activation
                        my_dropout = cfg['hidden'][i].get('dropout')
                        if my_dropout is not None:
                            my_dropout = 1. - (1. - my_dropout) * train_indicator
                        m.fc(nf, init = 'xavier', activation = my_activation, bias = .01, stddev = stddev, dropout = my_dropout)
                        nodes_for_bypass.append(m.output)
	return m.output

def categorical_sample(logits, d, one_hot = True):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    if not one_hot:
    	return value
    return tf.one_hot(value, d)

def feedforward_conv_loop(input_node, m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print=True, return_bypass=False, sub_bypass = None):
        m.output = input_node
        encode_nodes = [input_node]
        #encoding
        encode_depth = cfg[desc + '_depth']
        cfs0 = None

        if bypass_nodes is None:
                bypass_nodes = [m.output]

        for i in range(1, encode_depth + 1):
        #not sure this usage ConvNet class creates exactly the params that we want to have, specifically in the 'input' field, but should give us an accurate record of this network's configuration
                with tf.variable_scope(desc + str(i)) as scope:
                        if reuse_weights:
                                scope.reuse_variables()

                        bypass = cfg[desc][i].get('bypass')
                        if bypass:
                                if type(bypass) == list:
                                        bypass_node = [bypass_nodes[bp] for bp in bypass]
                                elif type(bypass) == dict:
                                    if sub_bypass is None:
                                        raise ValueError('Bypass \
                                                is dict but no sub_bypass specified')
                                    for k in bypass:
                                        if int(k) == sub_bypass:
                                            if type(bypass[k]) == list:
                                                bypass_node = [bypass_nodes[bp] \
                                                        for bp in bypass[k]]
                                            else:
                                                bypass_node = bypass_nodes[bypass[k]]
                                else:
                                        bypass_node = bypass_nodes[bypass]
                                m.add_bypass(bypass_node)

                        bn = cfg[desc][i]['conv'].get('batch_normalize')
                        if bn:  
                                norm_it = bn
                        else:   
                                norm_it = batch_normalize



                        with tf.contrib.framework.arg_scope([m.conv], init='xavier', stddev=.01, bias=0, batch_normalize = norm_it):
                            cfs = cfg[desc][i]['conv']['filter_size']
                            cfs0 = cfs
                            nf = cfg[desc][i]['conv']['num_filters']
                            cs = cfg[desc][i]['conv']['stride']
                            if no_nonlinearity_end and i == encode_depth:
                                m.conv(nf, cfs, cs, activation = None)
                            else:
                                my_activation = cfg[desc][i].get('nonlinearity')
                                if my_activation is None:
                                        my_activation = 'relu'
                                m.conv(nf, cfs, cs, activation = my_activation)
       #TODO add print function
                        pool = cfg[desc][i].get('pool')
                        if pool:
                            pfs = pool['size']
                            ps = pool['stride']
                            m.pool(pfs, ps)
                        encode_nodes.append(m.output)
                        bypass_nodes.append(m.output)
        if return_bypass:
            return [encode_nodes, bypass_nodes]
        return encode_nodes

def deconv_loop(input_node, m, cfg, desc = 'deconv', bypass_nodes = None,
        reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print = True, return_bypass=False, sub_bypass = None):
    m.output = input_node
    deconv_nodes = [input_node]
    # deconvolving
    deconv_depth = cfg[desc + '_depth']
    cfs0 = None

    if bypass_nodes is None:
        bypass_nodes = [m.output]

    for i in range(1, deconv_depth + 1):
        with tf.variable_scope(desc + str(i)) as scope:
            if reuse_weights:
                scope.reuse_variables()

            bypass = cfg[desc][i].get('bypass')
            if bypass is not None:
                if type(bypass) == list:
                    bypass_node = [bypass_nodes[bp] for bp in bypass]
                elif type(bypass) == dict:
                    if sub_bypass is None:
                       raise ValueError('Bypass \
                               is dict but no sub_bypass specified')
                    for k in bypass:
                        if int(k) == sub_bypass:
                            if type(bypass[k]) == list:
                                bypass_node = [bypass_nodes[bp] \
                                        for bp in bypass[k]]
                            else:
                                bypass_node = bypass_nodes[bypass[k]]
                else:
                    bypass_node = bypass_nodes[bypass]
                m.add_bypass(bypass_node)

            bn = cfg[desc][i]['deconv'].get('batch_normalize')
            if bn:
                norm_it = bn
            else:
                norm_it = batch_normalize

            with tf.contrib.framework.arg_scope([m.deconv], 
                    init='xavier', stddev=.01, bias=0, batch_normalize = norm_it):
                cfs = cfg[desc][i]['deconv']['filter_size']
                cfs0 = cfs
                nf = cfg[desc][i]['deconv']['num_filters']
                cs = cfg[desc][i]['deconv']['stride']
                if 'output_shape' in cfg[desc][i]['deconv']:
                    out_shape = cfg[desc][i]['deconv']['output_shape']
                else:
                    out_shape = None
                if no_nonlinearity_end and i == deconv_depth:
                    m.deconv(nf, cfs, cs, activation = None, 
                            fixed_output_shape=out_shape)
                else:
                    my_activation = cfg[desc][i].get('nonlinearity')
                    if my_activation is None:
                        my_activation = 'relu'
                    m.deconv(nf, cfs, cs, activation = my_activation, 
                            fixed_output_shape=out_shape)
                    if do_print:
                        print('deconv out:', m.output)
                    #TODO add print function
                    pool = cfg[desc][i].get('pool')
                    if pool:
                        pfs = pool['size']
                        ps = pool['stride']
                        m.pool(pfs, ps)
                    deconv_nodes.append(m.output)
                    bypass_nodes.append(m.output)
    if return_bypass:
        return [deconv_nodes, bypass_nodes]
    return deconv_nodes


class ConvNetwithBypasses(ConvNet):
	'''Right now just borrowing from chengxuz's contribution...will edit if modified
	See https://github.com/neuroailab/barrel/blob/master/normals_relat/normal_pred/normal_encoder_asymmetric_with_bypass.py'''
	def __init__(self, seed=None, **kwargs):
            super(ConvNetwithBypasses, self).__init__(seed=seed, **kwargs)

	@property
	def params(self):
	    return self._params

	@params.setter
	def params(self, value):
		'''Modified from parent to allow for multiple calls of the same type within a scope name.
		This should not happen unless we are pushing more than one node through the same graph, this keeps a record of that.'''
		name = tf.get_variable_scope().name
		if name not in self._params:
		    self._params[name] = OrderedDict()
		if value['type'] in self._params[name]:
			self._params[name][value['type']]['input'] = self._params[name][value['type']]['input'] + ',' + value['input']
		else:
			self._params[name][value['type']] = value


	@tf.contrib.framework.add_arg_scope
	def conv_given_filters(self, kernel, biases, stride = 1, padding = 'SAME', activation = 'relu', batch_normalize = False, in_layer = None):
		if in_layer is None:
			in_layer = self.output
		k_shape = kernel.get_shape().as_list()
		out_shape = k_shape[3]
		ksize1 = k_shape[0]
		ksize2 = k_shape[1]
		conv = tf.nn.conv2d(in_layer, kernel,
		                    strides=[1, stride, stride, 1],
		                    padding=padding)
		
		if batch_normalize:
			#Using "global normalization," which is recommended in the original paper
			mean, var = tf.nn.moments(conv, [0, 1, 2])
			scale = tf.get_variable(initializer=tf.constant_initializer(bias),
			                         shape=[out_shape],
			                         dtype=tf.float32,
			                         name='scale')
			self.output = tf.nn.batch_normalization(conv, mean, var, biases, scale, 1e-3, name = 'conv')
		else:
			self.output = tf.nn.bias_add(conv, biases, name='conv')
		
		if activation is not None:
		    self.output = self.activation(kind=activation)
		
		self.params = {'input': in_layer.name,
		               'type': 'conv',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2),
		               'padding': padding,
		               'init': init,
		               'activation': activation}
		return self.output

	def activation(self, kind='relu', in_layer=None):
		if in_layer is None:
			in_layer = self.output
		last_axis = len(in_layer.get_shape().as_list()) - 1
		if type(kind) != list:
			kind = [kind]
		for_out = []
		for k in kind:
			if k == 'relu':
				for_out.append(tf.nn.relu(in_layer, name='relu'))
                        elif k == 'crelu':
                                for_out.append(tf.nn.crelu(in_layer, name='crelu'))
			elif k == 'tanh':
				for_out.append(tf.tanh(in_layer, name = 'tanh'))
			elif k == 'concat_square':
				for_out.append(tf_concat([in_layer, in_layer * in_layer], last_axis))
			elif k == 'square':
				for_out.append(in_layer * in_layer)
			elif k == 'safe_square':
				my_tanh = tf.tanh(in_layer)
				for_out.append(my_tanh * my_tanh)
			elif k == 'neg_relu':
				for_out.append(tf.nn.relu(-in_layer, name = 'neg_relu'))
			elif k == 'square_relu':
				rel_inlayer = tf.nn.relu(rel_inlayer)
				for_out.append(rel_inlayer)
			elif k == 'square_relu_neg':
				rel_in = tf.nn.relu(-inlayer)
				for_out.append(rel_in * rel_in)
			elif k == 'square_crelu':
				crel = tf.nn.crelu(in_layer, name = 'crelu')
				for_out.append(crel * crel)
			elif k == 'identity':
				for_out.append(in_layer)
			else:
				raise ValueError("Activation '{}' not defined".format(k))
		self.output = tf_concat(for_out, last_axis)
		return self.output

        @tf.contrib.framework.add_arg_scope
        def fc(self,
               out_shape,
               init='xavier',
               stddev=.01,
               bias=1,
               activation='relu',
               dropout=.5,
               in_layer=None,
               init_file=None,
               init_layer_keys=None,
               trainable=True):

            if in_layer is None:
                in_layer = self.output
            #let's assume things are flattened, ok?
            # resh = tf.reshape(in_layer,
            #                   [in_layer.get_shape().as_list()[0], -1],
            #                   name='reshape')
            resh = in_layer
            in_shape = resh.get_shape().as_list()[-1]
            if init != 'from_file':
                kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                         shape=[in_shape, out_shape],
                                         dtype=tf.float32,
                                         name='weights',
                                         trainable=trainable)
                biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                         shape=[out_shape],
                                         dtype=tf.float32,
                                         name='bias',
                                         trainable=trainable)
            else:
                init_dict = self.initializer(init,
                                             init_file=init_file,
                                             init_keys=init_layer_keys)
                kernel = tf.get_variable(initializer=init_dict['weight'],
                                         dtype=tf.float32,
                                         name='weights',
                                         trainable=trainable)
                biases = tf.get_variable(initializer=init_dict['bias'],
                                         dtype=tf.float32,
                                         name='bias',
                                         trainable=trainable)

            fcm = tf.matmul(resh, kernel)
            self.output = tf.nn.bias_add(fcm, biases, name='fc')
            if activation is not None:
                self.activation(kind=activation)
            if dropout is not None:
                self.output = tf.nn.dropout(self.output, dropout, seed = self.seed, name = 'dropout') 

            self.params = {'input': in_layer.name,
                           'type': 'fc',
                           'num_filters': out_shape,
                           'init': init,
                           'bias': bias,
                           'stddev': stddev,
                           'activation': activation,
                           'dropout': dropout,
                           'seed': self.seed}
            return self.output

	@tf.contrib.framework.add_arg_scope
	def coord_to_conv(self,
			out_shape,
			in_layer,
			ksize = 3,
			stride = 1,
			padding = 'SAME',
			init = 'xavier',
			stddev = .01,
			bias = 1,
			activation = 'relu',
			weight_decay = None,
			trainable = True
					):
		if weight_decay is None:
			weight_decay = 0.
		in_shape = in_layer.get_shape().as_list()
		assert len(in_shape) == 2
		batch_size = in_shape[0]

		if isinstance(ksize, int):
		    ksize1 = ksize
		    ksize2 = ksize
		else:
		    ksize1, ksize2 = ksize

		out_width = out_shape[1]
		out_height = out_shape[0]
		out_channels = out_shape[2]

		coord_kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
										 shape=[ksize1, ksize2, 2, out_channels],
										 dtype=tf.float32,
										 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
										 name='weights', trainable=trainable)
		biases = tf.get_variable(initializer=tf.constant_initializer(bias),
		                         shape=[out_channels],
		                         dtype=tf.float32,
		                         name='bias', trainable=trainable)

	   	input_kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
	   							shape = [in_shape[1], out_channels],
	   							dtype = tf.float32,
	                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
	                             name='inweights', trainable=trainable)

		X = tf.range(out_width)
		X = tf.expand_dims(X, 0)
		X = tf.expand_dims(X, 1)
		X = tf.expand_dims(X, 3)
		X = tf.tile(X, [batch_size, out_height, 1, 1])


		Y = tf.range(out_height)
		Y = tf.expand_dims(Y, 0)
		Y = tf.expand_dims(Y, 2)
		Y = tf.expand_dims(Y, 3)
		Y = tf.tile(Y, [batch_size, 1, out_width, 1])

		coord = tf_concat([Y, X], 3)
		coord = tf.cast(coord, tf.float32)

		coord_conv = tf.nn.conv2d(coord, coord_kernel, strides = [1, stride, stride, 1], padding = padding)
		input_mul = tf.matmul(in_layer, input_kernel)
		input_mul = tf.expand_dims(input_mul, 1)
		input_mul = tf.expand_dims(input_mul, 1)
		adding = coord_conv + input_mul
		self.output = tf.nn.bias_add(adding, biases, name='coordtoconv')

		if activation is not None:
		    self.output = self.activation(kind=activation)
		self.params = {'input': in_layer.name,
		               'type': 'coordtoconv',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2),
		               'padding': padding,
		               'init': init,
		               'stddev': stddev,
		               'bias': bias,
		               'activation': activation,
		               'weight_decay': weight_decay,
		               'seed': self.seed}
		return self.output

        @tf.contrib.framework.add_arg_scope
        def deconv3d(self,
                 out_shape,
                 ksize=3,
                 stride=1,
                 padding='SAME',
                 init='xavier',
                 stddev=.01,
                 bias=1,
		 fixed_output_shape=None,
                 activation='relu',
                 weight_decay=None,
                 in_layer=None,
                 init_file=None,
                 init_layer_keys=None,
                 batch_normalize=False,
                 trainable=True,
                   ):
                if in_layer is None:
                    in_layer = self.output
                if weight_decay is None:
                    weight_decay = 0.
                in_shape = in_layer.get_shape().as_list()[-1]

                if isinstance(ksize, int):
                    ksize1 = ksize
                    ksize2 = ksize
                    ksize3 = ksize
                else:
                    ksize1, ksize2, ksize3 = ksize

                if init != 'from_file':
                    kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                             shape=[ksize1, ksize2, ksize3, out_shape, in_shape],
                                             dtype=tf.float32,
                                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                             name='weights', trainable=trainable)
                    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                             shape=[out_shape],
                                             dtype=tf.float32,
                                             name='bias', trainable=trainable)
                else:
                    init_dict = self.initializer(init,
                                                 init_file=init_file,
                                                 init_keys=init_layer_keys)
                    kernel = tf.get_variable(initializer=init_dict['weight'],
                                             dtype=tf.float32,
                                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                             name='weights', trainable=trainable)
                    biases = tf.get_variable(initializer=init_dict['bias'],
                                             dtype=tf.float32,
                                             name='bias', trainable=trainable)
               
		if fixed_output_shape is None:
			in_shape = in_layer.get_shape().as_list()
			fixed_output_shape = [in_shape[0], \
				in_shape[1] * stride, in_shape[2] * stride, \
                                in_shape[3] * stride, out_shape]
                deconv = tf.nn.conv3d_transpose(in_layer, kernel, fixed_output_shape,
                                    strides=[1, stride, stride, stride, 1],
                                    padding=padding)

                if batch_normalize:
                        #Using "global normalization," which is recommended in the original paper
                        mean, var = tf.nn.moments(deconv, [0, 1, 2, 3])
                        scale = tf.get_variable(initializer=tf.constant_initializer(bias),
                                                 shape=[out_shape],
                                                 dtype=tf.float32,
                                                 name='scale', trainable=trainable)
                        self.output = tf.nn.batch_normalization(deconv, mean, var, biases, scale, 1e-3, name = 'deconv')
                else:
                        self.output = tf.nn.bias_add(deconv, biases, name='deconv')

                if activation is not None:
                    self.output = self.activation(kind=activation)
                self.params = {'input': in_layer.name,
                               'type': 'deconv3d',
                               'num_filters': out_shape,
                               'stride': stride,
                               'kernel_size': (ksize1, ksize2, ksize3),
                               'padding': padding,
                               'init': init,
                               'stddev': stddev,
                               'bias': bias,
                               'activation': activation,
                               'weight_decay': weight_decay,
                               'seed': self.seed}
                return self.output


        @tf.contrib.framework.add_arg_scope
        def deconv(self,
                 out_shape,
                 ksize=3,
                 stride=1,
                 padding='SAME',
                 init='xavier',
                 stddev=.01,
                 bias=1,
		 fixed_output_shape=None,
                 activation='relu',
                 weight_decay=None,
                 in_layer=None,
                 init_file=None,
                 init_layer_keys=None,
                 batch_normalize=False,
                 group=None,
                 trainable=True,
                   ):
                if in_layer is None:
                    in_layer = self.output
                if weight_decay is None:
                    weight_decay = 0.
                in_shape = in_layer.get_shape().as_list()[-1]

                if isinstance(ksize, int):
                    ksize1 = ksize
                    ksize2 = ksize
                else:
                    ksize1, ksize2 = ksize

                if init != 'from_file':
                    kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                             shape=[ksize1, ksize2, out_shape, in_shape],
                                             dtype=tf.float32,
                                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                             name='weights', trainable=trainable)
                    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                             shape=[out_shape],
                                             dtype=tf.float32,
                                             name='bias', trainable=trainable)
                else:
                    init_dict = self.initializer(init,
                                                 init_file=init_file,
                                                 init_keys=init_layer_keys)
                    kernel = tf.get_variable(initializer=init_dict['weight'],
                                             dtype=tf.float32,
                                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                             name='weights', trainable=trainable)
                    biases = tf.get_variable(initializer=init_dict['bias'],
                                             dtype=tf.float32,
                                             name='bias', trainable=trainable)
               
		if fixed_output_shape is None:
			in_shape = tf.shape(in_layer) #.get_shape().as_list()
			fixed_output_shape = [in_shape[0], \
                                in_shape[1] * stride, \
                                in_shape[2] * stride, out_shape]
                deconv = tf.nn.conv2d_transpose(in_layer, kernel,\
                        fixed_output_shape, strides=[1, stride, stride, 1],
                        padding=padding)

                if batch_normalize:
                        #Using "global normalization," which is recommended in the original paper
                        mean, var = tf.nn.moments(deconv, [0, 1, 2])
                        scale = tf.get_variable(initializer=tf.constant_initializer(bias),
                                                 shape=[out_shape],
                                                 dtype=tf.float32,
                                                 name='scale', trainable=trainable)
                        self.output = tf.nn.batch_normalization(deconv, mean, var, biases, scale, 1e-3, name = 'deconv')
                else:
                        self.output = tf.nn.bias_add(deconv, biases, name='deconv')

                if activation is not None:
                    self.output = self.activation(kind=activation)
                self.params = {'input': in_layer.name,
                               'type': 'deconv',
                               'num_filters': out_shape,
                               'stride': stride,
                               'kernel_size': (ksize1, ksize2),
                               'padding': padding,
                               'init': init,
                               'stddev': stddev,
                               'bias': bias,
                               'activation': activation,
                               'weight_decay': weight_decay,
                               'seed': self.seed}
                return self.output


        @tf.contrib.framework.add_arg_scope
        def conv3d(self,
                 out_shape,
                 ksize=3,
                 stride=1,
                 padding='SAME',
                 init='xavier',
                 stddev=.01,
                 bias=1,
                 activation='relu',
                 weight_decay=None,
                 in_layer=None,
                 init_file=None,
                 init_layer_keys=None,
                 batch_normalize=False,
                 trainable=True,
                 ):
                if in_layer is None:
                    in_layer = self.output
                if weight_decay is None:
                    weight_decay = 0.
                in_shape = in_layer.get_shape().as_list()[-1]

                if isinstance(ksize, int):
                    ksize1 = ksize
                    ksize2 = ksize
                    ksize3 = ksize
                else:
                    ksize1, ksize2, ksize3 = ksize

		if init != 'from_file':
		    kernel = tf.get_variable(initializer=self.initializer(init, \
                                                 stddev=stddev),
		                             shape=[ksize1, ksize2, ksize3, \
                                                 in_shape, out_shape],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
		                             shape=[out_shape],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)
		else:
		    init_dict = self.initializer(init,
		                                 init_file=init_file,
		                                 init_keys=init_layer_keys)
		    kernel = tf.get_variable(initializer=init_dict['weight'],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=init_dict['bias'],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)

		conv = tf.nn.conv3d(in_layer, kernel,
		                    strides=[1, stride, stride, stride, 1],
		                    padding=padding)

		if batch_normalize:
			#Using "global normalization," which is recommended in the original paper
			mean, var = tf.nn.moments(conv, [0, 1, 2, 3])
			scale = tf.get_variable(initializer=tf.constant_initializer(bias),
			                         shape=[out_shape],
			                         dtype=tf.float32,
			                         name='scale', trainable=trainable)
			self.output = tf.nn.batch_normalization(conv, mean, var, biases, scale, 1e-3, name = 'conv')
		else:
			self.output = tf.nn.bias_add(conv, biases, name='conv')

		if activation is not None:
		    self.output = self.activation(kind=activation)
		self.params = {'input': in_layer.name,
		               'type': 'conv3d',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2, ksize3),
		               'padding': padding,
		               'init': init,
		               'stddev': stddev,
		               'bias': bias,
		               'activation': activation,
		               'weight_decay': weight_decay,
		               'seed': self.seed}
		return self.output


	@tf.contrib.framework.add_arg_scope
	def conv(self, 
                 out_shape, 
                 ksize=3, 
                 stride=1, 
                 padding='SAME', 
                 init='xavier', 
                 stddev=.01, 
                 bias=1, 
                 activation='relu', 
                 weight_decay=None, 
                 in_layer=None, 
                 init_file=None, 
                 init_layer_keys=None, 
                 batch_normalize=False,
                 group=None,
                 trainable=True,
                   ):
		if in_layer is None:
		    in_layer = self.output
		if weight_decay is None:
		    weight_decay = 0.
		in_shape = in_layer.get_shape().as_list()[-1]

		if isinstance(ksize, int):
		    ksize1 = ksize
		    ksize2 = ksize
		else:
		    ksize1, ksize2 = ksize

		if init != 'from_file':
		    kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
		                             shape=[ksize1, ksize2, in_shape, out_shape],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
		                             shape=[out_shape],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)
		else:
		    init_dict = self.initializer(init,
		                                 init_file=init_file,
		                                 init_keys=init_layer_keys)
		    kernel = tf.get_variable(initializer=init_dict['weight'],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=init_dict['bias'],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)

                if group is None or group == 1:
		    conv = tf.nn.conv2d(in_layer, kernel,
		                    strides=[1, stride, stride, 1],
		                    padding=padding)
                else:
                    assert in_layer.get_shape()[-1] % group == 0
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride, stride, 1],\
                                                         padding=padding)
                    in_layers = tf.split(3, group, in_layer)
                    kernels = tf.split(3, group, kernel)
                    convs = [convolve(i, k) for i,k in zip(in_layers, kernels)]
                    conv = tf_concat(convs, 3)

		if batch_normalize:
			#Using "global normalization," which is recommended in the original paper
			mean, var = tf.nn.moments(conv, [0, 1, 2])
			scale = tf.get_variable(initializer=tf.constant_initializer(bias),
			                         shape=[out_shape],
			                         dtype=tf.float32,
			                         name='scale', trainable=trainable)
			self.output = tf.nn.batch_normalization(conv, mean, var, biases, scale, 1e-3, name = 'conv')
		else:
			self.output = tf.nn.bias_add(conv, biases, name='conv')

		if activation is not None:
		    self.output = self.activation(kind=activation)
		self.params = {'input': in_layer.name,
		               'type': 'conv',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2),
		               'padding': padding,
		               'init': init,
		               'stddev': stddev,
		               'bias': bias,
		               'activation': activation,
		               'weight_decay': weight_decay,
		               'seed': self.seed}
		return self.output

	@tf.contrib.framework.add_arg_scope
	def pool3d(self,
	         ksize=3,
	         stride=2,
	         padding='SAME',
	         in_layer=None,
	         pfunc='maxpool'):
	    if in_layer is None:
	        in_layer = self.output

	    if isinstance(ksize, int):
	        ksize1 = ksize
	        ksize2 = ksize
                ksize3 = ksize
	    else:
	        ksize1, ksize2, ksize3 = ksize

	    if pfunc=='maxpool':
	        self.output = tf.nn.max_pool3d(in_layer,
	                                     ksize=[1, ksize1, ksize2, ksize3, 1],
	                                     strides=[1, stride, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    else:
	        self.output = tf.nn.avg_pool(in_layer,
	                                     ksize=[1, ksize1, ksize2, ksize3, 1],
	                                     strides=[1, stride, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    self.params = {'input': in_layer.name,
	                   'type': pfunc,
	                   'kernel_size': (ksize1, ksize2, ksize3),
	                   'stride': stride,
	                   'padding': padding}
	    return self.output


	@tf.contrib.framework.add_arg_scope
	def pool(self,
	         ksize=3,
	         stride=2,
	         padding='SAME',
	         in_layer=None,
	         pfunc='maxpool'):
	    if in_layer is None:
	        in_layer = self.output

	    if isinstance(ksize, int):
	        ksize1 = ksize
	        ksize2 = ksize
	    else:
	        ksize1, ksize2 = ksize

	    if pfunc=='maxpool':
	        self.output = tf.nn.max_pool(in_layer,
	                                     ksize=[1, ksize1, ksize2, 1],
	                                     strides=[1, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    else:
	        self.output = tf.nn.avg_pool(in_layer,
	                                     ksize=[1, ksize1, ksize2, 1],
	                                     strides=[1, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    self.params = {'input': in_layer.name,
	                   'type': pfunc,
	                   'kernel_size': (ksize1, ksize2),
	                   'stride': stride,
	                   'padding': padding}
	    return self.output

	def reshape(self, new_size, in_layer=None):
		#TODO: add params update
	    if in_layer is None:
	        in_layer = self.output

	    size_l = [in_layer.get_shape().as_list()[0]]
	    size_l.extend(new_size)
	    self.output = tf.reshape(in_layer, size_l)
	    self.params = {'input' : in_layer.name, 'type' : 'reshape', 'new_shape' : size_l}
	    return self.output

	def resize_images(self, new_size, in_layer=None):
		#TODO: add params update
	    if in_layer is None:
	        in_layer = self.output
	    if not type(new_size) == list:
	    	assert type(new_size) == int
	    	new_size = [new_size, new_size]
	    self.output = tf.image.resize_images(in_layer, new_size)
	    self.params = {'input' : in_layer.name, 'type' : 'resize', 'new_size' : new_size}
	    return self.output

	def minmax(self, min_arg = 'inf', max_arg = 'ninf', in_layer = None):
		'''Note that this does nothing, silently, except modify params, if this is called with default arguments.
		'''
		if in_layer is None:
			in_layer = self.output
		self.params = {'input' : in_layer.name, 'type' : 'minmax', 'min_arg' : min_arg, 'max_arg' : max_arg}
		if max_arg != 'ninf':
			in_layer = tf.maximum(in_layer, max_arg)
		if min_arg != 'inf':
			in_layer = tf.minimum(in_layer, min_arg)
		self.output = in_layer
		return self.output


	def add_bypass(self, bypass_layers, in_layer=None):
	    if in_layer is None:
	        in_layer = self.output

	    if not isinstance(bypass_layers, list):
	    	bypass_layers = [bypass_layers]
	    in_shape = tf.shape(in_layer) 
            in_rank = len(in_layer.get_shape().as_list())
	    toconcat = [in_layer]
	    concat_type = None
	    if in_rank == 4:	
	    	ds1 = in_shape[1]
	    	ds2 = in_shape[2]
	    	for layer in bypass_layers:
	    		if layer.get_shape().as_list()[1] != ds1 \
                                or layer.get_shape().as_list()[2] != ds2:
                                    toconcat.append( \
                                            self.resize_images( \
                                            [ds1, ds2], in_layer = layer))
	    		else:
	    			toconcat.append(layer)
	    	self.output = tf_concat(toconcat, 3)
	    	concat_type = 'image'
	    elif in_rank == 2:
	    	toconcat.extend(bypass_layers)
	    	self.output = tf_concat(toconcat, 1)
	    	concat_type = 'flat'
            elif in_rank == 5:
                ds1 = in_shape[1]
                ds2 = in_shape[2]
                ds3 = in_shape[3]
                for layer in bypass_layers:
                    ls = layer.get_shape().as_list()
                    if ls[1] != ds1 or ls[2] != ds2 or ls[3] != ds3:
                        # resize zx plane first, then xy plane
                        layer = tf.reshape(layer, list(ls[:-2]) + list(ls[-2]*ls[-1]))
                        layer = self.resize_images([ds1, ds2], in_layer = layer)
                        layer = tf.reshape(layer, [ls[0] * ds1, ds2, ls[-2], ls[-1]])
                        layer = self.resize_images([ds2, ds3], in_layer = layer)
                        layer = tf.reshape(layer, [ls[0], ds1, ds2, ds3, ls[-1]])
                    toconcat.append(layer)
                self.output = tf.concat(toconcat, 4)
                concat_type = '3d_volume'
	    else:
	    	raise Exception('Bypass case not yet handled.')
	    self.params = {'input' : in_layer.name, 'type' : 'bypass', 'bypass_names' : [l.name for l in toconcat[1:]], 'concat_type' : concat_type}
	    return self.output



