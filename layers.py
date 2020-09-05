from __future__ import division
import paddle.fluid as fluid
import numpy as np
import os
import warnings
import math

# cudnn is not better when batch size is 1.
use_cudnn_conv2d_transpose = False
use_cudnn_conv2d = True
use_layer_norm = True

def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    valid_filter_size = dilation * (filter_size - 1) + 1
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2


def instance_norm(input, name=None):
    if use_layer_norm:
        return fluid.layers.layer_norm(input, begin_norm_axis=2) 

    helper = fluid.layer_helper.LayerHelper("instance_norm", **locals())
    dtype = helper.input_dtype()
    epsilon = 1e-5
    mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
    var = fluid.layers.reduce_mean(
        fluid.layers.square(input - mean), dim=[2, 3], keep_dim=True)
    if name is not None:
        scale_name = name + "_scale"
        offset_name = name + "_offset"
    scale_param = fluid.ParamAttr(
        name=scale_name,
        initializer=fluid.initializer.TruncatedNormal(1.0, 0.02),
        trainable=True)
    offset_param = fluid.ParamAttr(
        name=offset_name,
        initializer=fluid.initializer.Constant(0.0),
        trainable=True)
    scale = helper.create_parameter(
        attr=scale_param, shape=input.shape[1:2], dtype=dtype)
    offset = helper.create_parameter(
        attr=offset_param, shape=input.shape[1:2], dtype=dtype)

    tmp = fluid.layers.elementwise_mul(x=(input - mean), y=scale, axis=1)
    tmp = tmp / fluid.layers.sqrt(var + epsilon)
    tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
    return tmp

def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding="VALID",
           name="conv2d",
           norm=True,
           relu=True,
           relufactor=0.0,
           bias_attr_bool=False):

    need_crop = False
    if padding == "SAME":
        top_padding, bottom_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        left_padding, right_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
    else:
        height_padding = 0
        width_padding = 0

    padding = [height_padding, width_padding]
    param_attr = fluid.ParamAttr(
        name=name + "_w",
        initializer=fluid.initializer.TruncatedNormal(scale=stddev))
    if bias_attr_bool:
        bias_attr = fluid.ParamAttr(
            name=name + "_b", initializer=fluid.initializer.Constant(0.0))
    else:
        bias_attr = False
    
    conv = fluid.layers.conv2d(
        input,
        num_filters,
        filter_size,
        name=name,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn_conv2d,
        param_attr=param_attr,
        bias_attr=bias_attr)
    if need_crop:
        conv = fluid.layers.crop(
            conv,
            shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
            offsets=(0, 0, 1, 1))
    if norm:
        conv = instance_norm(input=conv, name=name + "_norm")
    if relu:
        conv = fluid.layers.leaky_relu(conv, alpha=relufactor)
    return conv

def norm_layer(input,
               norm_type='batch_norm',
               name=None,
               is_test=False,
               affine=True):
    if norm_type == 'batch_norm':
        if affine == True:
            param_attr = fluid.ParamAttr(
                name=name + '_w',
                initializer=fluid.initializer.Normal(
                    loc=1.0, scale=0.02))
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Constant(value=0.0))
        else:
            param_attr = fluid.ParamAttr(
                name=name + '_w',
                initializer=fluid.initializer.Constant(1.0),
                trainable=False)
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Constant(value=0.0),
                trainable=False)
        return fluid.layers.batch_norm(
            input,
            param_attr=param_attr,
            bias_attr=bias_attr,
            is_test=is_test,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_var')

    elif norm_type == 'instance_norm':
        if name is not None:
            scale_name = name + "_scale"
            offset_name = name + "_offset"
        if affine:
            scale_param = fluid.ParamAttr(
                name=scale_name,
                initializer=fluid.initializer.Constant(1.0),
                trainable=True)
            offset_param = fluid.ParamAttr(
                name=offset_name,
                initializer=fluid.initializer.Constant(0.0),
                trainable=True)
        else:
            scale_param = fluid.ParamAttr(
                name=scale_name,
                initializer=fluid.initializer.Constant(1.0),
                trainable=False)
            offset_param = fluid.ParamAttr(
                name=offset_name,
                initializer=fluid.initializer.Constant(0.0),
                trainable=False)
        return fluid.layers.instance_norm(
            input, param_attr=scale_param, bias_attr=offset_param)
    else:
        raise NotImplementedError("norm type: [%s] is not support" % norm_type)

def conv2d_with_filter(input,
                       filter,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=None,
                       bias_attr=None,
                       use_cudnn=True,
                       act=None,
                       name=None):
    helper = fluid.layer_helper.LayerHelper("conv2d_with_filter", **locals())
    num_channels = input.shape[1]
    num_filters = filter.shape[0]
    num_filter_channels = filter.shape[1]
    l_type = 'conv2d'
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'
    if groups is None:
        assert num_filter_channels == num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        if num_channels // groups != num_filter_channels:
            raise ValueError("num_filter_channels must equal to num_channels\
                              divided by groups.")
    stride = fluid.layers.utils.convert_to_list(stride, 2, 'stride')
    padding = fluid.layers.utils.convert_to_list(padding, 2, 'padding')
    dilation = fluid.layers.utils.convert_to_list(dilation, 2, 'dilation')
    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")
    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False
        })
    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return helper.append_activation(pre_act)

def conv2d_spectral_norm(input,
                         num_filters=64,
                         filter_size=7,
                         stride=1,
                         stddev=0.02,
                         padding=0,
                         name="conv2d_spectral_norm",
                         norm=None,
                         activation_fn=None,
                         relufactor=0.0,
                         use_bias=False,
                         padding_type=None,
                         initial="normal",
                         is_test=False,
                         norm_affine=True):
    b, c, h, w = input.shape
    height = num_filters
    width = c * filter_size * filter_size
    helper = fluid.layer_helper.LayerHelper("conv2d_spectral_norm", **locals())
    dtype = helper.input_dtype()
    weight_param = fluid.ParamAttr(
        name=name + ".weight_orig",
        initializer=fluid.initializer.Normal(
            loc=0.0, scale=1.0),
        trainable=True)
    weight = helper.create_parameter(
        attr=weight_param,
        shape=(num_filters, c, filter_size, filter_size),
        dtype=dtype)
    weight_spectral_norm = fluid.layers.spectral_norm(
        weight, dim=0, name=name + ".spectral_norm")
    weight = weight_spectral_norm
    if use_bias:
        bias_attr = fluid.ParamAttr(
            name=name + "_b",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=1.0))
    else:
        bias_attr = False
    conv = conv2d_with_filter(
        input, weight, stride, padding, bias_attr=bias_attr, name=name)
    if norm is not None:
        conv = norm_layer(
            input=conv,
            norm_type=norm,
            name=name + "_norm",
            is_test=is_test,
            affine=norm_affine)
    if activation_fn == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif activation_fn == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)
    return conv

def linear_spectral_norm(input,
                         num_filters=64,
                         filter_size=7,
                         stride=1,
                        #  stddev=0.02,
                         padding=0,
                         name="linear_spectral_norm",
                         spectral_norm=True,
                         get_weight=True,
                         norm=None,
                         activation_fn=None,
                         relufactor=0.0,
                         use_bias=False,
                         padding_type=None,
                         initial="normal",
                         is_test=False,
                         norm_affine=True):
    b, c, h, w = input.shape
    height = num_filters
    width = c * filter_size * filter_size
    helper = fluid.layer_helper.LayerHelper("linear_spectral_norm", **locals())
    dtype = helper.input_dtype()
    weight_param = fluid.ParamAttr(
        name=name + ".weight_orig",
        initializer=fluid.initializer.Normal(
            loc=0.0, scale=1.0),
        trainable=True)
    weight = helper.create_parameter(
        attr=weight_param,
        shape=(num_filters, c, filter_size, filter_size),
        dtype=dtype)
    if spectral_norm:
        weight_spectral_norm = fluid.layers.spectral_norm(
            weight, dim=0, name=name + ".spectral_norm")
        weight = weight_spectral_norm
    else:
        weight = weight
    if use_bias:
        bias_attr = fluid.ParamAttr(
            name=name + "_b",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=1.0))
    else:
        bias_attr = False
    conv = conv2d_with_filter(
        input, weight, stride, padding, bias_attr=bias_attr, name=name)
    if norm is not None:
        conv = norm_layer(
            input=conv,
            norm_type=norm,
            name=name + "_norm",
            is_test=is_test,
            affine=norm_affine)
    if activation_fn == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif activation_fn == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)
    if spectral_norm:
        return conv, weight
    else:
        if get_weight:
            return conv, weight
        else:
            return conv
def initial_type(name,
                 input,
                 op_type,
                 fan_out,
                 init="normal",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02):
    if init == "kaiming":
        if op_type == 'conv':
            fan_in = input.shape[1] * filter_size * filter_size
        elif op_type == 'deconv':
            fan_in = fan_out * filter_size * filter_size
        else:
            if len(input.shape) > 2:
                fan_in = input.shape[1] * input.shape[2] * input.shape[3]
            else:
                fan_in = input.shape[1]
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_b", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr


def var(input, dim=None, keep_dim=False, unbiased=True, name=None):
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in range(dims)]
    inp_shape = input.shape
    mean = fluid.layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = fluid.layers.reduce_mean((input - mean)**2, dim=dim, keep_dim=keep_dim, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp

def adaILN(input, num_features, gamma, beta):
    rho = fluid.layers.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', name='_rho1')
    in_mean, in_var = fluid.layers.reduce_mean(fluid.layers.reduce_mean(input, dim=2, keep_dim=True), dim=3, keep_dim=True), var(var(input, dim=2, keep_dim=True), dim=3, keep_dim=True)
    out_in = (input - in_mean) / fluid.layers.sqrt(in_var + 1e-5)
    ln_mean, ln_var = fluid.layers.reduce_mean(fluid.layers.reduce_mean(fluid.layers.reduce_mean(input, dim=1, keep_dim=True), dim=2, keep_dim=True), dim=3, keep_dim=True), var(var(var(input, dim=1, keep_dim=True), dim=2, keep_dim=True), dim=3, keep_dim=True)
    out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + 1e-5)
    rho_a = rho
    rho_b = 1-rho
    out = rho_a * out_in + rho_b * out_ln 
    gamma = fluid.layers.unsqueeze(fluid.layers.unsqueeze(gamma, 2), 3)
    beta = fluid.layers.unsqueeze(fluid.layers.unsqueeze(beta, 2), 3)
    out = out * gamma + beta
    return out

def ILN(input, num_features, name):
    rho2 = fluid.layers.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', name=name+'_rho2')
    gamma = fluid.layers.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', name=name+'_gamma')
    beta = fluid.layers.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', name=name+'_beta')
    in_mean, in_var = fluid.layers.reduce_mean(fluid.layers.reduce_mean(input, dim=2, keep_dim=True), dim=3, keep_dim=True), var(var(input, dim=2, keep_dim=True), dim=3, keep_dim=True)
    out_in = (input - in_mean) / fluid.layers.sqrt(in_var + 1e-5)
    ln_mean, ln_var = fluid.layers.reduce_mean(fluid.layers.reduce_mean(fluid.layers.reduce_mean(input, dim=1, keep_dim=True), dim=2, keep_dim=True), dim=3, keep_dim=True), var(var(var(input, dim=1, keep_dim=True), dim=2, keep_dim=True), dim=3, keep_dim=True)
    out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + 1e-5)
    rho2_a = rho2
    rho2_b = 1-rho2
    out = rho2_a * out_in + rho2_b * out_ln
    out = out * gamma + beta
    return out