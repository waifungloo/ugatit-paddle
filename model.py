from layers import conv2d, conv2d_spectral_norm, linear_spectral_norm, adaILN, ILN
import paddle.fluid as fluid


def build_resnet_block(inputres, dim, name="resnet"):
    out_res = fluid.layers.pad2d(inputres, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, "VALID", name + "_c1", relu=False)
    out_res = fluid.layers.relu(out_res)
    out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
    out_res = conv2d(out_res, dim, 3, 1, 0.02, "VALID", name + "_c2", relu=False)
    return out_res + inputres


def ResnetAdaILNBlock(input, dim, gamma, beta, name):
    pad_input = fluid.layers.pad2d(input, [1, 1, 1, 1], mode="reflect")
    conv1 = conv2d(pad_input, dim, 3, 1, 0.02, "VALID", name=name + "rab_c1", norm=False, relu=False)
    norm1 = adaILN(conv1, dim, gamma, beta)
    norm1 = fluid.layers.relu(norm1)
    norm1 = fluid.layers.pad2d(norm1, [1, 1, 1, 1], mode="reflect")
    conv2 = conv2d(norm1, dim, 3, 1, 0.02, "VALID", name=name + "rab_c2", norm=False, relu=False)
    norm2 = adaILN(conv2, dim, gamma, beta)
    return norm2

def Upsample(inputs, scale=2):
    shape_nchw = fluid.layers.shape(inputs)
    shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
    shape_hw.stop_gradient = True
    in_shape = fluid.layers.cast(shape_hw, dtype='int32')
    out_shape = in_shape * scale
    out_shape.stop_gradient = True
    out = fluid.layers.resize_nearest(
        input=inputs, scale=scale, actual_shape=out_shape)
    return out

def build_generator_resnet_blocks(inputgen, name="generator"):  

    # downsampling
    pad_input = fluid.layers.pad2d(inputgen, [3, 3, 3, 3], mode="reflect")
    o_c1 = conv2d(pad_input, 64, 7, 1, 0.02, "VALID", name=name + "_gc1", relu=False)
    o_c1 = fluid.layers.relu(o_c1)
    o_c2 = fluid.layers.pad2d(o_c1, [1, 1, 1, 1], mode="reflect")
    o_c2 = conv2d(o_c2, 128, 3, 2, 0.02, "VALID", name=name + "_gc2", relu=False)
    o_c2 = fluid.layers.relu(o_c2)
    o_c3 = fluid.layers.pad2d(o_c2, [1, 1, 1, 1], mode="reflect")
    o_c3 = conv2d(o_c3, 256, 3, 2, 0.02, "VALID", name=name + "_gc3", relu=False)
    o_c3 = fluid.layers.relu(o_c3)
    o_r1 = build_resnet_block(o_c3, 256, name + "_r1")
    o_r2 = build_resnet_block(o_r1, 256, name + "_r2")
    o_r3 = build_resnet_block(o_r2, 256, name + "_r3")
    o_r4 = build_resnet_block(o_r3, 256, name + "_r4")
    o_r5 = build_resnet_block(o_r4, 256, name + "_r5")
    o_r6 = build_resnet_block(o_r5, 256, name + "_r6")
    gap = fluid.layers.adaptive_pool2d(o_r6, 1, 'avg')
    _gap_logit, gap_weight = linear_spectral_norm(gap, 1, 1, 1, name=name+'_gl1', spectral_norm=False)
    gap = o_r6 * gap_weight
    gmp = fluid.layers.adaptive_pool2d(o_r6, 1, 'max')
    _gmp_logit, gmp_weight = linear_spectral_norm(gmp, 1, 1, 1, name=name+'_gl2', spectral_norm=False)
    gmp = o_r6 * gmp_weight
    cat = fluid.layers.concat([gap, gmp], 1)
    o_c4 = conv2d(cat, 256, 1, 1, 0.02, "VALID", name=name + "_gc4", relu=False, bias_attr_bool=True)
    o_c4 = fluid.layers.relu(o_c4)
    x = fluid.layers.adaptive_pool2d(o_c4, 1, 'avg') 
    x = linear_spectral_norm(x, 256, 1, 1, name=name+'_gl3', spectral_norm=False, get_weight=False)
    x = linear_spectral_norm(x, 256, 1, 1, name=name+'_gl4', spectral_norm=False, get_weight=False)
    gamma = linear_spectral_norm(x, 256, 1, 1, name=name+'_gl5', spectral_norm=False, get_weight=False)
    beta = linear_spectral_norm(x, 256, 1, 1, name=name+'_gl6', spectral_norm=False, get_weight=False)
    gamma = fluid.layers.reshape(gamma, (-1, 256))
    beta = fluid.layers.reshape(beta, (-1, 256))

    #upsampling
    a_r1 = ResnetAdaILNBlock(o_c4, 256, gamma, beta, name=name+'a1')
    a_r2 = ResnetAdaILNBlock(a_r1, 256, gamma, beta, name=name+'a2')
    a_r3 = ResnetAdaILNBlock(a_r2, 256, gamma, beta, name=name+'a3')
    a_r4 = ResnetAdaILNBlock(a_r3, 256, gamma, beta, name=name+'a4')
    a_r5 = ResnetAdaILNBlock(a_r4, 256, gamma, beta, name=name+'a5')
    a_r6 = ResnetAdaILNBlock(a_r5, 256, gamma, beta, name=name+'a6')
    out = Upsample(a_r6, scale=2)
    out = fluid.layers.pad2d(out, [1, 1, 1, 1], mode="reflect")
    out = conv2d(out, 128, 3, 1, 0.02, "VALID", name=name + "_u1", norm=False, relu=False)
    out = ILN(out, 128, 'ILN1')
    out = fluid.layers.relu(out)
    out = Upsample(out, scale=2)
    out = fluid.layers.pad2d(out, [1, 1, 1, 1], mode="reflect")
    out = conv2d(out, 64, 3, 1, 0.02, "VALID", name=name + "_u2", norm=False, relu=False)
    out = ILN(out, 64, 'ILN2')
    out = fluid.layers.relu(out)
    out = fluid.layers.pad2d(out, [3, 3, 3, 3], mode="reflect")
    out = conv2d(out, 3, 7, 1, 0.02, "VALID", name=name + "_u3", norm=False, relu=False)
    out = fluid.layers.tanh(out, name + "_t1")

    return out


def build_gen_discriminator(inputdisc, name="discriminator"):

    pad_input = fluid.layers.pad2d(inputdisc, [1, 1, 1, 1], mode="reflect")
    d_c1 = conv2d_spectral_norm(pad_input, 64, 4, 2, name="_dc1", activation_fn='leaky_relu', relufactor=0.2, use_bias=True)
    d_c2 = fluid.layers.pad2d(d_c1, [1, 1, 1, 1], mode="reflect")
    d_c2 = conv2d_spectral_norm(d_c2, 128, 4, 2, name="_dc2", activation_fn='leaky_relu', relufactor=0.2, use_bias=True)
    d_c3 = fluid.layers.pad2d(d_c2, [1, 1, 1, 1], mode="reflect")
    d_c3 = conv2d_spectral_norm(d_c3, 256, 4, 2, name="_dc3", activation_fn='leaky_relu', relufactor=0.2, use_bias=True)
    d_c4 = fluid.layers.pad2d(d_c3, [1, 1, 1, 1], mode="reflect")
    d_c4 = conv2d_spectral_norm(d_c4, 512, 4, 1, name="_dc4", activation_fn='leaky_relu', relufactor=0.2, use_bias=True)
    dap = fluid.layers.adaptive_pool2d(d_c4, 1, 'avg')
    _dap_logit, dap_weight = linear_spectral_norm(dap, 1, 1, 1, name=name+'_dl1')
    dap = d_c4 * dap_weight
    dmp = fluid.layers.adaptive_pool2d(d_c4, 1, 'avg')
    _dmp_logit, dmp_weight = linear_spectral_norm(dmp, 1, 1, 1, name=name+'_dl2')
    dmp = d_c4 * dmp_weight
    out = fluid.layers.concat([dap, dmp], 1)
    out = conv2d(out, 512, 1, 1, 0.02, "VALID", name=name + "_dc6", norm=False, relufactor=0.2, bias_attr_bool=True)
    out = fluid.layers.pad2d(out, [1, 1, 1, 1], mode="reflect")
    out = conv2d_spectral_norm(out, 1, 2, 1, name="_dc5")
    return out