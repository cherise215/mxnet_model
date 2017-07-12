# this file serves as a mxnet implementation of U-net.
# This implementation supports Batch Normalization
# U-net ref:O.Ronneberger et al. (2015). U-net: Convolutional networks for biomedical image segmentation. MICCAI.'
# author： cc
# cherise.chenchen@gmail.com
import mxnet as mx


def conv_unit(data, num_filter, workspace, name, act_type='relu', BatchNorm=False):
    conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               workspace=workspace, name=name + '_conv1')
    if BatchNorm:
        conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn1')
    act1 = mx.sym.Activation(data=conv1, act_type=act_type, name=name + '_relu1')

    return act1

def deconv_unit(data,num_filter,name,BatchNorm=False):
    net = mx.sym.Deconvolution(data, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=num_filter,workspace=2048,name=name+'deconv')
    if BatchNorm:
        net = mx.sym.BatchNorm(net,name=name+'_bn')
    return net

def symbol_unet_dc(data, label=None, num_class=6, BatchNorm=False):
    conv1_1 = conv_unit(data=data, num_filter=32, workspace=2048, name="conv1_1", BatchNorm=BatchNorm)
    conv1_2 = conv_unit(data=conv1_1, num_filter=32, workspace=2048, name="conv1_2", BatchNorm=BatchNorm)
    pool1 = mx.symbol.Pooling(data=conv1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")

    conv2_1 = conv_unit(data=pool1, num_filter=64, workspace=2048, name="conv2_1", BatchNorm=BatchNorm)
    conv2_2 = conv_unit(data=conv2_1, num_filter=64, workspace=2048, name="conv2_2", BatchNorm=BatchNorm)
    pool2 = mx.symbol.Pooling(data=conv2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")

    conv3_1 = conv_unit(data=pool2, num_filter=128, workspace=2048, name="conv3_1", BatchNorm=BatchNorm)
    conv3_2 = conv_unit(data=conv3_1, num_filter=128, workspace=2048, name="conv3_2", BatchNorm=BatchNorm)
    pool3 = mx.symbol.Pooling(data=conv3_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4_1 = conv_unit(data=pool3, num_filter=256, workspace=2048, name="conv4_1", BatchNorm=BatchNorm)
    conv4_2 = conv_unit(data=conv4_1, num_filter=256, workspace=2048, name="conv4_2", BatchNorm=BatchNorm)
    pool4 = mx.symbol.Pooling(data=conv4_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")

    conv5_1 = conv_unit(data=pool4, num_filter=512, workspace=2048, name="conv5_1", BatchNorm=BatchNorm)
    conv5_2 = conv_unit(data=conv5_1, num_filter=512, workspace=2048, name="conv5_2", BatchNorm=BatchNorm)

    u_conv5 = mx.symbol.UpSampling(conv5_2, scale=2, sample_type='nearest', name='u_conv5', workspace=2048)
    k_1 = mx.sym.Concat(*[u_conv5, conv4_2])

    conv6_1 = conv_unit(data=k_1, num_filter=256, workspace=2048, name="conv6_1", BatchNorm=BatchNorm)
    conv6_2 = conv_unit(data=conv6_1, num_filter=256, workspace=2048, name="conv6_2", BatchNorm=BatchNorm)
    u_conv6 = deconv_unit(data=conv6_2,num_filter=256,name='deconv_6',BatchNorm=BatchNorm)
    k_2 = mx.sym.Concat(*[u_conv6, conv3_2])

    conv7_1 = conv_unit(data=k_2, num_filter=128, workspace=2048, name="conv7_1", BatchNorm=BatchNorm)
    conv7_2 = conv_unit(data=conv7_1, num_filter=128, workspace=2048, name="conv7_2", BatchNorm=BatchNorm)
    u_conv7 = deconv_unit(data=conv7_2,num_filter=128,name='deconv_7',BatchNorm=BatchNorm)
    k_3 = mx.sym.Concat(*[u_conv7, conv2_2])

    conv8_1 = conv_unit(data=k_3, num_filter=64, workspace=2048, name="conv8_1", BatchNorm=BatchNorm)
    conv8_2 = conv_unit(data=conv8_1, num_filter=64, workspace=2048, name="conv8_2", BatchNorm=BatchNorm)
    u_conv8 = deconv_unit(data=conv8_2,num_filter=64,name='deconv_8',BatchNorm=BatchNorm)
    k_4 = mx.sym.Concat(*[u_conv8, conv1_2])

    conv9_1 = conv_unit(data=k_4, num_filter=32, workspace=2048, name="conv9_1", BatchNorm=BatchNorm)
    conv9_2 = conv_unit(data=conv9_1, num_filter=32, workspace=2048, name="conv9_2", BatchNorm=BatchNorm)
    conv10 = mx.symbol.Convolution(
        data=conv9_2, kernel=(1, 1), pad=(0, 0), num_filter=num_class, workspace=2048, name="conv10")

    softmax = mx.symbol.SoftmaxOutput(data=conv10, label=label, multi_output=True, use_ignore=True, ignore_label=255,
                                      name="softmax")
    return softmax

def symbol_unet(data, label=None, num_class=6, BatchNorm=False):
    conv1_1 = conv_unit(data=data, num_filter=32, workspace=2048, name="conv1_1", BatchNorm=BatchNorm)
    conv1_2 = conv_unit(data=conv1_1, num_filter=32, workspace=2048, name="conv1_2", BatchNorm=BatchNorm)
    pool1 = mx.symbol.Pooling(data=conv1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")

    conv2_1 = conv_unit(data=pool1, num_filter=64, workspace=2048, name="conv2_1", BatchNorm=BatchNorm)
    conv2_2 = conv_unit(data=conv2_1, num_filter=64, workspace=2048, name="conv2_2", BatchNorm=BatchNorm)
    pool2 = mx.symbol.Pooling(data=conv2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")

    conv3_1 = conv_unit(data=pool2, num_filter=128, workspace=2048, name="conv3_1", BatchNorm=BatchNorm)
    conv3_2 = conv_unit(data=conv3_1, num_filter=128, workspace=2048, name="conv3_2", BatchNorm=BatchNorm)
    pool3 = mx.symbol.Pooling(data=conv3_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4_1 = conv_unit(data=pool3, num_filter=256, workspace=2048, name="conv4_1", BatchNorm=BatchNorm)
    conv4_2 = conv_unit(data=conv4_1, num_filter=256, workspace=2048, name="conv4_2", BatchNorm=BatchNorm)
    pool4 = mx.symbol.Pooling(data=conv4_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")

    conv5_1 = conv_unit(data=pool4, num_filter=512, workspace=2048, name="conv5_1", BatchNorm=BatchNorm)
    conv5_2 = conv_unit(data=conv5_1, num_filter=512, workspace=2048, name="conv5_2", BatchNorm=BatchNorm)

    u_conv5 = mx.symbol.UpSampling(conv5_2, scale=2, sample_type='nearest', name='u_conv5', workspace=2048)
    k_1 = mx.sym.Concat(*[u_conv5, conv4_2])

    conv6_1 = conv_unit(data=k_1, num_filter=256, workspace=2048, name="conv6_1", BatchNorm=BatchNorm)
    conv6_2 = conv_unit(data=conv6_1, num_filter=256, workspace=2048, name="conv6_2", BatchNorm=BatchNorm)
    u_conv6 = mx.symbol.UpSampling(conv6_2, scale=2, sample_type='nearest', name='u_conv6', workspace=2048)
    k_2 = mx.sym.Concat(*[u_conv6, conv3_2])

    conv7_1 = conv_unit(data=k_2, num_filter=128, workspace=2048, name="conv7_1", BatchNorm=BatchNorm)
    conv7_2 = conv_unit(data=conv7_1, num_filter=128, workspace=2048, name="conv7_2", BatchNorm=BatchNorm)
    u_conv7 = mx.symbol.UpSampling(conv7_2, scale=2, sample_type='nearest', name='u_conv7', workspace=2048)
    k_3 = mx.sym.Concat(*[u_conv7, conv2_2])

    conv8_1 = conv_unit(data=k_3, num_filter=64, workspace=2048, name="conv8_1", BatchNorm=BatchNorm)
    conv8_2 = conv_unit(data=conv8_1, num_filter=64, workspace=2048, name="conv8_2", BatchNorm=BatchNorm)

    u_conv8 = mx.symbol.UpSampling(conv8_2, scale=2, sample_type='nearest', name='u_conv8', workspace=2048)
    k_4 = mx.sym.Concat(*[u_conv8, conv1_2])

    conv9_1 = conv_unit(data=k_4, num_filter=32, workspace=2048, name="conv9_1", BatchNorm=BatchNorm)
    conv9_2 = conv_unit(data=conv9_1, num_filter=32, workspace=2048, name="conv9_2", BatchNorm=BatchNorm)
    conv10 = mx.symbol.Convolution(
        data=conv9_2, kernel=(1, 1), pad=(0, 0), num_filter=num_class, workspace=2048, name="conv10")

    softmax = mx.symbol.SoftmaxOutput(data=conv10, label=label, multi_output=True, use_ignore=True, ignore_label=255,
                                      name="softmax")

    return softmax


def get_u_net(name=None,num_class=6):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    if name == 'deconv':
        net=symbol_unet_dc(data,label,BatchNorm=True,num_class=num_class) ## use deconvolution for upsampling
    else: net = symbol_unet(data,label,BatchNorm=True,num_class=num_class) ## use basic upsampling method.
    print net.list_arguments()
    arg_shape, output_shape, aux_shape = net.infer_shape(data=(1, 3, 512, 512))
    print 'arg_shape ', arg_shape, 'output_shape, ', output_shape, 'aux shape ', aux_shape
    return net






if __name__ == "__main__":
    get_u_net（）

