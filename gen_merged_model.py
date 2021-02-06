#!/usr/bin/env python
import os
import sys
import numpy as np
import google.protobuf as pb
sys.path.insert(0, "/home0/caffe/python") # set your caffe python path
import caffe

num_classes = 1000 # no use
num_blocks  = [1, 2, 4, 14, 1] # add 1 for fisrt stage
width_multiplier = [0.75, 0.75, 0.75, 2.5]
channels = [min(64, int(64 * width_multiplier[0])), 64, 128, 256, 512]# add 64 for fisrt stage

# find the key layer
def analysis_network(num_blocks):
    layer_number = 0
    layer_3x3_1x1 = []
    layer_count = []
    for i, layer_in_block in enumerate(num_blocks):
        layer_count.append(2*layer_number + 1)
        layer_3x3_1x1.append("conv" + str(2*layer_number + 1))
        layer_number += layer_in_block
    return layer_3x3_1x1, layer_count

# copy data
def copy_float(data):
    return np.array(data, copy=True, dtype=np.float32)


# fuse 3x3  
def fuse_conv_bn_3x3(net_src, id_layer_conv, id_layer_bn, Flag_conv_has_bias):
       
    key_conv = "conv" + str(id_layer_conv)
    key_bn = "batch_norm" + str(id_layer_bn)
    key_scale = "bn_scale" + str(id_layer_bn)     
    print ('Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale))
    
    # copy bn value
    bn_mean = copy_float(net_src.params[key_bn][0].data)
    bn_variance = copy_float(net_src.params[key_bn][1].data)
    num_bn_samples = copy_float(net_src.params[key_bn][2].data)

    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    # copy scale value
    scale_weight = copy_float(net_src.params[key_scale][0].data)
    scale_bias = copy_float(net_src.params[key_scale][1].data)
    
    # copy conv value
    weight = copy_float(net_src.params[key_conv][0].data)
    if Flag_conv_has_bias:
        bias = copy_float(net_src.params[key_conv][1].data)
    else:
        bias =(0,)*weight.shape[0]
        bias = np.array(bias, dtype=np.float32)
    
    # update
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.float32).eps)
    
    # merge 
    new_bias = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    new_weight = weight  
    for i in range(len(alpha)):
        new_weight[i]= weight[i] * alpha[i]  
    
    return new_weight, new_bias

# fuse 1x1   
def fuse_conv_bn_1x1(net_src, id_layer_conv, id_layer_bn, Flag_conv_has_bias):
        
    key_conv = "conv" + str(id_layer_conv)
    key_bn = "batch_norm" + str(id_layer_bn)
    key_scale = "bn_scale" + str(id_layer_bn)     
    print ('Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale))
    
    # copy bn value
    bn_mean = copy_float(net_src.params[key_bn][0].data)
    bn_variance = copy_float(net_src.params[key_bn][1].data)
    num_bn_samples = copy_float(net_src.params[key_bn][2].data)

    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    # copy scale value
    scale_weight = copy_float(net_src.params[key_scale][0].data)
    scale_bias = copy_float(net_src.params[key_scale][1].data)
    
    # copy conv value
    weight = copy_float(net_src.params[key_conv][0].data)
    if Flag_conv_has_bias:
        bias = copy_float(net_src.params[key_conv][1].data)
    else:
        bias =(0,)*weight.shape[0]
        bias = np.array(bias, dtype=np.float32)
    
    # update
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.float32).eps)
    
    # merge 
    new_bias = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    new_weight = weight  
    for i in range(len(alpha)):
        new_weight[i]= weight[i] * alpha[i]  
    
    # pad_1x1_to_3x3_tensor
    new_weight_pad = np.pad(new_weight, ((0,0),(0,0),(1,1),(1,1)), 'constant')
    return new_weight_pad, new_bias

# fuse identity
def fuse_conv_bn_id(net_src, id_layer_bn, Flag_conv_has_bias):
        
    key_bn = "batch_norm" + str(id_layer_bn)
    key_scale = "bn_scale" + str(id_layer_bn)     
    print ('Combine {:s} + {:s}'.format(key_bn, key_scale))
    
    # copy bn value
    bn_mean = copy_float(net_src.params[key_bn][0].data)
    bn_variance = copy_float(net_src.params[key_bn][1].data)
    num_bn_samples = copy_float(net_src.params[key_bn][2].data)

    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    # copy scale value
    scale_weight = copy_float(net_src.params[key_scale][0].data)
    scale_bias = copy_float(net_src.params[key_scale][1].data)
    
    # copy conv value
    input_dim = bn_mean.shape[0]
    
    # create 3x3_tensor
    weight = np.zeros((input_dim, input_dim, 3, 3), dtype=np.float32)
    for i in range(input_dim):
        weight[i, i % input_dim, 1, 1] = 1
                
    bias =(0,)*weight.shape[0]
    bias = np.array(bias, dtype=np.float32)

    # update
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.float32).eps)
    
    # merge 
    new_bias = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    new_weight = weight  
    for i in range(len(alpha)):
        new_weight[i]= weight[i] * alpha[i]  
    
    return new_weight, new_bias


def load_convert_fill(src_model, src_weights, dst_model, dst_weights):
    # load model
    with open(src_model) as f:
        model_old = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model_old)
          
    # load weights
    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    
    # get key layer
    layer_3x3_1x1, layer_count = analysis_network(num_blocks)
    print("key_layer: ", layer_3x3_1x1)
    
    # set id for loop
    id_layer_conv = 0
    id_layer_bn = 0
    id_layer_conv_merge = 0
    id_layer_fc_merge = 0
    for i, layer in enumerate(model_old.layer):
        if layer.type == 'Convolution': 
            # check conv layer has bias
            Flag_conv_bias = True
            if layer.convolution_param.bias_term == False:
                Flag_conv_bias = False
                
            # process key layer
            if layer.name in layer_3x3_1x1:
                idx = layer_3x3_1x1.index(layer.name)
                layer_repeate = num_blocks[idx]
                print("layer: ", layer.name, idx, layer_repeate, layer_count[idx])
                
                # set layer id
                id_layer_conv = layer_count[idx]
  
                # more than one layer
                for j in range(1, layer_repeate+1):

                    # id path
                    if j>1:
                        id_layer_bn += 1
                        new_weights_id, new_bias_id = fuse_conv_bn_id(net_src, id_layer_bn, Flag_conv_bias)
                        print("shape id: ", new_weights_1x1.shape, new_bias_1x1.shape)
                    
                    # 3x3 path
                    id_layer_bn += 1
                    new_weights_3x3, new_bias_3x3 = fuse_conv_bn_3x3(net_src, id_layer_conv, id_layer_bn, Flag_conv_bias)
                    print("shape 3x3: ", new_weights_3x3.shape, new_bias_3x3.shape)
                    
                    # 1x1 path
                    id_layer_conv += 1
                    id_layer_bn += 1
                    new_weights_1x1, new_bias_1x1 = fuse_conv_bn_1x1(net_src, id_layer_conv, id_layer_bn, Flag_conv_bias)
                    print("shape 1x1: ", new_weights_1x1.shape, new_bias_1x1.shape)
                    id_layer_conv += 1

                    # cal equivalent_kernel_bias
                    if j>1:
                        merge_weights = new_weights_3x3 + new_weights_1x1 + new_weights_id
                        merge_bias = new_bias_3x3 + new_bias_1x1 + new_bias_id
                    else:
                        merge_weights = new_weights_3x3 + new_weights_1x1
                        merge_bias = new_bias_3x3 + new_bias_1x1
                        
                    print("merge-weights: ", merge_weights.shape)
                    print("merge-bn: ", merge_bias.shape)
                    
                    # fill layer
                    id_layer_conv_merge += 1
                    print("id_layer_conv_merge: ", id_layer_conv_merge)
                    merge_conv_layer_name = "conv" + str(id_layer_conv_merge)
                    net_dst.params[merge_conv_layer_name][0].data[:] = merge_weights
                    net_dst.params[merge_conv_layer_name][1].data[:] = merge_bias
                    
                    print("--------------------------------------------------------\n")
                    
        if layer.type == 'InnerProduct':
            id_layer_fc_merge += 1
            merge_fc_layer_name = "fc" + str(id_layer_fc_merge)
            print("id_layer_fc_merge: ", merge_fc_layer_name)
            net_dst.params[merge_fc_layer_name][0].data[:] = net_src.params[merge_fc_layer_name][0].data[:]
            net_dst.params[merge_fc_layer_name][1].data[:] = net_src.params[merge_fc_layer_name][1].data[:]
            
    # save merged model
    net_dst.save(dst_weights)


if __name__ == '__main__':
    # original model
    old_model = "models/RepVGG-A0-train.caffemodel"
    old_prototxt = "models/RepVGG-A0-train.prototxt"
    
    # merged model
    new_model = "models/RepVGG-A0-deploy.caffemodel"
    new_prototxt = "models/RepVGG-A0-deploy.prototxt"

    # start merge
    load_convert_fill(old_prototxt, old_model, new_prototxt, new_model)