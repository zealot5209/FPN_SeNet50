# import caffe
# from caffe import layers as L
# from caffe import params as P
#
#
# # Function: Set lenet net
# def init_net_lenet(netName,netType,batch_size,prototxt_root,data_root):
#
#     # --------
#     # set the type and name of the net, e.g., train_lenet, or test_lenet
#     NetTypeName = netType + '_'+netName
#
#     # --------
#     n = caffe.NetSpec()
#
#     # --------
#     # set the input layer
#     n.data, n.label = L.ImageData(
#             image_data_param={"batch_size": batch_size, "is_color" : False,"shuffle": True},
#             source= data_root+netType+".data", transform_param=dict(scale=1./255), ntop=2)
#
#     # ------
#     # set other layers
#     n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=6, weight_filler=dict(type='xavier'))
#     n.tanh1 = L.TanH(n.conv1, in_place=True)
#     n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#
#     n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=16, weight_filler=dict(type='xavier'))
#     n.tanh2 = L.TanH(n.conv2, in_place=True)
#     n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#
#     n.ip3 = L.InnerProduct(n.pool2, num_output=120, weight_filler=dict(type='xavier'))
#     n.tanh3 = L.TanH(n.ip3, in_place=True)
#
#     n.ip4 = L.InnerProduct(n.ip3, num_output=84, weight_filler=dict(type='xavier'))
#     n.tanh4 = L.TanH(n.ip4, in_place=True)
#
#     n.ip5 = L.InnerProduct(n.ip4, num_output=2, weight_filler=dict(type='xavier'))
#     n.loss = L.SoftmaxWithLoss(n.ip5, n.label)
#
#     # --------
#     # write the prototxt file
#     print('Writing net to %s' % prototxt_root+NetTypeName+'.prototxt')
#     with open(prototxt_root + NetTypeName+'.prototxt', 'w') as f:
#         f.write(str(n.to_proto()))
#     print 'done...'
#
#     # --------
#     # return the name of the output layer (used for predicting)
#     return 'ip5'
#
#
# if __name__ == '__main__':
#     init_net_lenet('test_net', 'test_net', 1000, '/home/yansong/', 'data_root')


#-*- coding: UTF-8 -*-
import caffe                                                         #

caffe_root = "/home/Jack-Cui/caffe-master/my-caffe-project/"         #
train_lmdb = caffe_root + "img_train.lmdb"                           #
mean_file = caffe_root + "mean.binaryproto"                          #


net = caffe.NetSpec()

net.data, net.label = caffe.layers.Data(source=train_lmdb, backend=caffe.params.Data.LMDB, batch_size=64, ntop=2,
                                        transform_param=dict(crop_size=40, mean_file=mean_file, mirror=True))
print str(net.to_proto())