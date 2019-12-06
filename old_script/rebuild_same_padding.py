import sys
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.python.keras import layers as klayers
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow import name_scope
import argparse
import re
from pathlib import Path
import random
import yaml
import math

args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build 3D_U_Net program')
    parser.add_argument("datafile", help="Input Dataset file(stracture:data_path label_path)")
    parser.add_argument("-o", "--outfile", help="Output model structure file in YAML format (*.yml).")
    parser.add_argument("-t","--testfile", help="Input Dataset file for validation (stracture:data_path label_path)")
    parser.add_argument("-p", "--patchsize", help="Patch size. (ex. 44x44x28)", default="128x128x32")
    parser.add_argument("-c", "--nclasses", help="Number of classes of segmentaiton including background.", default=3, type=int)
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=30, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size*(Warning:memory use a lot)", default=3, type=int)
    parser.add_argument("-l", "--learningrate", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("--nobn", help="Do not use batch normalization layer", dest="use_bn", action='store_false')
    parser.add_argument("--nodropout", help="Do not use dropout layer", dest="use_dropout", action='store_false')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.floor(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def ImportImage(self, filename):
        image = sitk.ReadImage(filename)
        imagearry = sitk.GetArrayFromImage(image)
        if image.GetNumberOfComponentsPerPixel() == 1:
            imagearry = imagearry[..., np.newaxis]
        return imagearry

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            
            #x_train数据
            image = self.ImportImage(data[0])
            #image = image[42:-42,2:-2,2:-2]
            images.append(image)
            #y_train数据 
            onehotlabel = self.ImportImage(data[1])
            #onehotlabel = onehotlabel[42:-42,2:-2,2:-2]
            
            labels.append(onehotlabel)

        #如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(images), np.array(labels)


def CreateConv3DBlock(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock',dilation_rate = (1,1,1)):
    for i in range(n):
        if i == 0:
            x = klayers.Conv3D(filters[i], (3,3,3), padding='same', name=name+'_conv'+str(i+1))(x)
            if use_bn:
                x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
            x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)
        else:
            x = klayers.Conv3D(filters[i], (3,3,3), padding='same',dilation_rate=dilation_rate, name=name+'_conv'+str(i+1))(x)
            if use_bn:
                x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
            x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)
    

    convresult = x

    if apply_pooling:
        x = klayers.MaxPool3D(pool_size=(2,2,2), name=name+'_pooling')(x)

    return x, convresult

def CreateUpConv3DBlock(x, contractpart, filters, n = 2, use_bn = True, name = 'upconvblock'):
    # upconv x
    x = klayers.Conv3DTranspose((int)(x.shape[-1]), (2,2,2), strides=(2,2,2), padding='same', use_bias = False, name=name+'_upconv')(x)
    # concatenate contractpart and x
    """
    c = [(i-j)//2 for (i, j) in zip(contractpart[0].shape[1:4].as_list(), x.shape[1:4].as_list())]
    contract_crop = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[0])
    if len(contractpart) > 1:
        crop1 = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[1])
        #crop2 = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[2])
        #x = klayers.concatenate([contract_crop, crop1, crop2, x])
        x = klayers.concatenate([contract_crop, crop1, x])
    else:
    """
    x = klayers.concatenate([contractpart[0], x])

    # conv x 2 times
    for i in range(n):
        if i == 0:
            x = klayers.Conv3D(filters[i], (3,3,3), padding='same', name=name+'_conv'+str(i+1))(x)
            if use_bn:
                x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
            x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)
        else:
            x = klayers.Conv3D(filters[i], (3,3,3), padding='same', name=name+'_conv'+str(i+1))(x)
            if use_bn:
                x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
            x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)

    return x

def Construct3DUnetModel(input_images, nclasses, use_bn = True, use_dropout = True):
    with name_scope("contract1"):
        x, contract1 = CreateConv3DBlock(input_images, (64, 64), n = 2, use_bn = use_bn, name = 'contract1')

    with name_scope("contract2"):
        x, contract2 = CreateConv3DBlock(x, (128, 128), n = 2, use_bn = use_bn, name = 'contract2')

    with name_scope("contract3"):
        x, contract3 = CreateConv3DBlock(x, (256, 256), n = 2, use_bn = use_bn, name = 'contract3')

    with name_scope("contract4"):
        x, _ = CreateConv3DBlock(x, (256, 512), n = 2, use_bn = use_bn, apply_pooling = False, name = 'contract4')

    with name_scope("dropout"):
        if use_dropout:
            x = klayers.Dropout(0.5, name='dropout')(x)

    with name_scope("expand3"):
        x = CreateUpConv3DBlock(x, [contract3], (256, 256), n = 2, use_bn = use_bn, name = 'expand3')

    with name_scope("expand2"):
        x = CreateUpConv3DBlock(x, [contract2], (128, 128), n = 2, use_bn = use_bn, name = 'expand2')

    with name_scope("expand1"):
        x = CreateUpConv3DBlock(x, [contract1], (64, 64), n = 2, use_bn = use_bn, name = 'expand1')

    with name_scope("segmentation"):
        layername = 'segmentation_{}classes'.format(nclasses)
    
        x = klayers.Conv3D(nclasses, (1,1,1), activation='softmax', padding='same', name=layername)(x)

    return x

def ReadSliceDataList(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            imagefile, labelfile = line.strip().split()
            datalist.append((imagefile, labelfile))

    return datalist

def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def weighted_crossentropy_loss(y_true, y_pred, axis = -1):
    print("y_pred shape",y_pred.shape)
    print("y_true shape",y_true.shape)
    K = tf.keras.backend
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(1e-7, tf.float32)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    weight_y_org = tf.reduce_sum(y_true,[0,1,2,3])

    weight_y_3 = tf.pow(weight_y_org,1.0/3)
    #weight_y_log = tf.log(weight_y_org+1)

    weight_y = weight_y_3 / tf.reduce_sum(weight_y_3)

    return - tf.reduce_sum( 1 / (weight_y + _epsilon) * y_true * tf.log(y_pred), axis)

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

class MyCbk(tf.keras.callbacks.Callback):

    def __init__(self, model, filepath):
         self.model_to_save = model
         self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save_weights(self.filepath+'model_at_epoch_%d.h5' % epoch)

class ParallelModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

def main(_):
    #Build 3DU-net
    matchobj = re.match("([0-9]+)x([0-9]+)x([0-9]+)", args.patchsize)
    if matchobj is None:
        print('[ERROR] Invalid patch size : {}'.format(args.patchsize))
        return
    patchsize = [ int(s) for s in matchobj.groups() ]
    patchsize = tuple(patchsize)

    #padding = 44
    #imagesize = tuple([ p + 2*padding for p in patchsize ]) 
    inputshape = patchsize + (1,)
    nclasses = args.nclasses
    print("Input shape:", inputshape)
    print("Number of classes:", nclasses)

    
    #with tf.device('/cpu:0'):
    if args.weightfile == None:
        inputs = tf.keras.layers.Input(shape=inputshape, name="input")
        segmentation = Construct3DUnetModel(inputs, nclasses, args.use_bn, args.use_dropout)

        model = tf.keras.models.Model(inputs, segmentation,name="3DUnet")
        #mulmodel = multi_gpu_model(model,2)
    else:
         model = tf.keras.models.load_model(args.weightfile,custom_objects={'dice':dice,'weighted_crossentropy_loss':weighted_crossentropy_loss})

    model.summary()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(0)):
        optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[dice])
        #weighted_crossentropy_loss
    if args.outfile is not None:
        with open(args.outfile, 'w') as f:
            yamlobj = yaml.load(model.to_yaml())
            yaml.dump(yamlobj, f)
            
    #get padding size
    ps = np.array(model.output_shape[1:4])[::-1]
    ips = np.array(model.input_shape[1:4])[::-1]
    #paddingsize = ((ips - ps) / 2).astype(np.int)

    #A retraining of interruption
    if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile, by_name=True)
        initial_epoch = args.initialepoch


    if not os.path.exists(args.logdir+'/model'):
        os.makedirs(args.logdir+'/model')


    
    #hist_all = ['10','09','08','07','06','05','04','03','02','01','00']
    hist_all = ['03']
    for histid in hist_all:

        tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
        #best_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=bestfile, save_best_only = True)#, save_weights_only = True)
        #latest_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=latestfile)#, save_weights_only = True)
        every_cbk =  tf.keras.callbacks.ModelCheckpoint(filepath=args.logdir + '/model/hist_'+histid+'/model_{epoch:02d}_{val_loss:.3f}.hdf5')
        if not os.path.exists(args.logdir + '/model/hist_'+histid+'/'):
            os.makedirs(args.logdir + '/model/hist_'+histid+'/')
        callbacks = [tb_cbk,every_cbk]

        data_file = "list/com30/hist"+histid+"_train.txt"
        val_file = "list/com30/hist"+histid+"_val.txt"
        #read dataset
        trainingdatalist = ReadSliceDataList(data_file)
        #trainingdatalist = random.sample(trainingdatalist, int(len(trainingdatalist)*0.1))
        #train_data = GenerateBatchData(, paddingsize, batch_size = args.batchsize)
        train_data = DataGenerator(trainingdatalist, batch_size = args.batchsize)
        if args.testfile is not None:
            testdatalist = ReadSliceDataList(val_file)
            testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.3))
            validation_data = DataGenerator(testdatalist,batch_size = args.batchsize)
            validation_steps = len(testdatalist) / args.batchsize
        else:
            validation_data = None
            validation_steps = None

        steps_per_epoch = len(trainingdatalist) / args.batchsize
        print ("Number of samples:", len(trainingdatalist))
        print ("hist No.:", data_file)
        print ("Number of Epochs:", args.epochs)
        print ("Learning rate:", args.learningrate)
        print ("Number of Steps/epoch:", steps_per_epoch)



        #with tf.device('/gpu:0'):
        model.fit_generator(train_data,
                steps_per_epoch = steps_per_epoch,
                epochs = args.epochs,
                callbacks=callbacks,
                validation_data = validation_data,
                validation_steps = validation_steps,
                initial_epoch = 0,
                workers=8
                )


    tf.keras.backend.clear_session()


if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])