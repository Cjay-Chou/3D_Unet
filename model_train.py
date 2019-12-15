from config import UConfig

import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras import layers as klayers
from tensorflow import name_scope
import argparse
import random


def CreateConv3DBlock(x, filters, n=2, use_bn=True, apply_pooling=True, name='convblock'):
    for i in range(n):
        x = klayers.Conv3D(filters[i], (3, 3, 3), padding='valid', name=name + '_conv' + str(i + 1))(x)
        if use_bn:
            x = klayers.BatchNormalization(name=name + '_BN' + str(i + 1))(x)
        x = klayers.Activation('relu', name=name + '_relu' + str(i + 1))(x)

    convresult = x

    if apply_pooling:
        x = klayers.MaxPool3D(pool_size=(2, 2, 2), name=name + '_pooling')(x)

    return x, convresult


def CreateUpConv3DBlock(x, contractpart, filters, n=2, use_bn=True, name='upconvblock'):
    # upconv x
    x = klayers.Conv3DTranspose((int)(x.shape[-1]), (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=False,
                                name=name + '_upconv')(x)
    # concatenate contractpart and x
    c = [(i - j) // 2 for (i, j) in zip(contractpart[0].shape[1:4].as_list(), x.shape[1:4].as_list())]
    contract_crop = klayers.Cropping3D(cropping=((c[0], c[0]), (c[1], c[1]), (c[2], c[2])))(contractpart[0])
    if len(contractpart) > 1:
        crop1 = klayers.Cropping3D(cropping=((c[0], c[0]), (c[1], c[1]), (c[2], c[2])))(contractpart[1])
        # crop2 = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[2])
        # x = klayers.concatenate([contract_crop, crop1, crop2, x])
        x = klayers.concatenate([contract_crop, crop1, x])
    else:
        x = klayers.concatenate([contract_crop, x])

    # conv x 2 times
    for i in range(n):
        x = klayers.Conv3D(filters[i], (3, 3, 3), padding='valid', name=name + '_conv' + str(i + 1))(x)
        if use_bn:
            x = klayers.BatchNormalization(name=name + '_BN' + str(i + 1))(x)
        x = klayers.Activation('relu', name=name + '_relu' + str(i + 1))(x)

    return x


def Construct3DUnetModel(input_images, nclasses, use_bn=True, use_dropout=True):
    with name_scope("contract1"):
        x, contract1 = CreateConv3DBlock(input_images, (32, 64), n=2, use_bn=use_bn, name='contract1')

    with name_scope("contract2"):
        x, contract2 = CreateConv3DBlock(x, (64, 128), n=2, use_bn=use_bn, name='contract2')

    with name_scope("contract3"):
        x, contract3 = CreateConv3DBlock(x, (128, 256), n=2, use_bn=use_bn, name='contract3')

    with name_scope("contract4"):
        x, _ = CreateConv3DBlock(x, (256, 512), n=2, use_bn=use_bn, apply_pooling=False, name='contract4')

    with name_scope("dropout"):
        if use_dropout:
            x = klayers.Dropout(0.5, name='dropout')(x)

    with name_scope("expand3"):
        x = CreateUpConv3DBlock(x, [contract3], (256, 256), n=2, use_bn=use_bn, name='expand3')

    with name_scope("expand2"):
        x = CreateUpConv3DBlock(x, [contract2], (128, 128), n=2, use_bn=use_bn, name='expand2')

    with name_scope("expand1"):
        x = CreateUpConv3DBlock(x, [contract1], (64, 64), n=2, use_bn=use_bn, name='expand1')

    with name_scope("segmentation"):
        layername = 'segmentation_{}classes'.format(nclasses)
        x = klayers.Conv3D(nclasses, (1, 1, 1), activation='softmax', padding='same', name=layername)(x)

    return x


def dice(y_true, y_pred):
    K = tf.keras.backend
    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.math.count_nonzero(predictions, dtype=tf.float32) + tf.math.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice


def ImportImage(filename):
    image = sitk.ReadImage(filename)
    imagearry = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        imagearry = imagearry[..., np.newaxis]
    return imagearry


def GenerateBatchData(datalist, paddingsize, batch_size=32):
    ps = paddingsize[::-1]  # (x, y, z) -> (z, y, x) for np.array
    # j = 0

    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            imagelist = []
            outputlist = []

            for idx in indices[i:i + batch_size]:
                image = ImportImage(datalist[idx][0])
                onehotlabel = ImportImage(datalist[idx][1])

                onehotlabel = onehotlabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
                imagelist.append(image)
                outputlist.append(onehotlabel)

            yield (np.array(imagelist), np.array(outputlist))


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, datas, paddingsize, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.paddingsize = paddingsize
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.floor(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        """ 
        step = int(len(self.datas)/self.batch_size)-1
        if index > step :
          index=0
        """
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]
        """ 
        if len(batch_datas) != 8:
          with open("drive/My Drive/DATA/inter/wron_data.txt",'a') as f:
            f.write("len Wrong \n" )
            f.write(str(len(batch_datas) ))
            f.write(str(index))
        """

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def ImportImage(self, filename):
        image = sitk.ReadImage("drive/My Drive/DATA/inter/" + filename)
        imagearry = sitk.GetArrayFromImage(image)
        if image.GetNumberOfComponentsPerPixel() == 1:
            imagearry = imagearry[..., np.newaxis]
        return imagearry

    def data_generation(self, batch_datas):
        ps = self.paddingsize[::-1]  # (x, y, z) -> (z, y, x) for np.array
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            # x_train数据
            image = self.ImportImage(data[0])
            image = image[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
            images.append(image)

            # y_train数据
            onehotlabel = self.ImportImage(data[1])
            onehotlabel = onehotlabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
            labels.append(onehotlabel)

            """
            #For test
            image = self.ImportImage(self.datas[0][0])
            image = image[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
            images.append(image)
            onehotlabel = self.ImportImage(self.datas[0][1])
            onehotlabel = onehotlabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
            labels.append(onehotlabel)
            """
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(images), np.array(labels)


def start_train(logdir, train_file, val_file=None, batchsize=16, epochs=30, learningrate=1e-3, ):
    # Build 3DU-net

    patchsize = (44, 44, 28)
    # read from config outputsize
    padding = 44
    imagesize = tuple([p + 2 * padding for p in patchsize])
    inputshape = imagesize + (1,)
    nclasses = 9

    print("Input shape:", inputshape)
    print("Number of classes:", nclasses)

    inputs = tf.keras.layers.Input(shape=inputshape, name="input")
    segmentation = Construct3DUnetModel(inputs, nclasses, True, True)

    model = tf.keras.models.Model(inputs, segmentation, name="3DUnet")
    model.summary()

    # Start training
    optimizer = tf.keras.optimizers.Adam(lr=learningrate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[dice])

    # get padding size
    ps = np.array(model.output_shape[1:4])
    ips = np.array(model.input_shape[1:4])
    paddingsize = ((ips - ps) / 2).astype(np.int)

    # A retraining of interruption
    initial_epoch = 0
    '''if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile, by_name=True)
        initial_epoch = args.initialepoch'''

    if not os.path.exists(logdir + '/model'):
        os.makedirs(logdir + '/model')
    latestfile = logdir + '/latestweights.hdf5'
    bestfile = logdir + '/bestweights.hdf5'
    tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    best_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=bestfile, save_best_only=True)  # , save_weights_only = True)
    latest_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=latestfile)  # , save_weights_only = True)
    every_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=logdir + '/model/model_{epoch:02d}_{val_loss:.2f}.hdf5')
    callbacks = [tb_cbk, best_cbk, latest_cbk, every_cbk]

    # data_file = "list/hist_train.txt"
    # val_file = "list/hist_val.txt"

    # read dataset
    trainingdatalist = train_file
    train_data = GenerateBatchData(trainingdatalist, paddingsize, batch_size=batchsize)
    if val_file is not None:
        testdatalist = val_file
        # testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.3))
        validation_data = GenerateBatchData(testdatalist, paddingsize, batch_size=batchsize)
        validation_steps = len(testdatalist) / batchsize
    else:
        validation_data = None
        validation_steps = None

    steps_per_epoch = len(trainingdatalist) / batchsize
    print("Number of samples:", len(trainingdatalist))
    print("Batch size:", batchsize)
    print("Number of Epochs:", epochs)
    print("Learning rate:", learningrate)
    print("Number of Steps/epoch:", steps_per_epoch)

    model.fit_generator(train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)


def train(config_path):
    c = UConfig(config_path)
    train_list = ReadSliceDataList(c, 'train')
    val_list = ReadSliceDataList(c, 'val')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_soft_device_placement(True)
    print("Num GPUs Available: ", len(gpus))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    start_train(c.log_dir, train_list, val_list)
    print(len(train_list), len(val_list), c.log_dir)
    print(train_list[:10])


def ReadSliceDataList(c: UConfig, train_val):
    data_list_path = os.path.join('./lists', c.train_data)
    label_list_path = os.path.join('./lists', c.train_label)
    datalist = []
    if train_val == 'train':
        num_data_lists = c.train_list
    else:
        num_data_lists = c.val_list

    for i in num_data_lists:
        dfname = os.path.join(data_list_path, i)
        lfname = os.path.join(label_list_path, i)
        with open(dfname) as f:
            datas = f.readlines()
        with open(lfname) as f:
            labels = f.readlines()
        for imagefile, labelfile in zip(datas, labels):
            imagefile = imagefile.replace("\\\\", "\\")
            imagefile = imagefile.replace("\n", "")
            labelfile = labelfile.replace("\\\\", "\\")
            labelfile = labelfile.replace("\n", "")
            datalist.append((imagefile, labelfile))

    return datalist


def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build 3D_U_Net program')
    parser.add_argument("config_path", help="Input config file")
    parser.add_argument("-f", "--force", help="over write on old file if exist")
    parser.add_argument("--noLabel", help="not extract label file", dest='do_label', action='store_false')
    myargs = parser.parse_args()
    return myargs


if __name__ == '__main__':
    args = ParseArgs()
    train(args.config_path)
