import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np
import yaml
import SimpleITK as sitk
from pathlib import Path
import pickle



def createParentPath(filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def Padding(image, patchsize, imagepatchsize, mirroring = False):
    padfilter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    padfilter.SetPadLowerBound(patchsize.tolist())
    padfilter.SetPadUpperBound(imagepatchsize.tolist())
    padded_image = padfilter.Execute(image)
    return padded_image

def resampling(image,up_down,up_size = None):
    #down CT, up result
    
    oldsize = image.GetSize()
    oldspacing = image.GetSpacing()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None


    if up_down == "up":
        newsize = tuple(up_size)
        newspacing = tuple([oldspacing[0],oldspacing[1]/2.0,oldspacing[2]/2.0])
    elif up_down == "down":
        newsize = tuple([oldsize[0],oldsize[1]//2,oldsize[2]//2])
        newspacing = tuple([oldspacing[0],oldspacing[1]*2,oldspacing[2]*2])
    

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newsize)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(newspacing)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if up_down == "up":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    if up_down == "up":
        resampledarr = sitk.GetArrayFromImage(resampled)
        resampledarr = resampledarr.astype(np.uint8)
        outimage = sitk.GetImageFromArray(resampledarr)
        outimage.SetOrigin(resampled.GetOrigin())
        outimage.SetSpacing(resampled.GetSpacing())
        outimage.SetDirection(resampled.GetDirection())
        resampled = outimage


    return oldsize, resampled

def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)    

def weighted_crossentropy_loss(y_true, y_pred, axis = -1):
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

def prediction(model, file_abspath, out_abspath, mask=None, gpuid = 0, stepscale = 1):
    #prediction(model_path,file_path,out_path,mask="mask_re12.mha")


    # get patchsize model(x,y,z)->z,y,x
    ps = np.array(model.output_shape[1:4][::-1])
    ips = np.array(model.input_shape[1:4][::-1])
    ds = ((ips - ps) / 2).astype(np.int)
    
    print("loading input image", file_abspath, end="...", flush=True)

    image = sitk.ReadImage(file_abspath)
    #big_size, downsampled = resampling(image,"down",up_size = None)
    print(" image size: ",image.GetSize())
    if image.GetSize()[0]>250:
        image = image[:250,:,:]
        print("Z > 250 Cutted")

    image_padded = Padding(image, ds, ips, mirroring = True)
    print("done")
    s = image_padded.GetSize()
    print("ps:{}, ips:{}, ds:{}, padded_s:{}".format(ps, ips, ds, s))

    maskarry = None

    bb = (ds[0], ds[0]+image.GetSize()[0]-1, ds[1], ds[1]+image.GetSize()[1]-1, ds[2], ds[2]+image.GetSize()[2]-1)
    print("bb", bb)
    if mask is not None:
        print("loading mask image", mask, end="...", flush=True)
        maskimage = sitk.ReadImage(mask)
        maskimage_padded = Padding(maskimage, ds, ips)
        maskarry = sitk.GetArrayFromImage(maskimage_padded)
        print("mask image size:",maskimage.GetSize())
        print("done")

    
    step = (ps / stepscale).astype(np.int)

    totalpatches = [i for i in product( range(bb[4], bb[5], step[2]), range(bb[2], bb[3], step[1]), range(bb[0], bb[1], step[0]))]
    num_totalpatches = len(totalpatches)
    #patchindices = [i for i in product( range(bb[4], bb[5], ps[2]), range(bb[2], bb[3], ps[1]), range(bb[0], bb[1], ps[0]))]

    label = sitk.Image(image_padded.GetSize(), sitk.sitkUInt8)
    labelarr = sitk.GetArrayFromImage(label)
    #print('labelarr shape: {}'.format(labelarr.shape))
    counterarr = sitk.GetArrayFromImage(sitk.Image(image_padded.GetSize(), sitk.sitkVectorUInt8, model.output_shape[-1]))
    #print(counterarr.GetSize())
    #array x,y,z,c.
    paarry = np.zeros(shape=(image_padded.GetSize()[::-1] + (model.output_shape[-1],)), dtype="float32")

    i = 1
    batchs = []
    for iz in range(bb[4], bb[5], step[2]):
        for iy in range(bb[2], bb[3], step[1]):
            for ix in range(bb[0], bb[1], step[0]):
                p = [ix, iy, iz]
                #print("patch [{} / {}] : {}".format(i, num_totalpatches, p), end="...", flush=True)
                
                ii = [p[0]-ds[0], p[1]-ds[1], p[2]-ds[2]]
                if ii[0]+ips[0] > s[0] or ii[1]+ips[1] > s[1] or ii[2]+ips[2] > s[2]:
                    print("skipped")
                    continue
                if maskarry is not None:
                    if np.sum(maskarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])]) < 1:
                        print("skipped")
                        continue

                patchimage = image_padded[ii[0]:(ii[0]+ips[0]), ii[1]:(ii[1]+ips[1]), ii[2]:(ii[2]+ips[2])]
                patchimagearray = sitk.GetArrayFromImage(patchimage)
                patchimagearray = patchimagearray[..., np.newaxis]
                #patchimagearray = np.array([patchimagearray[..., np.newaxis]])
                batchs.append(patchimagearray)
                """if i ==1:
                    batchsarry = patchimagearray
                else:
                    batchsarry = np.vstack((batchsarry,patchimagearray))"""

                i = i + 1

    batchsarry = np.array(batchs)
    del batchs
    print(batchsarry.shape)
    pavec = model.predict(batchsarry,batch_size=16, verbose=1)
    del batchsarry
    print(pavec.shape)
    #pickle.dump(pavec,out_abspath+"pcl")
    batchnum = 0
    for iz in range(bb[4], bb[5], step[2]):
        for iy in range(bb[2], bb[3], step[1]):
            for ix in range(bb[0], bb[1], step[0]):
                p = [ix, iy, iz]
                #segmentation = np.argmax(pavec, axis=-1).astype(np.uint8)
                #labelarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])] = segmentation
                pavec[batchnum][:,:,:,1] += 0.1
                if batchnum == 0:
                    print(pavec[batchnum].shape)
                paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += pavec[batchnum][48:-48,48:-48,8:-8]
                counterarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += np.ones(pavec[batchnum].shape, dtype = np.uint8)

                #if paarry is not None:
                #    paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] = pavec
                batchnum+=1
                #print("done",batchnum)
    del pavec
    counterarr[counterarr == 0] = 1
    paarry = paarry / counterarr
    labelarr = np.argmax(paarry, axis=-1).astype(np.uint8)

    print("saving segmented label to", out_abspath, end="...", flush=True)
    createParentPath(out_abspath)
    labelarr = labelarr[ds[2]:-ips[2], ds[1]:-ips[1], ds[0]:-ips[0]]
    label = sitk.GetImageFromArray(labelarr)
    label.SetOrigin(image.GetOrigin())
    label.SetSpacing(image.GetSpacing())
    label.SetDirection(image.GetDirection())
    #upsample
    #print("up sampling ...")
    #_, upsampled = resampling(label,"up",up_size=big_size)
    sitk.WriteImage(label, out_abspath, True)
    print("done")






test_list =  ['079', '056', '104', '185', '187', '039', '091', '031', '147', '160', '014', '051', '138', '123', '109', '202', '102','070', '116', '127', '043', '106', '094', '089', '103', '159', '016', '203', '065', '153', '174', '047', '139', '157']
mask_test =  ['147', '160', '014', '051', '138', '123', '109', '202', '102', '079', '056', '104', '185', '187', '039', '091', '031', '209', '049', '150', '060', '155', '190', '093', '145', '205', '180', '194']
test_all_list =  ['210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299']
#debug = ['209', '049', '150', '060', '155', '190', '093', '145', '205', '180', '194']
list_all =  ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209']

source_path = "E:/kits19_test/"
#source_path = "E:/kits19_inter_processed/"
model_abspath = "D:/Script/log_his_com30/model/hist_02/model_04_0.072.hdf5"
out_path = "D:/Script/log_mask128_more/result_test_all_e3p01/"
mask="mask.mha"
gpuid = 0

#############Ready Training################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

with tf.device('/device:GPU:{}'.format(gpuid)):
    print("loading 3D U-net model", model_abspath, end="...", flush=True)
    model = tf.keras.models.load_model(model_abspath,custom_objects={'dice':dice,'weighted_crossentropy_loss':weighted_crossentropy_loss})
    print("done")
print("input_shape =", model.input_shape)
print("output_shape =", model.output_shape)


for num in test_all_list:
    
    file_path = os.path.join(source_path,"case_00"+num)
    out_abspath = os.path.join(out_path,"result"+num+".mha")
    mask_abspath = os.path.join(file_path,"mask.mha")
    image_abspath = os.path.join(file_path,"ct_hist25.mha")
   

    prediction(model,image_abspath,out_abspath,stepscale=2)#, mask=mask_abspath)



tf.keras.backend.clear_session()