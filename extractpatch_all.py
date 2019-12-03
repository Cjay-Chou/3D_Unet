import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np

import yaml
import SimpleITK as sitk
from pathlib import Path
from keras.utils import to_categorical


def Padding(image, patchsize, imagepatchsize, mirroring = False):
    padfilter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    padfilter.SetPadLowerBound(patchsize.tolist())
    padfilter.SetPadUpperBound(imagepatchsize.tolist())
    padded_image = padfilter.Execute(image)
    return padded_image

def createParentPath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print("path created")

def do_extract(modelfile,imagefilepath,file_name,out_path,out_txt,islabel=False,mask=None,stepscale=1):
    
    input_shape=(1,32,128,128)
    output_shape=(1,16,64,64)

    # get patchsize z,y,x

    ps = np.array(output_shape[1:4])
    ips = np.array(input_shape[1:4])
    ds = ((ips - ps) / 2).astype(np.int)




    print("loading input image", imagefilepath+"/"+file_name, end="...", flush=True)
    image = sitk.ReadImage(imagefilepath+"/"+file_name)
    image_padded = Padding(image, ds, ips, mirroring = True)
    print("done")
    s = image_padded.GetSize()
    print("ps:{}, ips:{}, ds:{}, image_padded_s:{}".format(ps, ips, ds, s))

    #step = (ps/stepscale).astype(np.int)
    #step = np.array([20,32,32])
    step = (ps/stepscale).astype(np.int)


    bb = None
    maskarry = None

    #GetSize() is z y x
    bb = (ds[0], ds[0]+image.GetSize()[0]-1, ds[1], ds[1]+image.GetSize()[1]-1, ds[2], ds[2]+image.GetSize()[2]-1)
    print("bb, x,y,z", bb)

    if mask is not None:
        print("loading mask image", mask, end="...", flush=True)
        maskimage = sitk.ReadImage(imagefilepath+"/"+mask)
        maskimage_padded = Padding(maskimage, ds, ips)
        print("maskimage_padded shape",maskimage_padded.GetSize())
        maskarry = sitk.GetArrayFromImage(maskimage_padded)
        print("done")

    totalpatches = [i for i in product( range(bb[4], bb[5], step[2]), range(bb[2], bb[3], step[1]), range(bb[0], bb[1], step[0]))]
    num_totalpatches = len(totalpatches)

    if islabel:
        image_padded_array=sitk.GetArrayFromImage(image_padded)
        array_categorical=to_categorical(image_padded_array)
        i = 1
        j = 1
        arr_path = []
        for ix in range(bb[4], bb[5], step[2]):
            for iy in range(bb[2], bb[3], step[1]):
                for iz in range(bb[0], bb[1], step[0]):
                    p = [iz, iy, ix]
                    print("patch [{} / {}] : {}".format(i, num_totalpatches, p), end="...", flush=True)
                    i = i + 1
                    ii = [p[0]-ds[0], p[1]-ds[1], p[2]-ds[2]]
                    if ii[0]+ips[0] > s[0] or ii[1]+ips[1] > s[1] or ii[2]+ips[2] > s[2]:
                        print("skipped")
                        continue
                    if maskarry is not None:
                        if np.sum(maskarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])]) < 1:
                            print("skipped")
                            continue

                    patchimagearray = array_categorical[ii[2]:(ii[2]+ips[2]), ii[1]:(ii[1]+ips[1]), ii[0]:(ii[0]+ips[0]), :]
                    print(patchimagearray.shape)
                    print("saving cut to", out_path, end="...", flush=True)
                    createParentPath(out_path)
                    outfile = os.path.join(out_path,"image{}.mha".format(j))
                   
                    outimage = sitk.GetImageFromArray(patchimagearray.astype(np.uint8))
                    outimage.SetOrigin(image.GetOrigin())
                    outimage.SetSpacing(image.GetSpacing())
                    outimage.SetDirection(image.GetDirection())
                    sitk.WriteImage(outimage, outfile, True)
                    arr_path.append(outfile+"\n")

                    print("done")
                    j = j + 1
    else:
        i = 1
        j = 1
        arr_path = []
        for ix in range(bb[4], bb[5], step[2]):
            for iy in range(bb[2], bb[3], step[1]):
                for iz in range(bb[0], bb[1], step[0]):
                    p = [iz, iy, ix]
                    print("patch [{} / {}] : {}".format(i, num_totalpatches, p), end="...", flush=True)
                    i = i + 1
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
                    print(patchimagearray.shape)
                    print("saving cut to", out_path, end="...", flush=True)
                    createParentPath(out_path)
                    outfile = os.path.join(out_path,"image{}.mha".format(j))
                    
                    outimage = sitk.GetImageFromArray(patchimagearray)
                    outimage.SetOrigin(image.GetOrigin())
                    outimage.SetSpacing(image.GetSpacing())
                    outimage.SetDirection(image.GetDirection())
                    sitk.WriteImage(outimage, outfile, True)
                    arr_path.append(outfile+"\n")


                    print("done")
                    j = j + 1

    fo = open(out_txt, "w")
    fo.writelines(arr_path)
    fo.close()



model_path="444428.yml"

#coming from processing class  ipynb
list_all =  ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209']
c4_train =  ['000', '006', '041', '052', '059', '071', '075', '082', '087', '095', '096', '097', '100', '115', '117', '118', '122', '128', '131', '132', '140', '142', '163', '167', '169', '177', '204', '004', '009', '010', '011', '018', '022', '034', '036', '061', '062', '066', '083', '111', '148', '168', '175', '176', '198', '008', '012', '030', '055', '067', '073', '078', '092', '107', '108', '149', '172', '178', '179', '181', '186', '189', '192', '197', '199', '024', '025', '028', '029', '042', '088', '113', '114', '135', '166', '171', '001', '002', '003', '007', '013', '017', '019', '027', '032', '033', '035', '038', '040', '044', '045', '048', '049', '050', '053', '064', '068', '072', '076', '077', '080', '085', '086', '090', '099', '101', '105', '110', '121', '124', '126', '129', '130', '133', '136', '144', '146', '151', '158', '162', '164', '170', '173', '196', '201', '207']
c4_val =  ['159', '016', '203', '065', '153', '174', '047', '139', '157','070', '116', '127', '043', '106', '094', '089', '103']
c4_file = c4_train + c4_val
test_all_list =  ['210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299']
small96_forNormal =  ['005', '013', '015', '021', '026', '029', '043', '045', '060', '070', '074', '075', '079', '088', '094', '102', '104', '109', '112', '116', '119', '134', '139', '145', '147', '157', '161', '162', '165', '180', '182', '183', '184', '185', '187', '190', '191', '197', '199', '200', '203', '206', '208']
mask_train =  ['000', '006', '041', '052', '059', '071', '075', '082', '087', '095', '096', '097', '100', '115', '117', '118', '122', '128', '131', '132', '140', '142', '163', '167', '169', '177', '204', '004', '009', '010', '011', '018', '022', '034', '036', '061', '062', '066', '083', '111', '148', '168', '175', '176', '198', '008', '012', '030', '055', '067', '073', '078', '092', '107', '108', '149', '172', '178', '179', '181', '186', '189', '192', '197', '199', '024', '025', '028', '029', '042', '088', '113', '114', '135', '166', '171', '001', '002', '003', '007', '013', '017', '019', '027', '032', '033', '035', '038', '040', '044', '045', '048', '049', '050', '053', '064', '068', '072', '076', '077', '080', '085', '086', '090', '099', '101', '105', '110', '121', '124', '126', '129', '130', '133', '136', '144', '146', '151', '158', '162', '164', '170', '173', '196', '201', '207', '023', '054', '063', '120', '156', '191', '098', '057', '098', '150', '026', '069', '074', '082', '084', '102', '112', '119', '124', '125', '137', '141', '152', '156', '161', '178', '182', '183', '206', '041', '184', '005', '015', '154', '037', '021', '143', '208']
mask_val =  ['159', '016', '203', '065', '153', '174', '047', '139', '157', '070', '116', '127', '043', '106', '094', '089', '103', '081', '020', '057', '015', '058', '134', '200', '195', '193', '165', '188']
mask_file = mask_train+mask_val


hist_all = ['10','09','08','07','06','05','04','03','02','01','00','label']  
#hist_all = ['00']
for i in hist_all:
    outpath = "E:/Script_hist/patch/com30/hist"+i+"/"
    listpath = "E:/Script_hist/list/com30/hist"+i+"/"
    imagepath = "E:/hist_01/comp30/"
    createParentPath(outpath)
    createParentPath(listpath)
    for num in os.listdir(imagepath):

        filepath = os.path.join(imagepath,num)
        
        out_path = os.path.join(outpath,"ct"+num)
        out_txt = os.path.join(listpath,"ct"+num)
        
        if i == '00':
            do_extract(model_path,filepath,"c_ct.mha",out_path,out_txt,islabel=False,mask="c_mask_8.mha",stepscale=1)
        elif i == 'label':
            do_extract(model_path,filepath,"c_label_8.mha",out_path,out_txt,islabel=True,mask="c_mask_8.mha",stepscale=1)
        else:
            do_extract(model_path,filepath,"c_ct_hist"+i+".mha",out_path,out_txt,islabel=False,mask="c_mask_8.mha",stepscale=1)



"""
outpath = "E:/Script_hist/patch/com30-label/"
listpath = "E:/Script_hist/list/com30-label/"
createParentPath(outpath)
createParentPath(listpath)
imagepath = "E:/hist_01/comp30"
for num in os.listdir(imagepath):
    filepath = os.path.join(imagepath,num)

    out_path = os.path.join(outpath,"label"+num)
    out_txt = os.path.join(listpath,"llabel"+num)
    do_extract(model_path,filepath,"mask_8.mha",out_path,out_txt,islabel=True,mask="mask_8.mha",stepscale=1)





outpath = "E:/Script_hist/patch/label/"
listpath = "E:/Script_hist/list/label/"
createParentPath(outpath)
createParentPath(listpath)
imagepath = "E:/kits19_inter_processed/"
for num in mask_file:
    filepath = os.path.join(imagepath,"case_00"+num)

    out_path = os.path.join(outpath,"label"+num)
    out_txt = os.path.join(listpath,"llabel"+num)
    do_extract(model_path,filepath,"mask.mha",out_path,out_txt,islabel=True,mask="mask.mha",stepscale=1)


import shutil

for num in mask_file:

    filepath = os.path.join(imagepath,"case_00"+num)
    filepath2 = os.path.join(imagepath2,"case_00"+num)

    file_path = os.path.join(filepath2,"mask.mha")
    new_path = os.path.join(filepath,"mask.mha")

    path = shutil.copy(file_path, new_path)


for num in mask_val:
    
    filepath = os.path.join(imagepath,"case_00"+num)
    
     
    out_path = os.path.join(outpath,"label"+num)
    out_txt = os.path.join(listpath,"llabel"+num)
    do_extract(model_path,filepath,"mask.mha",out_path,out_txt,islabel=True,mask="mask.mha",stepscale=1)
    


    out_path = os.path.join(outpath,"ct"+num)
    out_txt = os.path.join(listpath,"ct"+num)
    do_extract(model_path,filepath,"ct_hist25.mha",out_path,out_txt,islabel=False,mask="mask.mha",stepscale=1)



    filepath = os.path.join(imagepath,"case_00"+num)

    out_path = os.path.join(outpath,"label"+num)
    out_txt = os.path.join(listpath,"llabel"+num)
    do_extract(model_path,filepath,"mask.mha",out_path,out_txt,islabel=True,mask="mask.mha",stepscale=1)



"""