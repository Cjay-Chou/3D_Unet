import os
from itertools import product
import numpy as np
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
from config import UConfig
import argparse


def Padding(image, patchsize, imagepatchsize, mirroring=False):
    padfilter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    padfilter.SetPadLowerBound(patchsize.tolist())
    padfilter.SetPadUpperBound(imagepatchsize.tolist())
    padded_image = padfilter.Execute(image)
    return padded_image


def createParentPath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print("path created")


def do_extract(input_shape,
               output_shape,
               image_file,  # abs path
               out_path,  # patch image folder
               out_txt,  # list folder
               islabel=False, mask=None, stepscale=1, step=None):
    # get patchsize x,y,z
    ps = np.array(output_shape)
    ips = np.array(input_shape)
    ds = ((ips - ps) / 2).astype(np.int)

    print("loading input image", image_file, end="...", flush=True)
    image = sitk.ReadImage(image_file)
    image_padded = Padding(image, ds, ips, mirroring=True)
    s = image_padded.GetSize()
    print("ps:{}, ips:{}, ds:{}, image_padded_s:{}".format(ps, ips, ds, s))

    if step is None:
        step = (ps / stepscale).astype(np.int)

    bb = None
    maskarry = None

    # GetSize() is z y x
    bb = (ds[0], ds[0] + image.GetSize()[0] - 1,
          ds[1], ds[1] + image.GetSize()[1] - 1,
          ds[2], ds[2] + image.GetSize()[2] - 1)
    print("bb, x,y,z", bb)

    if mask is not None:
        print("loading mask image", mask, end="...", flush=True)
        maskimage = sitk.ReadImage(mask)
        maskimage_padded = Padding(maskimage, ds, ips)
        print("maskimage_padded shape", maskimage_padded.GetSize())
        maskarry = sitk.GetArrayFromImage(maskimage_padded)
        print("done")

    totalpatches = [i for i in product(
        range(bb[4], bb[5], step[2]),
        range(bb[2], bb[3], step[1]),
        range(bb[0], bb[1], step[0]))]
    num_totalpatches = len(totalpatches)

    if islabel:
        image_padded_array = sitk.GetArrayFromImage(image_padded)
        array_categorical = to_categorical(image_padded_array)
        i = 1
        j = 1
        arr_path = []
        for ix in range(bb[4], bb[5], step[2]):
            for iy in range(bb[2], bb[3], step[1]):
                for iz in range(bb[0], bb[1], step[0]):
                    p = [iz, iy, ix]
                    print("patch [{} / {}] : {}".format(i, num_totalpatches, p), end="...", flush=True)
                    i = i + 1
                    ii = [p[0] - ds[0], p[1] - ds[1], p[2] - ds[2]]
                    if ii[0] + ips[0] > s[0] or ii[1] + ips[1] > s[1] or ii[2] + ips[2] > s[2]:
                        print("skipped")
                        continue
                    if maskarry is not None:
                        if np.sum(maskarry[p[2]:(p[2] + ps[2]), p[1]:(p[1] + ps[1]), p[0]:(p[0] + ps[0])]) < 1:
                            print("skipped")
                            continue

                    patchimagearray = array_categorical[ii[2]:(ii[2] + ips[2]), ii[1]:(ii[1] + ips[1]),
                                      ii[0]:(ii[0] + ips[0]), :]
                    print(patchimagearray.shape)
                    print("saving cut to", out_path, end="...", flush=True)
                    createParentPath(out_path)
                    outfile = os.path.join(out_path, "image{}.mha".format(j))

                    outimage = sitk.GetImageFromArray(patchimagearray.astype(np.uint8))
                    outimage.SetOrigin(image.GetOrigin())
                    outimage.SetSpacing(image.GetSpacing())
                    outimage.SetDirection(image.GetDirection())
                    sitk.WriteImage(outimage, outfile, True)
                    arr_path.append(outfile + "\n")

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
                    ii = [p[0] - ds[0], p[1] - ds[1], p[2] - ds[2]]
                    if ii[0] + ips[0] > s[0] or ii[1] + ips[1] > s[1] or ii[2] + ips[2] > s[2]:
                        print("skipped")
                        continue
                    if maskarry is not None:
                        if np.sum(maskarry[p[2]:(p[2] + ps[2]), p[1]:(p[1] + ps[1]), p[0]:(p[0] + ps[0])]) < 1:
                            print("skipped")
                            continue

                    patchimage = image_padded[ii[0]:(ii[0] + ips[0]), ii[1]:(ii[1] + ips[1]), ii[2]:(ii[2] + ips[2])]
                    patchimagearray = sitk.GetArrayFromImage(patchimage)
                    print(patchimagearray.shape)
                    print("saving cut to", out_path, end="...", flush=True)
                    createParentPath(out_path)
                    outfile = os.path.join(out_path, "image{}.mha".format(j))

                    outimage = sitk.GetImageFromArray(patchimagearray)
                    outimage.SetOrigin(image.GetOrigin())
                    outimage.SetSpacing(image.GetSpacing())
                    outimage.SetDirection(image.GetDirection())
                    sitk.WriteImage(outimage, outfile, True)
                    arr_path.append(outfile + "\n")

                    print("done")
                    j = j + 1

    fo = open(out_txt, "w")
    fo.writelines(arr_path)
    fo.close()


def all_extract(config_path):
    c = UConfig(config_path)
    ct_name = c.data_name
    is_label = ct_name == 'label.mha'
    extract_list = c.train_list + c.val_list
    patch_path = c.patch_path
    list_path = c.list_path
    createParentPath(list_path)
    for i in extract_list:
        abs_path = os.path.join(c.org_data_path, i)
        ct_abs_path = os.path.join(abs_path, ct_name)
        list_abs_path = os.path.join(list_path, i)
        out_path = os.path.join(patch_path, i)
        createParentPath(out_path)
        do_extract(c.input_shape,
                   c.output_shape,
                   ct_abs_path,
                   out_path,
                   list_abs_path,
                   islabel=is_label,
                   mask="c_mask_8.mha",
                   stepscale=c.step_scale)


def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build 3D_U_Net program')
    parser.add_argument("config_path", help="Input config file")
    parser.add_argument("-f", "--force", help="over write on old file if exist")
    parser.add_argument("--noLabel", help="not extract label file", dest='do_label', action='store_false')
    myargs = parser.parse_args()
    return myargs


if __name__ == '__main__':
    args = ParseArgs()
    all_extract(args.config_path)

