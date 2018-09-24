import sys, getopt
import os

import nibabel as nib
import numpy as np
from scipy import ndimage
from keras import backend as K

from utils import *
from model_FCNN_aniso_loc import generate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 





def main(argv):
    
    ## ============  Determine input argumentts =============== ##
    # Default values
    filename_in = 'QSM.nii.gz'
    filename_out = 'label.nii.gz'
    flag_flip = True
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:f",[])
    except getopt.GetoptError:
        print('segDGM_3DCNN.py -i <filename_in> -o <filename_out>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('segDGM_3DCNN.py -i <filename_in> -o <filename_out>')
            sys.exit()
        elif opt == '-i':
            filename_in = arg
        elif opt == '-o':
            filename_out = arg
        elif opt == '-f':
            print('Do not flip in slice direction.')
            flag_flip = False
            
            


    ## =================== Parameter setting ===================== ##
    num_classes = 11
    num_channel = 1
    num_dim_loc = 3
    matrix_size = (160, 220, 48)
    voxel_size = (1, 1, 3)

    # window level
    window_QSM = (-250, 250)
    window_XYZ = (-50, 50)

    # size (kernel, patch, etc.)
    segment_size = (27, 27, 9)
    core_size = (9, 9, 3)
    extraction_step = (9, 9, 3)
    
    # class mapper
    class_mapper = {0:0}
    class_mapper.update({ i+1:i for i in range(1, 1+10) })
    class_mapper_inv = {0:0}
    class_mapper_inv.update({ i+1:j+1 for i,j in enumerate(range(1, 1+10)) })

    # model weights
    fn_model = os.path.dirname(__file__) + '/models/weights_optimal.h5'
    
    
    
    
    
    ## ==================== Load QSM (nifti) ===================== ##
    img_in = nib.load(filename_in)
    matrix_size_raw = img_in.shape
    voxel_size_raw = img_in.header.get_zooms()

    QSM_raw = img_in.get_data().astype(precision_global)
    
    # IMPORTANT:
    #    flip in slice direction for nifti outcomed from dcm2nii(x), (default)
    if flag_flip:
        QSM_raw = np.flip(QSM_raw, axis=-1)

    # get mask
    MASK_raw = (QSM_raw >= -10000).astype(precision_global)

    # remove background
    QSM_raw[QSM_raw < -10000] = 0.0
    
    
    
    
    
    ## ====================== Pre-process ======================== ##
    # resize and crop
    QSM = resize_and_crop(QSM_raw, matrix_size_raw, voxel_size_raw, matrix_size, voxel_size)
    MASK = resize_and_crop(MASK_raw, matrix_size_raw, voxel_size_raw, matrix_size, voxel_size)

    # scale to 0 ~ 255
    QSM = scale(QSM, window_QSM)   
    
    # construct 3D coordinates
    idxs = [ np.arange(1,matrix_size[i]+1) * voxel_size[i] for i in (1,0,2) ]
    [Y,X,Z] = np.meshgrid(*idxs);

    X_cen = np.sum(X[MASK>0])/np.sum(MASK);
    Y_cen = np.sum(Y[MASK>0])/np.sum(MASK);
    Z_cen = np.sum(Z[MASK>0])/np.sum(MASK);
    X = X - X_cen;
    Y = Y - Y_cen;
    Z = Z - Z_cen;
    X = scale(X, window_XYZ);
    Y = scale(Y, window_XYZ);
    Z = scale(Z, window_XYZ);    
    
    # prepare
    data_in = np.reshape(QSM, (1,1) + matrix_size)
    if num_dim_loc > 0:
        aux_test = np.reshape(np.stack((X,Y,Z), axis=0), (1,3) + matrix_size)
        data_in = np.concatenate((data_in, aux_test), axis = 1)
    
    # normalisation (zero mean and unit variance)
    mean_norm = 127.0
    std_norm = 128.0
    data_in = (data_in - mean_norm) / std_norm
    
    # number of patches
    n_patch = extract_patches(np.zeros(matrix_size), patch_shape=segment_size, extraction_step=extraction_step).shape[0]
    
    
    
    ## ====================== Setup model ======================== ##
    model = generate_model(num_classes, num_channel, segment_size, core_size, num_dim_loc)
    model.load_weights(fn_model)
    
    # extract patch
    data_in = data_in[0, :, :, :, :]

    tmp = np.zeros((n_patch, num_channel+num_dim_loc,) + segment_size, dtype=precision_global)
    for i_channel in range(num_channel):
        tmp[:, i_channel, :, :, :] = extract_patches(data_in[i_channel], patch_shape=segment_size, extraction_step=extraction_step)

    patches, patches_aux = extract_aux(tmp, core_size, num_dim_loc)   
    
    # predict
    pred = model.predict([patches, patches_aux], verbose=1)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes),) + core_size)
    segmentation = reconstruct_volume(pred_classes, matrix_size)
    
    
    
    
    ## ======================= Post-process ====================== ##
    # pick the largest connected component for each class
    tmp = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    for class_idx in class_mapper_inv :
        mask = (segmentation == class_idx)

        if class_idx != 0 and mask.sum() > 0:
            labeled_mask, num_cc = ndimage.label(mask)
            largest_cc_mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))

            tmp[largest_cc_mask == 1] = class_idx

    segmentation = tmp
    
    # reverse mapping
    tmp = np.copy(segmentation)
    for class_idx in class_mapper_inv:
        segmentation[tmp == class_idx] = class_mapper_inv[class_idx]
    del tmp
    
    # resize and crop (recover)
    segmentation_raw = resize_and_crop(segmentation, matrix_size, voxel_size, matrix_size_raw, voxel_size_raw)
    
    # IMPORTANT:
    #    flip (back) in slice direction for nifti outcomed from dcm2nii(x), (default)
    if flag_flip:
        segmentation_raw = np.flip(segmentation_raw, axis=-1)
    
    
    ## ================= Save segmentation (nifti) =============== ##
    nib.save(nib.Nifti1Image(segmentation_raw.astype('uint8'), None, img_in.header), filename_out)
    
    
    
    
    ## ======================== House keeping ==================== ##
    img_in.uncache()
    
    
    
    
    
    print('Segmentation successful.')
    sys.exit(0)
    
    
    
def main_dbg(filename_in, filename_out, flag_flip = True):
    
    ## ============  Determine input argumentts =============== ##
    # Default values

    


    ## =================== Parameter setting ===================== ##
    num_classes = 11
    num_channel = 1
    num_dim_loc = 3
    matrix_size = (160, 220, 48)
    voxel_size = (1, 1, 3)

    # window level
    window_QSM = (-250, 250)
    window_XYZ = (-50, 50)

    # size (kernel, patch, etc.)
    segment_size = (27, 27, 9)
    core_size = (9, 9, 3)
    extraction_step = (9, 9, 3)
    
    # class mapper
    class_mapper = {0:0}
    class_mapper.update({ i+1:i for i in range(1, 1+10) })
    class_mapper_inv = {0:0}
    class_mapper_inv.update({ i+1:j+1 for i,j in enumerate(range(1, 1+10)) })

    # model weights
    fn_model = 'models/weights_optimal.h5'
    
    
    
    
    
    ## ==================== Load QSM (nifti) ===================== ##
    img_in = nib.load(filename_in)
    matrix_size_raw = img_in.shape
    voxel_size_raw = img_in.header.get_zooms()

    QSM_raw = img_in.get_data().astype(precision_global)
    
    # IMPORTANT:
    #    flip in slice direction for nifti outcomed from dcm2nii(x), (default)
    if flag_flip:
        QSM_raw = np.flip(QSM_raw, axis=-1)

    # get mask
    MASK_raw = (QSM_raw >= -10000).astype(precision_global)

    # remove background
    QSM_raw[QSM_raw < -10000] = 0.0
    
    
    
    
    
    ## ====================== Pre-process ======================== ##
    # resize and crop
    QSM = resize_and_crop(QSM_raw, matrix_size_raw, voxel_size_raw, matrix_size, voxel_size)
    MASK = resize_and_crop(MASK_raw, matrix_size_raw, voxel_size_raw, matrix_size, voxel_size)

    # scale to 0 ~ 255
    QSM = scale(QSM, window_QSM)   
    
    # construct 3D coordinates
    idxs = [ np.arange(1,matrix_size[i]+1) * voxel_size[i] for i in (1,0,2) ]
    [Y,X,Z] = np.meshgrid(*idxs);

    X_cen = np.sum(X[MASK>0])/np.sum(MASK);
    Y_cen = np.sum(Y[MASK>0])/np.sum(MASK);
    Z_cen = np.sum(Z[MASK>0])/np.sum(MASK);
    X = X - X_cen;
    Y = Y - Y_cen;
    Z = Z - Z_cen;
    X = scale(X, window_XYZ);
    Y = scale(Y, window_XYZ);
    Z = scale(Z, window_XYZ);    
    
    # prepare
    data_in = np.reshape(QSM, (1,1) + matrix_size)
    if num_dim_loc > 0:
        aux_test = np.reshape(np.stack((X,Y,Z), axis=0), (1,3) + matrix_size)
        data_in = np.concatenate((data_in, aux_test), axis = 1)
    
    # normalisation (zero mean and unit variance)
    mean_norm = 127.0
    std_norm = 128.0
    data_in = (data_in - mean_norm) / std_norm
    
    # number of patches
    n_patch = extract_patches(np.zeros(matrix_size), patch_shape=segment_size, extraction_step=extraction_step).shape[0]
    
    
    
    ## ====================== Setup model ======================== ##
    model = generate_model(num_classes, num_channel, segment_size, core_size, num_dim_loc)
    model.load_weights(fn_model)
    
    # extract patch
    data_in = data_in[0, :, :, :, :]

    tmp = np.zeros((n_patch, num_channel+num_dim_loc,) + segment_size, dtype=precision_global)
    for i_channel in range(num_channel):
        tmp[:, i_channel, :, :, :] = extract_patches(data_in[i_channel], patch_shape=segment_size, extraction_step=extraction_step)

    patches, patches_aux = extract_aux(tmp, core_size, num_dim_loc)   
    
    # predict
    pred = model.predict([patches, patches_aux], verbose=1)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes),) + core_size)
    segmentation = reconstruct_volume(pred_classes, matrix_size)
    
    
    
    
    ## ======================= Post-process ====================== ##
    # pick the largest connected component for each class
    tmp = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    for class_idx in class_mapper_inv :
        mask = (segmentation == class_idx)

        if class_idx != 0 and mask.sum() > 0:
            labeled_mask, num_cc = ndimage.label(mask)
            largest_cc_mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))

            tmp[largest_cc_mask == 1] = class_idx

    segmentation = tmp
    
    # reverse mapping
    tmp = np.copy(segmentation)
    for class_idx in class_mapper_inv:
        segmentation[tmp == class_idx] = class_mapper_inv[class_idx]
    del tmp
    
    # resize and crop (recover)
    segmentation_raw = resize_and_crop(segmentation, matrix_size, voxel_size, matrix_size_raw, voxel_size_raw)
    
    # IMPORTANT:
    #    flip (back) in slice direction for nifti outcomed from dcm2nii(x), (default)
    if flag_flip:
        segmentation_raw = np.flip(segmentation_raw, axis=-1)
    
    
    ## ================= Save segmentation (nifti) =============== ##
    nib.save(nib.Nifti1Image(segmentation_raw.astype('uint8'), None, img_in.header), filename_out)
    
    
    
    
    ## ======================== House keeping ==================== ##
    img_in.uncache()
    
    
    
    
    
    print('Segmentation successful.')
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
