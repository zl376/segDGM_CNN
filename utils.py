import nibabel as nib
import numpy as np
import itertools
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from matplotlib import pyplot as plt

precision_global = 'float32'

# General utils for reading and saving data
def get_filename(set_name, case_idx, input_name, loc='datasets') :
    pattern = '{0}/{1}/{2}_{3}.nii.gz'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx) :
    return 'training' if case_idx < 16 else 'testing'

def read_data(case_idx, input_name, loc='datasets') :
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)

    return nib.load(image_path)

def read_vol(case_idx, input_name, loc='datasets') :
    image_data = read_data(case_idx, input_name, loc)
    return image_data.get_data()[:, :, :]

def save_vol(segmentation, case_idx, loc='results') :
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'QSM')
    
    filename = get_filename(set_name, case_idx, 'label', loc)
    #nib.save(nib.Nifti1Image(segmentation.astype('uint8'), input_image_data.affine, input_image_data.header), filename)
    nib.save(nib.Nifti1Image(segmentation.astype('uint8'), None, input_image_data.header), filename)


# Data preparation utils
def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)
    
    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set(QSM_vols, label_vols, extraction_step=(9, 9, 9), patch_shape=(27, 27, 27), predictor_shape=(9, 9, 9)) :
    #patch_shape = (27, 27, 27)
    #label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]
    label_selector = [slice(None)] + [slice(int((patch_shape[i]-predictor_shape[i])/2), int((patch_shape[i]-predictor_shape[i])/2+predictor_shape[i])) for i in range(3)]
    
    num_classes = len(np.unique(label_vols))

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 1) + patch_shape, dtype=precision_global)
    y = np.zeros((0, predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)
    for idx in range(len(QSM_vols)) :
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)
        
        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 1) + patch_shape, dtype=precision_global)))
        y = np.vstack((y, np.zeros((len(label_patches), predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)))

        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i, : ,: ,:], num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        QSM_train = extract_patches(QSM_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = QSM_train[valid_idxs]
        del QSM_train

    return x, y

def build_set_mask(QSM_vols, label_vols, extraction_step=(9, 9, 9), mask_vols=np.array([]), patch_shape=(27, 27, 27), predictor_shape=(9, 9, 9)) :
    #patch_shape = (27, 27, 27)
    #label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]
    label_selector = [slice(None)] + [slice(int((patch_shape[i]-predictor_shape[i])/2), int((patch_shape[i]-predictor_shape[i])/2+predictor_shape[i])) for i in range(3)]
    
    num_classes = len(np.unique(label_vols))

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 1) + patch_shape, dtype=precision_global)
    y = np.zeros((0, predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)
    for idx in range(len(QSM_vols)) :
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]
                
        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)
        
        if (0,) != mask_vols.shape:
            mask_patches = extract_patches(mask_vols[idx], patch_shape, extraction_step)
            mask_patches = mask_patches[label_selector]
            valid_idxs = np.where((np.sum(label_patches, axis=(1, 2, 3)) != 0) | (np.sum(mask_patches, axis=(1, 2, 3)) != 0))
        
        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 1) + patch_shape, dtype=precision_global)))
        y = np.vstack((y, np.zeros((len(label_patches), predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)))

        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i, : ,: ,:], num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        QSM_train = extract_patches(QSM_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = QSM_train[valid_idxs]
        del QSM_train

    return x, y

# Reconstruction utils
def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    #poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]
    poss_shape = [patch_shape[i+1] * ((expected_shape[i]-18) // patch_shape[i+1]) + 18 for i in range(ndims-1)]

    #idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]
    idxs = [range(9, poss_shape[i] - 9, patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape, dtype=precision_global)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

# Utils for plotting
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    
    if(ims.ndim == 2):
        ims = np.tile(ims, (1,1,1));
    
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = np.tile(ims[:,:,:,np.newaxis], (1,1,1,3));
            
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')