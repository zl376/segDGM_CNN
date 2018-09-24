# Calculate statistics within ROI
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2018-09-24
import sys, getopt
import os

import nibabel as nib
import numpy as np
import pandas as pd
from utils import precision_global



def main(argv):
    
    ## ============  Determine input argumentts =============== ##
    # Default values
    filename_img = 'QSM.nii.gz'
    filename_roi = 'label.nii.gz'
    filename_out = 'stats.csv'
    verbose = False
    
    try:
        opts, args = getopt.getopt(argv,"hi:r:o:v",[])
    except getopt.GetoptError:
        print('calc_roi.py -i <filename_img> -r <filename_roi> -o <filename_out> \
                           [-v | verbose]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('calc_roi.py -i <filename_img> -r <filename_roi> -o <filename_out>')
            sys.exit()
        elif opt == '-i':
            filename_img = arg
        elif opt == '-r':
            filename_roi = arg
        elif opt == '-o':
            filename_out = arg
        elif opt == '-v':
            verbose = True
            


    ## =================== Parameter setting ===================== ##
    num_classes = 11      # 10 DGM + 1 Background
    labels = ['Red Nucleus (L)',
              'Red Nucleus (R)',
              'Substantia Nigra (L)',
              'Substantia Nigra (R)',
              'Globus Pallidus (L)',
              'Globus Pallidus (R)',
              'Putamen (L)',
              'Putamen (R)',
              'Caudate Nucleus (L)',
              'Caudate Nucleus (R)']
    labels_abbr = ['RN(L)',
                   'RN(R)',
                   'SN(L)',
                   'SN(R)',
                   'GP(L)',
                   'GP(R)',
                   'PU(L)',
                   'PU(R)',
                   'CN(L)',
                   'CN(R)']

    
    
    ## ==================== Load QSM (nifti) ===================== ##
    img = nib.load(filename_img)
    matrix_size = img.shape
    voxel_size = img.header.get_zooms()
    QSM = img.get_data().astype(precision_global)
    # remove background
    QSM[QSM < -10000] = 0.0
    
    
    
    ## ==================== Load ROI (nifti) ===================== ##
    roi = nib.load(filename_roi)
    assert matrix_size == roi.shape, 'Matrix size should match for image and ROI'
    ROI = roi.get_data().astype(np.int)

    
    
    ## ======================= Measure ROI ======================= ##
    df = pd.DataFrame(index=labels_abbr, columns=['Mean', 'Std'])
    for i, label in zip(np.arange(2, 2+num_classes-1), labels_abbr):
        df.loc[label, 'Mean'] = np.mean(QSM[ROI == i])
        df.loc[label, 'Std'] = np.std(QSM[ROI == i])        
    if verbose:
        print(df)
    
    
    
    ## ======================= Save stats ======================== ##
    df.to_csv(filename_out)
    
    
    sys.exit(0)
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
