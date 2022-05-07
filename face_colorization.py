'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import numpy as np
from PIL import Image
import __init_paths
from face_model.face_gan import FaceGAN

class FaceColorization(object):
    def __init__(self, base_dir='./', size=1024, out_size=None, model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facegan = FaceGAN(base_dir, size, out_size, model, channel_multiplier, narrow, key, device=device)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray):
        # colorize the face
        out = self.facegan.process(gray)

        return out
        

if __name__=='__main__':
    model = {'name':'GPEN-1024-Color', 'size':1024}
    
    indir = 'examples/grays'
    outdir = 'examples/couts'
    os.makedirs(outdir, exist_ok=True)

    facecolorizer = FaceColorization(size=model['size'], model=model['name'])

    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        grayf = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        dim = (grayf.shape[1], grayf.shape[0])
        grayf = cv2.cvtColor(grayf, cv2.COLOR_GRAY2BGR) # channel: 1->3

        colorf = facecolorizer.process(grayf)
        
        grayf = cv2.resize(grayf, (0,0))
        colorf = cv2.resize(colorf, dim)

        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'.jpg'), grayf)
        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'_COMP.jpg'), np.hstack((grayf, colorf)))
        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1])+'_GPEN.jpg'), colorf)
        
        if n%10==0: print(n, file)
        
