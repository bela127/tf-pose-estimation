import argparse
import logging
import time
import glob
import os
import sys

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

#from lifting.prob_model import Prob3dPose
#from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='./images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='432x368',
    help='if provided, resize images before they are processed. '
         'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
    help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    image_files = []
    path = args.folder
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image_files.append(os.path.join(path,f))

    all_humans = dict()
    for i, file in enumerate(image_files):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % file)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (file, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            a = fig.add_subplot(2, 2, 1)
            a.set_title('Result')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # show network output
            a = fig.add_subplot(2, 2, 2)
            plt.imshow(bgimg, alpha=0.5)
            tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            tmp2 = e.pafMat.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            a = fig.add_subplot(2, 2, 3)
            a.set_title('Vectormap-x')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 2, 4)
            a.set_title('Vectormap-y')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()
            plt.show()
            plt.savefig(os.path.join('infer', file.replace(args.folder, 'infer_')))
            plt.close()
            plt.clf()
        except Exception as e:
            logger.warning('matplitlib error, %s' % e)
            cv2.imshow('result', image)
            cv2.waitKey()