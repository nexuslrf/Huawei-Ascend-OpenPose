'''
For Video Demo

'''
import os
import paramiko
import cv2
import numpy as np
from openpose import openpose
import multiprocessing as mp
from queue import Queue, LifoQueue
import os
import sys
import time
from tqdm import tqdm
from threading import Thread

from dataloader import VideoLoader, saveFrame, PostProcessor
from ascendHost import AscendHost, DataCollector

def main():
    outSize = (64,64)
    sp = False
    video_loader = VideoLoader("kunkun_nmsl_Trim.mp4", batchSize=32, wh_rate=(1,1), inDim=outSize, sp=sp)

    (fourcc, fps, frameSize, shape_dst, ow, oh) = video_loader.videoinfo()
    # fps = 10
    # frameSize = outSize
    print(fourcc, fps, frameSize, shape_dst, ow, oh)
    num_frames = video_loader.videoLen()
    save_path = 'out_video/kunkun_nmsl_2.avi'    
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    
    stream = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
    assert stream.isOpened(), 'Cannot open video for writing'

    video_loader.start()
    save_loader = saveFrame(video_loader, sp=sp).start()
    ascend_loader = AscendHost(save_loader, sp=sp).start()
    collect_loader = DataCollector(ascend_loader, num_job=8, sp=sp).start()
    post_processor = PostProcessor(collect_loader, remap=(shape_dst, ow, oh), vis=True, nobg=True, sp=sp).start()
    # post_processor = PostProcessor(collect_loader, remap=(shape_dst, ow, oh), vis=True, sp=sp).start()

    cnt = 0
    while True:
        cnt += 1
        start = time.time()
        (img_render, img, orig_img, im_name) = post_processor.getitem()
        if orig_img is None:
            print('All done!')
            break
        
        # orig_img = img_render

        cv2.imshow("Demo", orig_img)
        cv2.waitKey(90)
        stream.write(orig_img)
        interval = time.time() - start
        print("Time: {:.4f}\t{:04d}/{:04d}".format(interval, cnt, num_frames))
        # sys.stdout.flush()

    stream.release()


if __name__ == "__main__":
    main()
