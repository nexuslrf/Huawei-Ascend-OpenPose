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
    save_path = './video_cap'
    video_name = 'webcam.avi'
    fourcc= cv2.VideoWriter_fourcc(*'XVID')
    fps = 24
    input("Press Any Key to Open WebCam (press 'q' to stop): ")
    cap = cv2.VideoCapture(0)
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(save_path+'/'+video_name, fourcc, fps, frameSize)
    for i in range(42):
        cap.read()
    start_write = False
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        if start_write:
            writer.write(frame)
            frame_cnt += 1

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print('Start Rec!')
            start_write = True
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Start Processing!")
    outSize = (128,128)
    sp = False

    video_loader = VideoLoader(save_path+'/'+video_name, batchSize=32, wh_rate=(1,1), inDim=outSize, sp=sp, dataLen=frame_cnt)

    (fourcc, fps, frameSize, shape_dst, ow, oh) = video_loader.videoinfo()
    # fps = 10
    print(fourcc, fps, frameSize, shape_dst, ow, oh, video_loader.videoLen())
    num_frames = video_loader.videoLen()
    out_path = 'out_video/'+ video_name    
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    
    stream = cv2.VideoWriter(out_path, fourcc, fps, frameSize)
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
        
        orig_img = img_render

        cv2.imshow("Demo", orig_img)
        cv2.waitKey(160)
        stream.write(orig_img)
        interval = time.time() - start
        print("Time: {:.4f}\t{:04d}/{:04d}".format(interval, cnt, num_frames))
        # sys.stdout.flush()

    stream.release()


if __name__ == "__main__":
    main()
