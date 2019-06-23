'''
For Json Demo

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
import tarfile
from dataloader import VideoLoader, saveFrame, PostProcessor, TxtCollector
from ascendHost import AscendHost, DataCollector

def main():
    num_frames = 100
    offset = 330
    inDim = (64, 64)
    shape_dst = (480, 480)
    ow, oh = 187, 0

    fh = open('json_out/size.txt', 'w')
    fh.write(f"{shape_dst[0]}\n{shape_dst[1]}\n")
    fh.close()

    collect_loader = TxtCollector(num_frames, inDim, offset=offset).start()
    post_processor = PostProcessor(collect_loader, remap=(shape_dst,ow,oh), nobg=True, vis=False).start()

    for i in range(num_frames):
        a = post_processor.getitem()
        # cv2.imwrite(f'{i+333}.jpg', a[0])
        # if a[0] is not None:
        #     cv2.imshow('Demo', a[2])
        #     cv2.waitKey(80)
        # else:
            # break
    tar = tarfile.open('json_out.tar', 'w')
    json_list = os.listdir('./json_out/json')
    json_list.sort()
    for f in json_list:
        tar.add('json_out/json/'+f)
    tar.add('json_out/size.txt')
    tar.close()   

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("202.120.39.158",  9989, "ruofan", "284613")
    print("ssh connect!")
    stransport = ssh.get_transport()
    sftp = ssh.open_sftp()

    sftp.put('json_out.tar', 
        '/data2/ruofan/MyFiles/3d-pose-baseline-vmd/json_out.tar'
        )

    print("Start Inference!")
    start_t = time.time()
    schan = stransport.open_session()
    stdin, stdout, stderr = ssh.exec_command(
        'bash ./start_vmd.sh'
    )
    line = stdout.readline()
    while line:
        print('ssh: ',line)
        line = stdout.readline()
    print("ssh error: ", stderr.readlines())

    print('Finish inference!', time.time() - start_t)

    sftp.get('/data2/ruofan/MyFiles/3d-pose-baseline-vmd/json_out/pos.txt',
        'pos.txt'
    )
    sftp.get('/data2/ruofan/MyFiles/3d-pose-baseline-vmd/json_out/movie_smoothing.gif',
        'movie_smoothing.gif'
    )
    print('Ok!')

if __name__ == "__main__":
    main()
