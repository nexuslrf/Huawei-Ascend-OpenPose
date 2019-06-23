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

# class WebcamLoader(object):
#     """docstring for VideoLoader"""
#     def __init__()

class VideoLoader(object):
    """docstring for VideoLoader"""
    def __init__(self, path, batchSize=1, queueSize=50, wh_rate = (1,1), sp=False, inDim=(256,256), dataLen=None):
        super(VideoLoader, self).__init__()
        self.path = path
        stream = cv2.VideoCapture(path)
        assert stream.isOpened(), 'Cannot capture source'

        self.batchSize = batchSize
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) if dataLen is None else dataLen
                # indicate the video info
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Usually width > height, only consider this case
        self.shape_dst = (self.frameSize[1] * wh_rate[0] // wh_rate[1], self.frameSize[1])
        self.oh = 0
        self.ow = (self.frameSize[0] - self.shape_dst[0]) // 2
        
        stream.release()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        self.sp = sp
        self.inDim = inDim

        # initialize the queue used to store frames read from
        # the video file
        if sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def videoLen(self):
        return self.datalen

    def start(self):
        if self.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'
        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i*self.batchSize, min((i +  1)*self.batchSize, self.datalen)):
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.Q.put((None, None, None))
                    print('All Frames Done!')
                    return
                # process and add the frame to the queue
                img_crop = frame[self.oh:self.oh+self.shape_dst[1], self.ow:self.ow+self.shape_dst[0]]
                # print(self.shape_dst, self.oh, self.ow)
                # print(img_crop.shape)
                img_k = cv2.resize(img_crop, self.inDim)
            
                img.append(img_k)
                orig_img.append(frame)
                im_name.append('{:05d}.jpg'.format(k))
                # print('get {:05d}.jpg'.format(k))

            while self.Q.full():
                time.sleep(2)
            
            self.Q.put((img, orig_img, im_name))

        self.Q.put((None, None, None))
        print('All Frames Done!')
        return

    def videoinfo(self):
        return (self.fourcc, self.fps, self.frameSize, self.shape_dst, self.ow, self.oh)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()


class saveFrame(object):
    """docstring for saveFrame"""
    def __init__(self, dataLoader, batchSize=1, savepath='./tmp_frames', queueSize=50, sp = False):
        super(saveFrame, self).__init__()
        self.dataLoader = dataLoader
        self.sp = sp
        self.savepath = savepath
        self.dev_path = '/home/HwHiAiUser/HIAI_DATANDMODELSET/workspace_mind_studio/pose_data'
        self.num_batches = dataLoader.num_batches
        self.inDim = dataLoader.inDim

        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        if sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def videoLen(self):
        return self.dataLoader.videoLen()

    def start(self):
        # start a thread to read frames from the file video stream
        if self.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("192.168.1.2",  22, "HwHiAiUser", "Mind@123")
        sftp = ssh.open_sftp()
        # keep looping the whole dataset
        for i in range(self.num_batches):
            img, orig_img, im_name = self.dataLoader.getitem()
            if img is None:
                self.Q.put((None, None, None, None))
                print("Save Done!")
                return
            
            batch_info = 'pose_data {lens} 0.0 0.0 0.0\n-1 0\n0 0\n1 {lens}\n'.format(lens=len(im_name))
            img_id = 0
            
            for k in range(len(img)):
                im = im_name[k]
                src = self.savepath + '/' + im
                dst = self.dev_path + '/dataIn/' + im
                cv2.imwrite(src, img[k])
                datasize = os.path.getsize(src)
                if datasize<10:
                    print('Bad Img!')
                    continue

                batch_info += f'{img_id} dataIn/{im} {self.inDim[0]} {self.inDim[1]} {datasize}\n'
                img_id += 1
                sftp.put(src, dst)
            
            batch_info += '2 0\n3 0\n4 0'

            while self.Q.full():
                time.sleep(2)

            self.Q.put((im_name, img, orig_img, batch_info))
        
        self.Q.put((None, None, None, None))
        print("Save Done!")
        return 

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()

class PostProcessor(object):
    def __init__(self, dataLoader, queueSize=100, remap=None, vis=False, nobg=False, sp=False):
        self.openpose = openpose()
        self.sp = sp
        self.dataLoader = dataLoader
        self.num_batches = dataLoader.num_batches
        self.vis = vis
        self.inDim = dataLoader.inDim 
        self.shape_dst = self.inDim
        self.oh, self.ow = 0,0
        self.remap = False
        self.nobg = nobg
        # self.json = json
        if remap is not None:
            self.remap = True
            self.shape_dst, self.ow, self.oh = remap
            # print(self.shape_dst, self.ow, self.oh)
            
        if sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)
        
    def start(self):
        if self.sp:
            if self.vis:
                t = Thread(target=self.update_vis, args=())
            else:
                t = Thread(target=self.update_pt, args=())
            t.daemon = True
            t.start()
        else:
            if self.vis:
                p = mp.Process(target=self.update_vis, args=())
            else:
                p = mp.Process(target=self.update_pt, args=())
            p.daemon = True
            p.start()
        return self

    def update_pt(self):
        for i in range(self.num_batches):
            (l1, l2, im_name, img, orig_img) = self.dataLoader.getitem()
            # print('Sb:', im_name)
            with_img = True
            if orig_img is None:
                with_img = False
            if l1 is None:
                self.Q.put((None, None, None, None))
                print('post done!')
                return
            for k in range(len(l1)):
                # try:
                json_name = None
                if im_name is not None:
                    json_name = "json_out/json/img_0"+im_name[k][:-4]+"_keypoints.json"
                poses = self.openpose.get_list(l1[k],l2[k], self.shape_dst, self.inDim, json_name=json_name)
                # except:
                    # continue
                while self.Q.full():
                    time.sleep(2)
                if with_img:
                    self.Q.put((poses, im_name[k], img[k], orig_img[k]))
                else:
                    self.Q.put((poses, None, None, None))
        
        self.Q.put((None, None, None, None))
        print('post done!')
        return

    def update_vis(self):
        for i in range(self.num_batches):
            (l1, l2, im_name, img, orig_img) = self.dataLoader.getitem()
            # print('Sb:', im_name)
            with_img = True
            if orig_img is None:
                with_img = False
            if l1 is None:
                self.Q.put((None, None, None, None))
                print('post done!')
                return
            for k in range(len(l1)):
                if not with_img:
                    out_img = np.zeros((self.shape_dst[1], self.shape_dst[0]+2*self.ow, 3), dtype=np.uint8)
                else:
                    out_img = orig_img[k]

                if self.remap:
                    img_crop = out_img[self.oh:self.oh+self.shape_dst[1], self.ow:self.ow+self.shape_dst[0]]
                else:
                    if self.nobg:
                        img_crop = np.zeros((*self.inDim[::-1],3), dtype=np.uint8)
                    else:
                        img_crop = img[k]
                try:
                    img_render = self.openpose.visualization(img_crop, l1[k], l2[k], self.shape_dst, self.inDim)
                    if self.remap:
                        out_img[self.oh:self.oh+self.shape_dst[1], self.ow:self.ow+self.shape_dst[0]] = img_render
                except:
                    continue
                while self.Q.full():
                    time.sleep(2)
                if with_img:
                    self.Q.put((img_render, img[k], out_img, im_name[k]))
                else:
                    self.Q.put((img_render, None, out_img, None))
            
        self.Q.put((None, None, None, None))
        print('post done!')
        return
    
    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()
    
        

class TxtCollector(object):
    """docstring for TxtCollector"""
    def __init__(self, num_frames, inDim, offset=0, queueSize=50, sp=False):
        super(TxtCollector, self).__init__()
        self.sp = sp
        self.offset = offset
        self.local_path = './raw_out'
        self.inDim = inDim
        self.num_batches = num_frames
        self.outDim = (self.inDim[0]//8, self.inDim[1]//8)
        self.num_frames = num_frames
        if sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        if self.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

        
    def update(self):
        # keep looping the whole dataset
        for i in range(self.offset, self.offset+self.num_frames):
            tmp = self.local_path+'/SaveFilePostProcess_1/davinci_{:05d}_output_0_Mconv7_stage6_L1_0.txt'.format(i)
            if not os.path.exists(tmp):
                self.Q.put((None, None, None, None, None))
                return

            # print(i)
            l1 = np.loadtxt(self.local_path+'/SaveFilePostProcess_1/davinci_{:05d}_output_0_Mconv7_stage6_L1_0.txt'.format(i)).reshape(1, 38, *self.outDim)
            l2 = np.loadtxt(self.local_path+'/SaveFilePostProcess_1/davinci_{:05d}_output_1_Mconv7_stage6_L2_0.txt'.format(i)).reshape(1, 19, *self.outDim)

            while self.Q.full():
                time.sleep(2)

            self.Q.put(([l1], [l2], ["{:05d}.jpg".format(i)], None, None))

        self.Q.put((None, None, None, None, None))
        return
            # except:
            #     print('Exception output, skip~')
    
    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()

def main():
    M = VideoLoader('kunkun_nmsl.mp4',sp=False).start()
    # print(M.videoinfo())
    S = saveFrame(M, sp=False).start()
    for i in range(30):
        S.getitem()

if __name__ == '__main__':
    main()
