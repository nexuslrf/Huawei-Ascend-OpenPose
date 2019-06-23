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
import tarfile
from tqdm import tqdm
from threading import Thread


def untar_rm(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)
    t.close()
    os.remove(fname)


class AscendHost(object):
    """docstring for AscendHost"""
    def __init__(self, dataLoader, queueSize=50, sp=False):
        super(AscendHost, self).__init__()
        self.dataLoader = dataLoader
        self.num_batches = dataLoader.num_batches
        self.inDim = dataLoader.inDim

        self.module_name = 'openpose_{}_{}_bs32'.format(self.inDim[0], self.inDim[1])
        self.data_path = '/home/HwHiAiUser/HIAI_DATANDMODELSET/workspace_mind_studio/pose_data'
        self.cmd_path = '/home/HwHiAiUser/HIAI_PROJECTS/workspace_mind_studio/{}/out'.format(self.module_name)
        self.out_path = '/home/HwHiAiUser/HIAI_PROJECTS/workspace_mind_studio/{}/out/result_files/0'.format(self.module_name)
        self.sp = sp

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
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("192.168.1.2",  22, "HwHiAiUser", "Mind@123")
        stransport = ssh.get_transport()
        sftp = ssh.open_sftp()
        # keep looping the whole dataset
        for i in range(self.num_batches):
            info_path = 'batch_data.info'
            data_info = open(info_path, 'w')
            
            l1 = []
            l2 = []
    
            (im_name, img, orig_img, batch_info) = self.dataLoader.getitem()
            # print('Sb:', im_name)
            if orig_img is None:
                self.Q.put((None, None, None, None))
                schan = stransport.open_session()
                schan.exec_command(f'rm {self.data_path}/dataIn/*')
                while not schan.exit_status_ready():
                    pass
                return
            
            data_info.write(batch_info)
            data_info.close()

            sftp.put(info_path, self.data_path+'/.pose_data_data.info')
            
            start_t = time.time()
            schan = stransport.open_session()
            schan.exec_command(
                f'cd {self.cmd_path}; ./workspace_mind_studio_{self.module_name};' +
                f'cd {self.out_path}; tar -cvf raw_out_{i}.tar ./SaveFilePostProcess_1/*; rm ./SaveFilePostProcess_1/*'
            )

            while not schan.exit_status_ready():
                pass
            print(time.time()-start_t, "Finish One Training Pass")
            while self.Q.full():
                time.sleep(2)

            self.Q.put((im_name, img, orig_img, f'raw_out_{i}.tar'))
    
    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()


class DataCollector(object):
    """docstring for DataCollector"""
    def __init__(self, dataLoader, queueSize=50, num_job=1, sp=False):
        super(DataCollector, self).__init__()
        self.dataLoader = dataLoader
        self.num_batches = dataLoader.num_batches
        self.module_name = dataLoader.module_name
        self.sp = sp
        self.out_path = '/home/HwHiAiUser/HIAI_PROJECTS/workspace_mind_studio/{}/out/result_files/0'.format(self.module_name)
        self.local_path = './raw_out'
        self.inDim = dataLoader.inDim
        self.outDim = (self.inDim[0]//8, self.inDim[1]//8)
        self.num_job = num_job
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
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("192.168.1.2",  22, "HwHiAiUser", "Mind@123")
        sftp = ssh.open_sftp()
        # keep looping the whole dataset
        for i in range(self.num_batches):                
            (im_name, img, orig_img, raw_name) = self.dataLoader.getitem()
            if orig_img is None:
                self.Q.put((None, None, None, None, None))
                return
            
            sftp.get(self.out_path+'/'+raw_name, self.local_path+'/'+raw_name)
            sftp.remove(self.out_path+'/'+raw_name)
            untar_rm(self.local_path+'/'+raw_name, self.local_path)

            l1 = [None]*len(im_name)
            l2 = [None]*len(im_name)

            start_t = time.time()
            threads = []
            for k in range(self.num_job):
                t = Thread(target=self.collect_job, args=(k, self.num_job, im_name, l1, l2))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            print(time.time()-start_t, 'data_collecting')
            while self.Q.full():
                time.sleep(2)

            self.Q.put((l1, l2, im_name, img, orig_img))

    def collect_job(self, idx_group, num_job, im_name, l1, l2):
        for i in range(idx_group,len(im_name),num_job):
            # try:
            im_prefix = im_name[i][:-4]
            l1_k = np.loadtxt(self.local_path+f'/SaveFilePostProcess_1/davinci_{im_prefix}_output_0_Mconv7_stage6_L1_0.txt').reshape(1, 38, *self.outDim[::-1])
            l2_k = np.loadtxt(self.local_path+f'/SaveFilePostProcess_1/davinci_{im_prefix}_output_1_Mconv7_stage6_L2_0.txt').reshape(1, 19, *self.outDim[::-1])
            l1[i] = l1_k
            l2[i] = l2_k
            # except:
            #     print('Exception output, skip~')
    
    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()

