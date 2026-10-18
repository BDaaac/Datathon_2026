import json, time, glob, sys, cv2, numpy as np, os
from rtmlib import RTMPose, draw_skeleton
import rtmlib.tools.solution.body as B
mode = sys.argv[1] if len(sys.argv)>1 else 'm'
urls = {'s':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
        'm':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'}
sz=(192,256)
import onnxruntime as ort
m = RTMPose(urls[mode], model_input_size=sz, to_openpose=False, backend='onnxruntime', device='cpu')
import cv2
nth=int(sys.argv[2]) if len(sys.argv)>2 else 0
ts=[]; npers=[]; tf=[]
for fid in ['18','02']:
    d=json.load(open(f'cache_adapt/{fid}.json'))['frames']
    files=sorted(glob.glob(f'frames/{fid}/*.jpg'))
    ks=list(range(0,len(d),max(1,len(d)//10)))[:10]
    for k in ks:
        img=cv2.imread(files[k]) if k<len(files) else None
        if img is None: continue
        bb=[x[2] for x in d[k]['dets'] if x[0]=='person' and x[1]>=0.5]
        if not bb: continue
        t0=time.perf_counter(); kp,sc=m(img,bboxes=bb); dt=time.perf_counter()-t0
        tf.append(dt); npers.append(len(bb)); 
        if mode=='m' or True:
            out=draw_skeleton(img.copy(),kp,sc,kpt_thr=0.4,radius=3,line_width=2)
            for b in bb: cv2.rectangle(out,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),1)
            cv2.imwrite(f'pose_out/{mode}_{fid}_{k:04d}.jpg',out)
print(mode,'frames',len(tf),'persons',sum(npers),'ms/frame',1000*np.mean(tf[1:]),'ms/person',1000*sum(tf[1:])/sum(npers[1:]), 'files', files[0], 'nfiles',len(files),'nframes',len(d))
