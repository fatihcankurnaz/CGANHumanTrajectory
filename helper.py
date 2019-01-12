import c3d
import sys
import glob, os
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
overall_sum = 0
overall_train = []
overall_add = []
xoverall_train = []
xoverall_add = []
flag = 0
xflag = 0
if not os.path.exists('current/'):
    os.makedirs('current/')
if not os.path.exists('xcurrent/'):
    os.makedirs('xcurrent/')
def dotproduct(v1, v2):
  return np.dot(v1,v2)

def length(v):
  return LA.norm(v)

def angle(v1, v2):
  return np.arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def per_file_creator(filename):
    global flag
    global xflag
    global overall_add
    global overall_train
    global xoverall_train
    global xoverall_add
    global overall_sum
    train_data = []
    add_data = []
    holder = []
    keeper = []
    sr = 120
    with open(filename,'rb') as f:
        try:
            reader = c3d.Reader(f)
        except Exception as e:
            return e

        
        labels = reader.point_labels
        RELB = -1
        RWRB = -1
        RUPA = -1
        for i in range(0,len(labels)):
            if "RELB" in labels[i] and "RELB-" not in labels[i] : 

                RELB = i
            if "RWRB" in labels[i] and "RWRB-" not in labels[i] : 

                RWRB = i
            if "RUPA" in labels[i] and "RUPA-" not in labels[i] : 
               
                RUPA = i
        if RELB == -1 or RUPA==-1 or RWRB == -1:
            print filename,"false form"
            exit()
        for i,points,analog in reader.read_frames():
            #print('frame {} :'.format(i))
            rhum = np.array(points[RUPA]-points[RELB]) # right humerus
            radi = np.array(points[RWRB]-points[RELB]) # bullshit bone (aka. forearm)
            ang = angle(rhum,radi)
            holder.append(ang)
            






    N = 10

    cumsum, moving_aves = [0], []

    for i, x in enumerate(holder, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)



    xadder = []
    xstart = moving_aves[0]
    xend = N
    print filename,len(moving_aves)
    overall_sum += len(moving_aves)
    sflag = 0
    for i in range(0,len(moving_aves)):
        if i%10==0 and i != 0:
            
            xoverall_train.append(xadder)
            
            xend = moving_aves[i]
            
            if sflag == 0:
                sflag +=1
                fig = plt.figure()
                plt.gca().set_color_cycle(['blue', 'red','green', 'black'])
                plt.plot(xadder,linewidth=2.0)
                fig.suptitle("start: "+str(xstart)+" - "+" end: "+str(xend), fontsize=20)
                plt.savefig('/home/fatih/c_gan/xcurrent/{}.png'.format(str(xflag).zfill(3)), bbox_inches='tight')
                xflag += 1 
                plt.close(fig)
            xoverall_add.append([xstart,xend])
            start = moving_aves[i]
            xadder = []
        xadder.append(moving_aves[i])






os.chdir(sys.argv[1]+'/')
input_dir = glob.glob("*.c3d")

for file in input_dir:
    inFile = sys.argv[1]+'/'+file
    per_file_creator(inFile)


final_trainx = np.array(xoverall_train)
final_addx = np.array(xoverall_add)
np.savez_compressed("/home/fatih/c_gan/compxXx",train = final_trainx,add =final_addx)
print "overall sun ",overall_sum