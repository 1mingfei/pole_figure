#run with "pythonw" as framework
import numpy as np
from ase.io import read,write
from ase  import neighborlist
import formatxfer as fx
from sklearn import preprocessing

def save_nb_list(file_in,cutoff):
    atoms = read(file_in)
    nblist=neighborlist.neighbor_list('ijD', atoms, cutoff)
    nnn = np.bincount(nblist[0]) #number of nearesr neighbors
    import matplotlib.pyplot as plt
    plt.hist(nnn, bins='auto')
    plt.savefig('cn_dist.eps')
    return(nnn,nblist[0],nblist[1],nblist[2])

def pole_single(ori_0,ori,pole_family):
    if pole_family==111:
        h=1.0/np.sqrt(3.0)*np.asarray(
                [
                 [1.0, 1.0, 1.0],
                 [-1.0, 1.0 ,1.0],
                 [1.0, -1.0, 1.0],
                 [-1.0,-1.0, 1.0],
                 ]
                ).T
    elif pole_family==100:
        h=1.0/np.sqrt(1.0)*np.asarray(
                [
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0,-1.0, 0.0],
                 ]
                ).T
    elif pole_family==110:
        h=1.0/np.sqrt(2.0)*np.asarray(
                [
                [ 1.0,0.0,1.0],
                [-1.0,0.0,1.0],
                [ 0.0,1.0,1.0],
                [0.0,-1.0,1.0],
                ]
               ).T
    #g=np.linalg.lstsq(ori_0,ori)
    g=np.eye(3)
    h_prim=np.dot(np.linalg.inv(g),h)
    #print(h)
    theta=np.arccos(h_prim[2,:])
    phi=np.zeros((h_prim.shape[1]))
    px=np.zeros((h_prim.shape[1]))
    py=np.zeros((h_prim.shape[1]))
    for i in range(h_prim.shape[1]):
        phi[i]=np.arctan2(h_prim[1][i],h_prim[0][i])
        px[i]=np.tan(theta[i]/2.0)*np.cos(phi[i])
        py[i]=np.tan(theta[i]/2.0)*np.sin(phi[i])
        print (px[i],py[i])
    #plot figures
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #ax = fig.add_subplot(111, polar=True)
    ax = fig.add_subplot(111)
    circle1 = plt.Circle((0, 0), 1.0, color='black', fill=False)
    ax.add_artist(circle1)
    ax.plot(px,py,'o')
    ax.set_aspect('equal')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    #ax.legend()
    #plt.show()
    plt.savefig(str(pole_family)+'.png')
    plt.close()
    return

def atom_pole(n,nnn,nb1,nb3):
    #find 001 direction by two 110 direction vectors
    lst=np.argwhere(nnn == 12)
    lst=lst.flatten()
    np.random.shuffle(lst)
    i=0
    epoch=0
    h_prim=[]
    while epoch < n:
        print(i)
        atom=lst[i] #atom number draw from the randomized list where CN==12
        nbl_index=np.argwhere(nb1 == atom)
        tmp_list=[]
        for j in range(12):
            if nb3[nbl_index[j]][0][2] > 0.0:
                tmp_list.append(nb3[nbl_index[j]][0])
        tmp_list=np.asarray(tmp_list)
        norm_list=preprocessing.normalize(tmp_list, norm='l2')
        flag1=0
        for j in range(len(norm_list)-1):
            if flag1==1 : break
            for k in range(j,len(norm_list)):
                if flag1==1 : break
                tmp1=np.dot(norm_list[j],norm_list[k])
                if abs(tmp1) <= 0.05:
                    epoch+=1
                    h001=np.add(norm_list[j],norm_list[k])/2.0
                    h_prim.append(h001)
                    flag1=1
        i+=1
        if (i==len(lst)):
            break

    h_prim=np.asarray(h_prim)
    #calculate pole figure px py 
    theta=np.arccos(h_prim.T[2,:])
    phi=np.zeros((h_prim.T.shape[1]))
    px=np.zeros((h_prim.T.shape[1]))
    py=np.zeros((h_prim.T.shape[1]))
    for i in range(h_prim.T.shape[1]):
        phi[i]=np.arctan2(h_prim.T[1][i],h_prim.T[0][i])
        px[i]=np.tan(theta[i]/2.0)*np.cos(phi[i])
        py[i]=np.tan(theta[i]/2.0)*np.sin(phi[i])
        print (px[i],py[i])

    #plot figures
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #ax = fig.add_subplot(111, polar=True)
    ax = fig.add_subplot(111)
    circle1 = plt.Circle((0, 0), 1.0, color='black', fill=False)
    ax.add_artist(circle1)
    ax.plot(px,py,'o')
    ax.set_aspect('equal')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    plt.axis('off')
    #ax.legend()
    #plt.show()
    plt.savefig('pole_fig.png')
    plt.close()

    return

def select_surface(file_in,nnn,nb1,nb3,cn):
    aa=fx.info(file_in,'cfg',1)
    data=aa.data     
    n_type=aa.atom_type_num
    lst=np.argwhere(nnn <= cn)
    tot_num=len(lst)
    lst=lst.flatten()
    data_new=data[lst,:]
    aa.data=data_new
    aa.tot_num=tot_num
    aa.get_cfg_file('surface.cfg')
    return




#pole_single(1,1,111)
#pole_single(1,1,100)
#pole_single(1,1,110)

file_in='Ag.cfg'
cutoff=3.5
nnn,nb1,nb2,nb3=save_nb_list(file_in,cutoff)
select_surface(file_in,nnn,nb1,nb3,9)

n=5000
atom_pole(n,nnn,nb1,nb3)
