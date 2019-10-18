from SBRG import *

import json

import argparse

parser = argparse.ArgumentParser(description='')
group = parser.add_argument_group('calculation  parameters')

group.add_argument("-l", type=int,default = 22)
group.add_argument("-delta",type=float default=0.1)
group.add_argument("-jx", type=float,default=1.0)
group.add_argument("-jy", type=float,default=1.0)
group.add_argument("-jz", type=float,default=1.0)
group.add_argument("-gamma", type=float,default=4.0)
group.add_argument("-realization", type=int, default=100)
args = parser.parse_args()

SBRG.tol = 1e-8
SBRG.max_rate = 2
SBRG.max_len = 1000


lx =int(args.l)
ly =int(args.l)
delta = args.delta
qs1 = [((lx/2-i)*np.pi/lx,(lx/2-i)*np.pi/lx) for i in range(int(lx/2+1))]
qs2 = [(i*np.pi/lx,0) for i in range(1,lx+1)]
qs = qs1 + qs2

scoefsXYZ_tri_dict = {}

realization = 3

spectrum_res = {}

def main():
    for real_id in range(int(args.realization)):
        print('Process:{}%'.format(real_id/args.realization*100))
        para = {'jx':args.jx, 'jy':args.jy, 'jz':args.jz, 'alpha':1./args.gamma}
        # lx = 20
        # ly = 20
        random.seed()
        model_tri = triangular_XYZ(lx,ly, **para)
        system = SBRG(model_tri)
        system.run()
        g_state_tri, g_energy = system.grndstate_blk()
        system.site2Heffmats()
        scoefsXYZ_tri=[]
        for xyz in range(1,4):
            opsXYZ =  Ham([Term(mkMat({i:xyz})) for i in range(system.size)]) 
            scoefsXYZ_tri.append( system.two_spin_chf2(opsXYZ, g_state_tri) )
    #delta = 0.1
    #qs = [(i*np.pi/lx,i*np.pi/lx) for i in range(int(lx/2+1))]
        scoefsXYZ_tri_dict[real_id] = scoefsXYZ_tri
    
        model_tri.lx=model_tri.ly=ly
        q2ijs = []
        for q in qs:
            q2ij = {}
            for i in range(system.size):
                ix = i%model_tri.lx
                iy = i//model_tri.lx
                for j in range(system.size):
                    jx = j%model_tri.lx
                    jy = j//model_tri.lx
                    q2ij[str(i)+"-"+str(j)] = q[0]*(jx-ix) + q[1]*(jy-iy)
            q2ijs.append(q2ij)
        omegas = np.linspace(0,1,1000)
        spectrum = np.zeros( (len(omegas), len(q2ijs)), dtype=np.float32)#, dtype=np.complex128 )
        for _ in range(3):
            spectrum += system.Sqw(scoefsXYZ_tri[_], q2ijs, omegas, delta)
        spectrum_res[real_id]=spectrum.tolist()

    data = {}
    data["spectrum_res"]=spectrum_res
    data["scoefs"] = scoefsXYZ_tri_dict

    with open("./tri_XYZ_jx{}jy{}jz{}gamma{}_L{}.json".format(args.jx,args.jy,args.jz,args.gamma, args.l),"w") as write_file:
        json.dump(data, write_file)


if __name__ == '__main__':
    main()











