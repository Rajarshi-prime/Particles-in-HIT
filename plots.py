#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib,h5py
from mpi4py import MPI
from scipy.fft import fftfreq,rfftn, irfftn
# %%
N = 256
Nbin = 256
num_process = 256
Np = N//num_process
g = 0.
re = np.array([320, 6189700196426916767658409984.0])
nu = 1/re
datapath = lambda re: pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/hyper-v-normal/data_cosine/forced_True/N_256_Re_{re:.1f}")
#%%


PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky,  Kz,  indexing = 'ij')
k = (kx**2 + ky**2 + kz**2)**0.5
n = np.zeros((N,N,N))
kint = np.round(k).astype(np.int16)
dealias = kint < N/3
shells = np.arange(-0.5,3**0.5*Nf, 1.)
normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
Xbins = Ybins = Zbins = np.linspace(0, L, Nbin+1, endpoint= True)
xbins, ybins, zbins = np.meshgrid(Xbins, Ybins, Zbins, indexing='ij')

def clip_zero(x):
    oldmean = x.mean()
    x = np.clip(x,0,None)
    newmean = x.mean()
    return x*oldmean/newmean
def e3d_to_e1d(x):  return np.histogram(kint.ravel(),bins = shells,weights=x.ravel())[0]  #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 

#%%
ek = []
Pik = []
times = np.arange(40,160.1,0.5)
eseries = np.zeros((len(re),len(times)))
for i, r in enumerate(re):
    e = 0.0
    pi = 0.0
    count = 0
    for j,t in enumerate(times):
        e += np.load(datapath(r)/f"time_{t:.1f}/Energy_spectrum.npz")["ek"]/TWO_PI**3
        pi += np.load(datapath(r)/f"time_{t:.1f}/Flux_spectrum.npz")["Pik"]/TWO_PI**3
        count +=1
        eseries[i,j] = (np.load(datapath(r)/f"time_{t:.1f}/Energy_spectrum.npz")["ek"]/TWO_PI**3).sum()
    ek.append(e/count)
    Pik.append(pi/count)
    
#%%
fig,ax = plt.subplots(1,2,figsize = (7.5,3),dpi = 300)
k1d = np.arange(N//2+1)
names = ["vis","hyp-vis"]
handles = []
ylabels = [r'$E(k)/E$',r'$\Pi(k)$']
for i,r,name in zip(range(len(re)),re,names):
    print(i,r,name)
    h1, = ax[0].plot(k1d[1:],ek[i][1:]/ek[i].sum(),'.-')
    handles.append(h1)
    ax[1].plot(k1d[1:],Pik[i][1:],'.-')
    ax[i].set_xscale('log')
    ax[i].set_xlabel(r'$k$')
    ax[i].set_ylabel(ylabels[i])
ax[0].set_yscale('log')
ax[0].set_ylim(1e-6,10)
fig.tight_layout()
fig.legend(handles, names, loc = 'upper center',ncol = 2)

#%%
#%%
for i in range(len(re)):
    plt.plot(eseries[i],'.-',label = names[i])
plt.legend()
# %%

#%%
# n_field = np.zeros((N,N,N))
# for i in range(num_process):
#     data = datapath + f"/n_{i}.npz"
#     n_field[i*Np:(i+1)*Np] = np.load(data)['n']
#%%
prtcl_path = lambda re, t,N = N: pathlib.Path(f"/home/rajarshi.chattopadhyay/fluid/hyper-v-normal/data_cosine/forced_True/N_{N}_Re_{re:.1f}/time_{t:.1f}/tracer/")
Nprtcl = 128**3
Ntimes = len(times)
prtcl_mass = np.zeros((len(re),Ntimes,Nprtcl))
prtcl_id = np.zeros((len(re),Ntimes,Nprtcl))
prtcl_pos = np.zeros((len(re),Ntimes,Nprtcl,3))
prtcl_vel = np.zeros((len(re),Ntimes,Nprtcl,3))
prtcl_umat = np.zeros((len(re),Ntimes,Nprtcl,3))
# %%

# %%

# %%
for ii,r in enumerate(re):
    for i,t in enumerate(times):
        print(t, end = '\r')
        prtcl_count = 0
        for rank in range(num_process):
            data = np.load(prtcl_path(r,t)/f"state_{rank}.npz")
            prtcl_id[ii,i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['prtclid'].ravel()
            prtcl_pos[ii,i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['pos']
            prtcl_vel[ii,i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['vel']
            prtcl_umat[ii,i,prtcl_count: prtcl_count+data['prtclid'].shape[0],:] = data['umat']
            prtcl_mass[ii,i,prtcl_count: prtcl_count+data['prtclid'].shape[0] ] = data['mass']
            prtcl_count += data['prtclid'].shape[0]
        # print(prtcl_count/Nprtcl)
        argsind = prtcl_id[ii,i].argsort()
        prtcl_pos[ii,i] = prtcl_pos[ii,i,argsind]
        prtcl_mass[ii,i] = prtcl_mass[ii,i,argsind]
        prtcl_id[ii,i] = prtcl_id[ii,i,argsind]
# %%

# %%

#%%
sep = np.linalg.norm(prtcl_pos - prtcl_pos[:,0][:,None,...],axis = -1)
umag = np.linalg.norm(prtcl_umat - prtcl_umat[:,0][:,None,...],axis = -1)
vmag = np.linalg.norm(prtcl_vel - prtcl_vel[:,0][:,None,...],axis = -1)
#%%
rrms = (sep**2).mean(axis= -1)**0.5
ueng = 0.5*(prtcl_umat**2).sum(axis= (-1,-2))/Nprtcl
veng =0.5*(prtcl_vel**2).sum(axis= (-1,-2))/Nprtcl
#%%
veng.shape, eseries.shape
#%%
cols = ['r','g']
#%%
(np.abs(prtcl_pos)> 0.0).sum()/ Nprtcl*Ntimes*len(re)
#%%
#%%
for i,r in enumerate(re):
    plt.plot(times-80, rrms[i]**2,'.-',color = cols[i],label = names[i])
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(0)
plt.legend()
#%%
for i,r in enumerate(re):
    plt.plot(times, ueng[i],'.-',color = cols[i],label = names[i])
    plt.plot(times, eseries[i],'--',color = cols[i])
plt.legend()
#* To coarse grain from the particle data. 
#%%
from particles import MPI_particles
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
stb = MPI_particles(comm, L, N, Nprtcl,st, st,g,nu, 0.4,1000,1e-6,3,X,Y,Z, x,y,z)
stb.to_exterp(1)
#%%
def calc_n_exterp(pos,n,stb):
    stb.coord[:,:stb.d] = pos
    stb.exterpmat[:] = 1.0
    n[:] = stb.exterp_cosine_scalar(stb.coord,n)
    return n

#%%
n = calc_n_exterp(prtcl_pos[0],n,stb)*TWO_PI**3/Nprtcl
#%%
plt.plot(times,umag[:,np.random.randint(0,Nprtcl,10)]**2)
# %%
n = np.histogramdd(prtcl_pos[0], bins = [Xbins, Ybins, Zbins])[0]/Nprtcl*N**3
(print(n).mean(),(n.var(() + n).mean()**2)**0.5)
pdf_n,bins_n = np.histogram(n.ravel(),bins = 100,density = True)
#%%
np.unique((n), n).mean()
#%%
plt.semilogy(bins_n[1:], pdf_n,'.')
# plt.semilogy(bins_n, np.exp(-bins_n/16),'x')
#%%
# nk_field = rfftn(n_field,axes = (-3,-2,-1))
# n_fieldspectra = e3d_to_e1d(np.abs(nk_field)**2*normalize)
# n_fieldspectra.sum(), (n_field**2).sum()*dx*dy*dz
n_binnedspectra_old = n_binnedspectra.copy()
#%%
nk_binned = rfftn(n,axes = (-3,-2,-1))

n_binnedspectra = e3d_to_e1d(np.abs(nk_binned)**2*normalize)
((n**2)).mean()**0.5

#%% 
n_binnedspectra.shape
#%%
# nk_binned_dealiased = rfftn(n,axes = (-3,-2,-1))*dealias
# n_dealiased = irfftn(nk_binned_dealiased,axes= (-3,-2,-1))
# n_dealiased = clip_zero(n_dealiased)
# nk_binned_dealiased = rfftn(n_dealiased,axes = (-3,-2,-1))
# print(nk_binned_dealiased.shape, k.shape)
# n_binned_dealiasedspectra = e3d_to_e1d(np.abs(nk_binned_dealiased)**2*normalize)
# n_binned_dealiasedspectra.sum(), (n_dealiased**2).sum()*dx*dy*dz
#%%
k1d =np.arange(shells.size - 1)
logk = np.log(k1d)
# l_field_slope = 0.5*(n_fieldspectra[1:] - n_fieldspectra[:-1])/(logk[1:] - logk[:-1])
# l_binned_dealised_slope = 0.5*(n_binned_dealiasedspectra[1:] - n_binned_dealiasedspectra[:-1])/(logk[1:] - logk[:-1])
#%%
with h5py.File("spectra.hdf5","a") as f:
    try:
        f[f'/st_{st}/k'][...] = k1d
        f[f'/st_{st}/n2k1d'][...] = n_binnedspectra
        f[f'/st_{st}/n'][...] = n
    except:
        f[f'/st_{st}/k'] = k1d
        f[f'/st_{st}/n2k1d'] = n_binnedspectra
        f[f'/st_{st}/n'] = n
#%%
mpl.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1,figsize = (4,3),dpi = 300)
# ax[0].loglog(k[1:],(n_fieldspectra)[1:],'.-',label = 'Field')
ax.loglog(k1d[1:],(n_binnedspectra)[1:],'.-',label = 'Delta-smeared')
ax.loglog(k1d[1:],(n_binnedspectra_old)[1:],'.-',label = 'Binned')
# ax[0].loglog(k[1:],(n_binned_dealiasedspectra)[1:],'.-',label = 'binned + processed')
# ax[1].semilogx(k[1:],(l_field_slope),'.-',label = 'Field')
# ax[1].semilogx(k[1:],(l_binned_dealised_slope),'.-',label = 'binned + processed')
ax.axvline(N//2,color = 'k',ls = '--',lw = 0.5)
ax.axvline(2**0.5*N//3,color = 'k',ls = '--',lw = 0.5)
# ax[0].set_ylim(1e-3,30)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$|\hat{n}_k|^2$',rotation = 0,labelpad = 10)
ax.legend()
ax.set_ylim(1e-5,None)
ax.set_xlim(1,None)
fig.tight_layout()
#%%
p1  = plt.imshow(n[100],cmap = "Greys",extent = (0,X[-1],0,Y[-1]),origin = 'lower',vmax = 4.0)
plt.colorbar(p1,extend = 'max(')
#%%
n).mean()
#%%
# norm = mpl.colors.Normalize(vmin=0, vmax=10)

# fig,ax = plt.subplots(1,3,figsize = (7.5,3),dpi=300)
# p1 = ax[0].imshow(n[10],origin = 'lower', cmap = 'Greys',norm = norm,extent = (0,X[-1],0,Y[-1]))
# # ax[1].imshow(n_dealiased[10],origin = 'lower', cmap = 'Greys',norm = norm,extent = (0,X[-1],0,Y[-1]))
# # ax[2].imshow(n_field[10],origin = 'lower', cmap = 'Greys',norm = norm,extent = (0,X[-1],0,Y[-1]))
# ax[0].set_title('(a) Binned\n')
# # ax[1].set_title('(b) Binned \n+ processed')
# # ax[2].set_title('(c) Field\n')
# fig.tight_layout()
# fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,shrink = 0.5, extend = 'max')
# plt.show()
# %%
# %%
with h5py.File("spectra.hdf5","a") as f:
    try:
        f[f'/N_{N}/k'][...] = k
        f[f'/N_{N}/n2k1d'][...] = n_binnedspectra
    except:
        f[f'/N_{N}/k'] = k
        f[f'/N_{N}/n2k1d'] = n_binnedspectra
        
# %%
Nbins = [4,8,16,32,64,128,256,512]
with h5py.File("spectra.hdf5","a") as f:
    for N in Nbins:
        k = f[f'/N_{N}/k'][1:N//2]
        n_binnedspectra  = f[f'/N_{N}/n2k1d'] [1:N//2]
        plt.loglog(k, k**(-2.0)*n_binnedspectra,'.-', label = f'{N}')
        
plt.legend(ncols = len(Nbins),handlelength = 0)
plt.ylim(1e-4,1)
# %%
