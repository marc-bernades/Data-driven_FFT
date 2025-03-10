#!/usr/bin/python3

import sys
import os
import glob
import numpy as np
import h5py
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc,rcParams
plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 18 )
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb}')
#import pint
#ureg = pint.UnitRegistry()
########## PARULA COLORMAP ##########
from matplotlib.colors import LinearSegmentedColormap
import dask.array as da

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def get_DNS(d_dir, dataset_name):
    ########## OPEN DATA FILES ##########
    data_file = h5py.File( f"../data_resolvent/{dataset_name}.h5", 'r' )
    rho_data        = data_file['rho'][:,:,:]
    u_data          = data_file['u'][:,:,:]
    v_data          = data_file['v'][:,:,:]
    w_data          = data_file['w'][:,:,:]
    T_data          = data_file['T'][:,:,:]
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]
    num_points_x    = x_data[0,0,:].size
    num_points_y    = y_data[0,:,0].size
    num_points_z    = z_data[:,0,0].size
   
    # Define container
    n_vars     = 5
    output_var = np.zeros((int(n_vars),int( num_points_z - 2 ),int( num_points_y - 2),int( num_points_x - 2) ))
    output_var[0,:,:,:] = rho_data[1:-1,1:-1,1:-1]
    output_var[1,:,:,:] = u_data[1:-1,1:-1,1:-1]
    output_var[2,:,:,:] = v_data[1:-1,1:-1,1:-1]
    output_var[3,:,:,:] = w_data[1:-1,1:-1,1:-1]
    output_var[4,:,:,:] = T_data[1:-1,1:-1,1:-1]
    
    # Convert to x, z, y (for FFT space on periodic dimensions kx, kz, y)
    for idx in range(0,n_vars):
        output_var[idx,:,:,:] = np.einsum("ijk->kij",output_var[idx,:,:,:])

    os.system(f"mkdir -p {d_dir}")
    np.save( f"{d_dir}/{dataset_name}.npy", output_var)
    
    return output_var

def save_DNS_grid(d_dir,dataset_name):

    ########## OPEN DATA FILES ##########
    print("Saving DNS grid...")
    data_file = h5py.File( f"../data_resolvent/{dataset_name}.h5", 'r' )
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]

    num_points_x    = x_data[0,0,:].size
    num_points_y    = y_data[0,:,0].size
    num_points_z    = z_data[:,0,0].size

    n_vars = 3
    grid   = np.zeros((int(n_vars),int( num_points_z - 2 ),int( num_points_y - 2),int( num_points_x - 2) ))

    grid[0,:,:,:] = x_data[1:-1,1:-1,1:-1]
    grid[1,:,:,:] = y_data[1:-1,1:-1,1:-1]
    grid[2,:,:,:] = z_data[1:-1,1:-1,1:-1]
    grid = np.einsum("ijkl->ilkj",grid)

    np.save( f"{d_dir}/grid.npy", grid)

def fft_space(output_var, L_x, L_z):
    """
    Read a specific time snapshot from the data and convert it from physical
    to spectral space. dimensions are (nx, nz, ny).

    Parameters:
    u (np.ndarray): The x-velocity field dimensioned (nx, nz, ny).
    v (np.ndarray): The y-velocity field dimensioned (nx, nz, ny).
    w (np.ndarray): The z-velocity field dimensioned (nx, nz, ny).
    """
    #rho = output_var[0,:,:,:]
    u   = output_var
    #v   = output_var[2,:,:,:]
    #w   = output_var[3,:,:,:]
    #T   = output_var[4,:,:,:]

    # Number of points
    nk_x = u.shape[0]
    nk_z = u.shape[1]
    # grid spacing
    dx = L_x/nk_x
    dz = L_z/nk_z

    kx = np.fft.fftfreq(nk_x, d=dx)*2*math.pi 
    kz = np.fft.fftfreq(nk_z, d=dz)*2*math.pi  
    #kx = kx[kx>0]
    #kz = kz[kz>0]
    
    #rho_hat = np.fft.fftn(rho, axes=(0, 1)) / nk_x / nk_z
    u_hat   = np.fft.fftn(u, axes=(0, 1))   / nk_x / nk_z
    #v_hat   = np.fft.fftn(v, axes=(0, 1))   / nk_x / nk_z
    #w_hat   = np.fft.fftn(w, axes=(0, 1))   / nk_x / nk_z
    #T_hat   = np.fft.fftn(T, axes=(0, 1))   / nk_x / nk_z


    # Store in container
    #output_hat = np.zeros((output_var.shape), dtype=complex)
    #output_hat[0,:,:,:] = rho_hat
    output_hat = u_hat
    #output_hat[2,:,:,:] = v_hat
    #output_hat[3,:,:,:] = w_hat
    #output_hat[4,:,:,:] = T_hat
    
    return output_hat, kx, kz

def fft_time(output_hat_space, kx, kz, dt, n_snapshots):



    freq   = np.fft.fftfreq(n_snapshots, d=dt)
    freq   = freq[freq > 0]
    omega  = 2*math.pi*freq

    #rho = output_hat_space[0,:,:,:,:]
    u   = output_hat_space[:,:,:,:]
    #v   = output_hat_space[2,:,:,:,:]
    #w   = output_hat_space[3,:,:,:,:]
    #T   = output_hat_space[4,:,:,:,:]
    
    #rho_hat = np.fft.fftn(rho, axes=(3,)) / n_snapshots
    u_hat   = np.fft.fftn(u,   axes=(3,)) / n_snapshots
    #v_hat   = np.fft.fftn(v,   axes=(3,)) / n_snapshots
    #w_hat   = np.fft.fftn(w,   axes=(3,)) / n_snapshots
    #T_hat   = np.fft.fftn(T,   axes=(3,)) / n_snapshots

    # Store in container
    output_hat = np.zeros((output_hat_space.shape), dtype=complex)
    #output_hat[0,:,:,:,:] = rho_hat
    output_hat[:,:,:,:] = u_hat
    #output_hat[2,:,:,:,:] = v_hat
    #output_hat[3,:,:,:,:] = w_hat
    #output_hat[4,:,:,:,:] = T_hat

    return output_hat, omega


def inv_fft_space(kx, kz, output_hat, L_x, L_z):
    
    #rho_hat = output_hat[0,:,:,:]
    u_hat   = output_hat[:,:,:]
    #v_hat   = output_hat[2,:,:,:]
    #w_hat   = output_hat[3,:,:,:]
    #T_hat   = output_hat[4,:,:,:]

    # Number of points
    nk_x = u_hat.shape[0]
    nk_z = u_hat.shape[1]
    # grid spacing
    dx = L_x/nk_x
    dz = L_z/nk_z
    #print(np.fft.ifftfreq(nk_x, d=dx))
    #print(np.fft.ifftfreq(nk_z, d=dz))   
    
    #rho_inv_fft = np.fft.ifftn(rho_hat, axes=(0, 1)) * nk_x * nk_z
    u_inv_fft   = np.fft.ifftn(u_hat, axes=(0, 1))   * nk_x * nk_z
    #v_inv_fft   = np.fft.ifftn(v_hat, axes=(0, 1))   * nk_x * nk_z
    #w_inv_fft   = np.fft.ifftn(w_hat, axes=(0, 1))   * nk_x * nk_z
    #T_inv_fft   = np.fft.ifftn(T_hat, axes=(0, 1))   * nk_x * nk_z

    # Store in container
    output_inv_fft = np.zeros((output_hat.shape), dtype=complex)
    #output_inv_fft[0,:,:,:] = np.real(rho_inv_fft)
    output_inv_fft[:,:,:] = np.real(u_inv_fft)
    #output_inv_fft[2,:,:,:] = np.real(v_inv_fft)
    #output_inv_fft[3,:,:,:] = np.real(w_inv_fft)
    #output_inv_fft[4,:,:,:] = np.real(T_inv_fft)
   
    return output_inv_fft

def inv_fft_time(output_hat, omega, dt):
    
    #rho_hat = output_hat[0,:,:,:,:]
    u_hat   = output_hat[:,:,:,:]
    #v_hat   = output_hat[2,:,:,:,:]
    #w_hat   = output_hat[3,:,:,:,:]
    #T_hat   = output_hat[4,:,:,:,:]

    # Number of points
    n_snapshots = u_hat.shape[3]
    
    #rho_inv_fft = np.fft.ifftn(rho_hat, axes=(3,)) * n_snapshots
    u_inv_fft   = np.fft.ifftn(u_hat,   axes=(3,)) * n_snapshots
    #v_inv_fft   = np.fft.ifftn(v_hat,   axes=(3,)) * n_snapshots
    #w_inv_fft   = np.fft.ifftn(w_hat,   axes=(3,)) * n_snapshots
    #T_inv_fft   = np.fft.ifftn(T_hat,   axes=(3,)) * n_snapshots

    # Store in container
    output_inv_fft = np.zeros((output_hat.shape), dtype=complex)
    #output_inv_fft[0,:,:,:,:] = (rho_inv_fft)
    output_inv_fft[:,:,:,:] = (u_inv_fft)
    #output_inv_fft[2,:,:,:,:] = (v_inv_fft)
    #output_inv_fft[3,:,:,:,:] = (w_inv_fft)
    #output_inv_fft[4,:,:,:,:] = (T_inv_fft)
   
    return output_inv_fft


def turbulent_kinetic_energy(y_plus_TKE_bw, y_plus_TKE_tw, kx, kz, output_hat, save_dir,dataset, y_plus_bw, y_plus_tw):
        
        
    # Half-numbers
    nx_max = int(0.5*len(kx)-1)
    nz_max = int(0.5*len(kz)-1)

    ### Bottom wall
    # Find idx y
    idx = np.zeros((y_plus_TKE_bw.size), dtype = int)
    idx[0] = int(np.argmin((np.abs(y_plus_bw - y_plus_TKE_bw[0]))))
    idx[1] = int(np.argmin((np.abs(y_plus_bw - y_plus_TKE_bw[1]))))
        
    E_hat_0 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[0]]))

    E_hat_1 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[1]]))
        
    [KX, KZ] = np.meshgrid(kx[0:nx_max],kz[0:nz_max])
    fig, axs = plt.subplots(2,2, figsize=(5.4, 4), sharex=True, sharey='row')
    #axs[0, 0].set_title("$y^+ = " + str(idx[0]) + "}$", fontsize=9)
    #axs[0, 1].set_title("$y^+ = " + str(idx[1]) + "}$", fontsize=9)

    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.text(0.05, 0.5, r"$k_z$", va='center', rotation='vertical')
    fig.text(0.51, 0.01, r"$k_x$", ha='center')
        
    axs[0, 0].contourf(KX, KZ, E_hat_0, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))     

    axs[0, 1].contourf(KX, KZ, E_hat_1, cmap='RdBu') # levels=np.linspace(min_val, max_val,100)) 

    ### Top wall
    # Find idx y
    idx = np.zeros((y_plus_TKE_tw.size), dtype = int)
    idx[0] = int(np.argmin((np.abs(y_plus_tw - y_plus_TKE_tw[0]))))
    idx[1] = int(np.argmin((np.abs(y_plus_tw - y_plus_TKE_tw[1]))))
        
    E_hat_0 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[0]]))

    E_hat_1 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[1]]))
        
    [KX, KZ] = np.meshgrid(kx[0:nx_max],kz[0:nz_max])
    fig, axs = plt.subplots(2,2, figsize=(5.4, 4), sharex=True, sharey='row')
    #axs[0, 0].set_title("$y^+ = " + str(idx[0]) + "}$", fontsize=9)
    #axs[0, 1].set_title("$y^+ = " + str(idx[1]) + "}$", fontsize=9)

    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.text(0.05, 0.5, r"$k_z$", va='center', rotation='vertical')
    fig.text(0.51, 0.01, r"$k_x$", ha='center')
        
    axs[1, 0].contourf(KX, KZ, E_hat_0, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))     

    axs[1, 1].contourf(KX, KZ, E_hat_1, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))  



    plt.savefig(f'figures/TKE/TKE_{dataset}.png', dpi=300)
    plt.close()




class FFT_process:
    def __init__(self, save_dir, cases, dt, delta, L_x, L_z,
             u_tau_bw, mu_bw, rho_bw, y_plus_TKE_bw,
             u_tau_tw, mu_tw, rho_tw, y_plus_TKE_tw):
        self.save_dir   = save_dir
        self.cases      = cases
        self.dt         = dt
        self.delta_h    = delta
        self.L_x        = L_x
        self.L_z        = L_z
        self.u_tau_bw   = u_tau_bw
        self.mu_bw      = mu_bw
        self.rho_bw     = rho_bw
        self.y_plus_TKE_bw = y_plus_TKE_bw
        self.u_tau_tw   = u_tau_tw
        self.mu_tw      = mu_tw
        self.rho_tw     = rho_tw
        self.y_plus_TKE_tw = y_plus_TKE_tw
        self.snapshots  = self.preprocess_snapshots()
        if not(os.path.exists(f"{self.save_dir}/grid.npy")):
            save_DNS_grid(self.save_dir,f"{cases[0]}")
        self.output_fft_space, self.kx, self.kz = self.preprocess_fft_space()
        self.output_fft_time,  self.omega       = self.preprocess_fft_time()

        #self.plots_FFT_check()
        #self.plot_invFFT_recovery()

    def plot_invFFT_recovery(self):

        # Find omegas corresponding to velocity
        output_target_hat = self.output_fft_time

        # Inverse FFT in time
        inv_output_time_hat = inv_fft_time(output_target_hat, self.omega, self.dt)

        # Inverse FFT in space
        output_target = np.real(inv_fft_space(self.kx, self.kz, inv_output_time_hat, self.L_x, self.L_z))

        # Velocity first snapshot
        u_recovery = output_target[:,:,:,0]

        self.plot_c_target_XY(output_target[:,:,:,0], "recovery_invFFT")
        self.plot_c_target_XY(self.snapshots[:,:,:,0], "raw")




    def plots_FFT_check(self):

        # Plot the magnitude of the FFT of the first frame in spatial frequency domain
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(np.fft.fftshift(self.output_fft_space[1,:,:,60,2])), cmap='jet', extent=(-len(self.kx)//2, len(self.kx)//2, -len(self.kz)//2, len(self.kz)//2))
        plt.colorbar()
        plt.title("FFT Magnitude (First Frame, Spatial Domain)")
        #plt.show()
        plt.savefig(f'figures/Test_space.png', format = 'png', bbox_inches = 'tight', dpi=600 )

        # Plot the magnitude of the FFT along the time axis for the first spatial frequency component
        plt.figure(figsize=(6, 6))
        plt.plot(np.abs(self.output_fft_time[1,len(self.kx)//2, len(self.kx)//2, 60, :]))  # Check the center spatial frequency component
        plt.title("FFT Magnitude Along Time Axis (Center Spatial Frequency Component)")
        plt.xlabel("Time Frames")
        plt.ylabel("Magnitude")
        #plt.show()
        plt.savefig(f'figures/Test_time.png', format = 'png', bbox_inches = 'tight', dpi=600 )


    def preprocess_wall_units(self):

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        y_data_bw = grid[1,0,:,0]                    # BW
        y_data_tw = (2*self.delta_h - grid[1,0,:,0]) # TW
                                   
        y_plus_bw = y_data_bw*(self.u_tau_bw/(self.mu_bw/self.rho_bw))
        y_plus_tw = y_data_tw*(self.u_tau_tw/(self.mu_tw/self.rho_tw))

        return y_plus_bw, y_plus_tw


    def preprocess_snapshots(self):
        # Initialize data container
        n_t   = 0
        n_snapshots = len(self.cases)
        print("Preprocessing snapshots...")

        for id, case in tqdm(enumerate(self.cases),total=len(self.cases)):
            if not(os.path.exists(f"{self.save_dir}/{case}.npy")):
                print("Preprocessing DNS: ", case)
                get_DNS(self.save_dir,f"{case}")
            output_DNS = np.load(f"{self.save_dir}/{case}.npy")
            n_dim, n_x, n_z, n_y = output_DNS.shape
            # Obtain delta_x, delta_y, delta_z
            if n_t == 0:
                snapshots   = np.zeros((n_dim, n_x, n_z, n_y, int(n_snapshots))) # Initialize data container
                mean_DNS    = np.zeros((n_y,n_dim))
                
            # Ensemble-averaged periodic dimensions
            for var in range (0,n_dim):
                mean_DNS[:,var] += np.mean(output_DNS[var,:,:,:], axis = (0,1))

            # Save snapshots
            snapshots[:,:,:,:,n_t]  = output_DNS
            n_t += 1

        # Subtract mean flow in rho, u and T
        mean_DNS *= 1.0/(n_t)

        for n_snapshots in range (0, n_t):
            for idx_x in range (0,n_x):
                for idx_z in range (0,n_z):
                    snapshots[0,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,0] 
                    snapshots[1,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,1] 
                    snapshots[4,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,4]

        #print("Snapshots preprocess completed...")

        return snapshots


            
    def preprocess_fft_space(self):
        n_t   = 0
        var   = 1 # u-velocity
        #output_fft_space = np.zeros((self.snapshots.shape), dtype=complex)
        #output_fft_space = sp.dok_matrix((self.snapshots.shape), dtype=complex)
        shape  = self.snapshots.shape[1:]
        chunks = (32,32,64,1)
        output_fft_space = da.zeros(shape, dtype=complex, chunks = chunks)
        print("Preprocessing FFT space...")
        # Spatial FFT
        #for id, case in enumerate(self.cases):
        for id, case in tqdm(enumerate(self.cases), total=len(self.cases)):
            #print("Computing FFT Space: ", case)
            # FFT spatial
            fft_hat_space, kx, kz = fft_space(self.snapshots[var,:,:,:,n_t], self.L_x, self.L_z)
            # Turbulent kinetic energy
            #y_plus_bw, y_plus_tw  = self.preprocess_wall_units()
            #turbulent_kinetic_energy(self.y_plus_TKE_bw, self.y_plus_TKE_tw, 
                    #kx, kz, fft_hat_space, self.save_dir, case, y_plus_bw, y_plus_tw)
            # Check recovery Inv FFT spatial
            # output_inv_fft = inv_fft_space(kx, kz, fft_hat_space, self.L_x, self.L_z)
            # Fill container
            output_fft_space[:,:,:,n_t] = fft_hat_space
            n_t += 1

        print("Computing FFT Space chunks...")
        output_fft_space = output_fft_space.compute()
        print("FFT Space completed...")
       
        return output_fft_space, kx, kz

    def preprocess_fft_time(self):

        print("Computing FFT Time ...")
        # For each realization FFT in time
        output_fft_time, omega   = fft_time(self.output_fft_space, self.kx, self.kz, self.dt, len(self.cases))
         
        print("FFT Time completed...")

        return output_fft_time, omega


    def obtain_spectra_target(self,c_plus_target, wall_str):

        print("Fourier condition to phase speed target...")

        # Phase speed in outer scales
        if wall_str == "bw":
            c_target = c_plus_target*self.u_tau_bw
        else:
            c_target = c_plus_target*self.u_tau_tw

        # Define gate bandwidth
        gate_bandwidth = 2*math.pi*10 # rad/s, the range around the desired frequency

        # Output target desired phase speed
        output_target_hat_gated = self.output_fft_time.copy() # np.zeros((self.output_fft_time.shape), dtype=complex)

        # Find omegas corresponding to phase speed for each y
        for idx_y in range (0,self.output_fft_time.shape[2]):
            for idx_kz in range (0, self.output_fft_time.shape[1]):
                for idx_omega in range (0, len(self.omega)):
                    # Find positions that match omega
                    #idx_target = np.argmin(np.abs(self.kx/self.omega[idx_omega] - c_target))
                    idx_target = np.abs(self.omega[idx_omega] - self.kx/c_target) < gate_bandwidth
                    
                    #for var in range (0,self.output_fft_time.shape[0]):
                        #output_target_hat[var,idx_target,idx_kz,idx_y,idx_omega] = self.output_fft_time[var,idx_target,idx_kz,idx_y,idx_omega]
                    output_target_hat_gated[~idx_target,idx_kz,idx_y,idx_omega] = 0

                
        # Inverse FFT in time
        inv_output_time_hat = inv_fft_time(output_target_hat_gated, self.omega, self.dt)

        # Inverse FFT in space
        output_target     = np.zeros((self.output_fft_time.shape))
        output_target = np.real(inv_fft_space(self.kx, self.kz, inv_output_time_hat, self.L_x, self.L_z))

        return output_target

    def plot_c_target_XY(self,output_target,c_plus_target,wall_str,n_snapshot):

        print("Plotting contour phase speed target in physical space...")
        
        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,:]
        y_data = grid[1,:,:,:]
        z_data = grid[2,:,:,:]

        num_points_x    = x_data[:,0,0].size
        num_points_y    = y_data[0,:,0].size
        num_points_z    = z_data[0,0,:].size
        num_points_xz   = num_points_x*num_points_z

        # Normalize grid
        y_data_norm = y_data[:,:,int(num_points_z/2)]/self.delta_h
        x_data_norm = x_data[:,:,int(num_points_z/2)]/self.delta_h

        # Obtain velocity field (input kx, kz, y)
        output_target = np.einsum("ijk->ikj",output_target) # Convert to var, x, y, z
        u_data        = output_target
        u_data        = u_data[:,:,int(num_points_z/2)]

        # Norm wall units
        if wall_str == "bw":
            u_norm = self.u_tau_bw
        else:
            u_norm = self.u_tau_tw


        # Format data
        y_data_norm      = np.asarray( y_data_norm.flatten() )
        x_data_norm      = np.asarray( x_data_norm.flatten() )
        u_data_norm      = np.asarray( u_data.flatten() )/u_norm

        u_min = 0
        u_max = c_plus_target #np.max(u_data_norm)
        # Clip data
        u_data_norm[u_data_norm < u_min ] = u_min
        u_data_norm[u_data_norm > u_max ] = u_max

        ### STREAMWISE VELOCITY

        # Clear plot
        plt.clf()

        # Plot data
        my_cmap = parula_map
        my_norm = colors.Normalize( vmin = u_min, vmax = u_max )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_norm, cmap = my_cmap, norm = my_norm, levels = np.arange( u_min, u_max + 1e-1, 1.0e-3 ) )

        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.15, pad = 0.02) #, ticks = [0,0.5,1.0] )
        cbar.ax.tick_params( labelsize = 9 ) 
        plt.text( 12.9, 2.05, r'$u^{+}$', fontsize = 9 )
        plt.clim( u_min, u_max )

        ## Configure plot
        #plt.xlim( 0.0, 12.0 )
        #plt.xticks( np.arange( 0.0, 12.1, 2.0 ) )
        pi = math.pi
        plt.xlim(0.0, 4*pi)
        plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
        plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.ylim( 0.0, 2)
        plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.gca().set_aspect( 'equal', adjustable = 'box' )
        ax = plt.gca()
        ax.tick_params( axis = 'both', pad = 7.5 )
        plt.xlabel( r'${x/\delta}$', size = 9)
        plt.ylabel( r'${y/\delta}$', size = 9 )
        label_save = c_plus_target
        plt.savefig(f'figures/Cond_phase_speed_trim_band/XY_u_c_plus_target_{label_save}_{wall_str}_{n_snapshot}.png', format = 'png', bbox_inches = 'tight', dpi=600 )
  

    
if __name__ == "__main__":

    ### DEFINED INPUTS ###
    # Define domain length
    delta    = 100*1e-6           # Fixed channel half height
    L_x      = 4*math.pi*delta
    L_z      = 4/3*math.pi*delta
    save_dir = f"data"            # Store DMD outputs
    delta    = 100*1e-6           # Fixed channel half height
    dt       = 50000*8*1e-10      # Snapshots sample rate
    u_tau_bw  = 0.19               # Bottom Wall friction velocity
    mu_bw     = 1.5312*1E-05    
    rho_bw    = 148.04
    y_plus_TKE_bw = np.array([1,10])
    u_tau_tw  = 0.19               # Top Wall friction velocity
    mu_tw     = 1.5312*1E-05    
    rho_tw    = 148.04
    y_plus_TKE_tw = np.array([1,10])


    # Snapshots (on file *.h5)
    cases = []
    for file in glob.glob("../data_resolvent/*.h5"):
        cases.append(file[18:-3])
    cases.sort()
    cases = cases[0:500]
    #cases = ["3d_high_pressure_turbulent_channel_flow_28500000", 
    #        "3d_high_pressure_turbulent_channel_flow_30000000"]
    #print(cases)
    ### DEFINE FFT CLASS ###
    # FFT Process
    FFT_class = FFT_process(save_dir,cases,dt, delta, L_x, L_z, 
            u_tau_bw, mu_bw, rho_bw, y_plus_TKE_bw,
            u_tau_tw, mu_tw, rho_tw, y_plus_TKE_tw)

    # Obtain desired omega fields
    c_plus_target = np.array([1, 5, 10, 15, 20])
    n_snapshots   = np.linspace(0,len(cases),int(len(cases)/2+1)) #np.array([0,2,4])
    print("Conditioning phase speed data...")
    for idx, nn in tqdm(enumerate(n_snapshots), total=len(n_snapshots)):
        for cc in range(0,len(c_plus_target)):
            output_target = FFT_class.obtain_spectra_target(c_plus_target[cc],"tw")
            FFT_class.plot_c_target_XY(output_target[:,:,:,int(nn)], c_plus_target[cc],"tw", int(nn))

