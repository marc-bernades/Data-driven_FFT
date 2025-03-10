Data-driven FFT-based method

Objective: FFT space and time snapshots (.h5 DNS inputs) to identify the structures travelling at desired phase speed

Procedure:

1) Obtain DNS data and build the containers of snapshots
2) FFT space each realization on periodic dimensions u = f(kx,kz,y,t)
3) FFT time u = f(kx,kz,y,omega)
4) For each realization finde the phase speed condition
5) Transform back to physical space the containers so that the velocity field contains only the structures that met the condition
