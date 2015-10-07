// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef __TRID_Z_ACC_HPP
#define __TRID_Z_ACC_HPP

void trid_z_acc(float *az, float *bz, float *cz, float *du, float *u, int nx, int ny, int nz) {
  int i,j,k; 
  float c2[N_MAX], d2[N_MAX];
  float aa, bb, cc, dd;
  int base, ind;
 
  #pragma acc data present(az[0:nx*ny*nz],bz[0:nx*ny*nz],cz[0:nx*ny*nz],u[0:nx*ny*nz],du[0:nx*ny*nz])
  //#pragma acc data deviceptr(az,bz,cz,u,du)
  //#pragma acc kernels loop collapse(2) independent private(aa,bb,cc,dd,base,ind,c2,d2) //async
  #pragma acc parallel loop collapse(2) private(aa,bb,cc,dd,base,ind,c2,d2)
  for(j=0; j<ny; j++) {
    for(i=0; i<NX; i++) {
      base = j*nx + i;
      //
      // forward pass
      //
      bb    = 1.0f/bz[base];
      cc    = bb*cz[base];
      dd    = bb*du[base];
      c2[0] = cc;
      d2[0] = dd;
      #pragma unroll
      for(k=1; k<nz; k++) {
        ind   = base + k*nx*ny;
        aa    = az[ind];
        bb    = bz[ind] - aa*cc;
        dd    = du[ind] - aa*dd;
        bb    = 1.0f/bb;
        cc    = bb*cz[ind];
        dd    = bb*dd;
        c2[k] = cc;
        d2[k] = dd;
      }
      //
      // reverse pass
      //
      base = base + (nz-1)*nx*ny;
      du[base] = dd;
      #pragma unroll
      for(k=1; k<nz; k++) {
        ind = base - k*nx*ny;
        dd  = d2[nz-k-1] - c2[nz-k-1]*dd;
        u[ind] += dd;
      }
    }
  }
}

#endif
