// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef __TRID_X_ACC_HPP
#define __TRID_X_ACC_HPP

void trid_x_acc(float *ax, float *bx, float *cx, float *du, float *u, int nx, int ny, int nz) {
  float c2[N_MAX], d2[N_MAX];
  float aa, bb, cc, dd;
  int base, ind, i, j, k;
 
  #pragma acc data present(ax[0:nx*ny*nz],bx[0:nx*ny*nz],cx[0:nx*ny*nz],u[0:nx*ny*nz],du[0:nx*ny*nz])
  //#pragma acc data deviceptr(ax,bx,cx,u,du)
  //#pragma acc kernels loop collapse(2) independent private(aa,bb,cc,dd,base,ind,c2,d2) //async
  #pragma acc parallel loop collapse(2) private(aa,bb,cc,dd,base,ind,c2,d2)
  for(k=0; k<NZ; k++) {
     for(j=0; j<NY; j++) {
      base = k*nx*ny + j*nx;

      //
      // forward pass
      //
      bb    = 1.0f/bx[base];
      cc    = bb*cx[base];
      dd    = bb*du[base];
      c2[0] = cc;
      d2[0] = dd;
      //ind=base;
      
      for(i=1; i<nx; i++) {
        ind   = base + i;
        aa    = ax[ind];
        bb    = bx[ind] - aa*cc;
        dd    = du[ind] - aa*dd;
        bb    = 1.0f/bb;
        cc    = bb*cx[ind];
        dd    = bb*dd;
        c2[i] = cc;
        d2[i] = dd;
      }
      //
      // reverse pass
      //
      du[ind] = dd;
      base = ind;
      for(i=1; i<nx; i++) {
        ind     = base - i;
        dd      = d2[nx-i-1] - c2[nx-i-1]*dd;
        du[ind] = dd;
      }
    }
  }
  //#pragma acc wait
  //acc_async_wait_all();
}
#endif
