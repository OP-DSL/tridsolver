// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef __TRID_Y_ACC_HPP
#define __TRID_Y_ACC_HPP

void trid_y_acc(float *ay, float *by, float *cy, float *du, float *u, int nx, int ny, int nz) {
  int i,j,k; 
  float c2[N_MAX], d2[N_MAX];
  float aa, bb, cc, dd;
  int base, ind;

  #pragma acc data present(ay[0:nx*ny*nz],by[0:nx*ny*nz],cy[0:nx*ny*nz],u[0:nx*ny*nz],du[0:nx*ny*nz])
  //#pragma acc data deviceptr(ay,by,cy,u,du)
  //#pragma acc kernels loop collapse(2) independent private(aa,bb,cc,dd,base,ind,c2,d2) //async
  //#pragma acc parallel loop collapse(2) private(aa,bb,cc,dd,base,ind,c2,d2)
  #pragma acc parallel loop private(aa,bb,cc,dd,base,ind,c2,d2)
  for(k=0; k<nz; k++) {
    #pragma acc loop private(aa,bb,cc,dd,base,ind,c2,d2)
    for(i=0; i<NX; i++) {
      base = k*nx*ny + i;

      //
      // forward pass
      //
      bb    = 1.0f/by[base];
      cc    = bb*cy[base];
      dd    = bb*du[base];
      c2[0] = cc;
      d2[0] = dd;

      for(j=1; j<ny; j++) {
        ind  = base + j*nx;
        aa    = ay[ind];
        bb    = by[ind] - aa*cc;
        dd    = du[ind] - aa*dd;
        bb    = 1.0f/bb;
        cc    = bb*cy[ind];
        dd    = bb*dd;
        c2[j] = cc;
        d2[j] = dd;
      }
      //
      // reverse pass
      //
      du[ind] = dd;
      base = ind;

      for(j=1; j<ny; j++) {
        ind    = base - j*nx;
        dd     = d2[ny-j-1] - c2[ny-j-1]*dd;
        du[ind] = dd;
      }
    }
  }
 //#pragma acc wait
}

#endif
