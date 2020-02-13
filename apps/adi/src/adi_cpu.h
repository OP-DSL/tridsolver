#ifndef __ADI_CPU_H
#define __ADI_CPU_H

//#pragma offload_attribute(push,target(mic))
//#ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof, __INTEL_OFFLOAD is another option
  #include <sys/time.h>
//#endif
//#pragma offload_attribute(pop)

//#include "trid_params.h"

//
// linux timing routine
//

//__attribute__((target(mic)))
inline double elapsed_time(double *et) {
  struct timeval t;
  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}

//__attribute__((target(mic)))
inline void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer);
}

//__attribute__((target(mic)))
inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
  double elapsed;
  if(prof==1) {
    elapsed = elapsed_time(timer);
    *elapsed_accumulate += elapsed;
    //printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed);
  }
}

#endif 
