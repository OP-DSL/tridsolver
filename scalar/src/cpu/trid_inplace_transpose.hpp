#ifndef INPLACE_TRANSPOSE
#define INPLACE_TRANSPOSE
#include <vector>

// Non-square matrix transpose of matrix of size r x c and base address A
// Transpose uses permutations. Transpose a 2D matrix with num_elems_per_point
// value in each point in the matrix.
template <typename T, int num_elems_per_point = 2>
void rcv_buffer_inplace_transpose(T *A, int r, int c) {
  int size = r * c - 1;
  T t[num_elems_per_point];          // holds element to be replaced, eventually
                                     // becomes next element to move
  int next;                          // location of 't' to be moved
  int cycleBegin;                    // holds start of cycle
  int i;                             // iterator
  std::vector<bool> b(r * c, false); // hash to mark moved elements

  b[0] = b[size] = 1;
  i              = 1; // Note that A[0] and A[size-1] won't move
  auto cpy       = [&](T *dst, T *src, int idx) {
    for (int i = 0; i < num_elems_per_point; ++i) {
      dst[i] = src[idx * num_elems_per_point + i];
    }
  };
  auto swp_buf = [&](int idx) {
    for (int i = 0; i < num_elems_per_point; ++i) {
      std::swap(A[num_elems_per_point * idx + i], t[i]);
    }
  };
  while (i < size) {
    cycleBegin = i;
    cpy(t, A, i);
    do {
      // Input matrix [r x c]
      // Output matrix [c x r]
      // i_new = (i*r)%(N-1)
      next = (i * r) % size;
      swp_buf(next);
      b[i] = 1;
      i    = next;
    } while (i != cycleBegin);

    // Get next cycle
    for (; i < size && b[i]; i++)
      ;
  }
}

// Transpose the rcv_buf of Gather operation. The input matrix is assumed to
// store the boundary of each process. The innermost dimension is the a, c, d
// values, the second is n_sys, and the outermost dimension is the processes.
// In the output layout the outermost dimension is a, c, d (hence aa_r will be
// equal to rcvbuf, cc_r = rcvbuf + ns * np * num_elems_per_proc, dd_r = rcvbuf
// + ns*np*num_elems_per_proc*2) and the innermost dimension is the numproc,
// hence each system will be contiguous in memory.
template <typename T, int num_elems_per_proc = 2>
void transpose_rcvbuf_to_reduced_allgather(T *rcvbuf, int ns, int numproc,
                                           T **aa_r, T **cc_r, T **dd_r) {
  // shuffle dims such that a, c, d will be the first dim
  rcv_buffer_inplace_transpose<T, num_elems_per_proc>(rcvbuf, ns * numproc, 3);

  *aa_r = rcvbuf;
  *cc_r = rcvbuf + ns * numproc * num_elems_per_proc;
  *dd_r = rcvbuf + ns * numproc * num_elems_per_proc * 2;
  // transpose each array, each system will be continuous
  rcv_buffer_inplace_transpose<T, num_elems_per_proc>(*aa_r, numproc, ns);
  rcv_buffer_inplace_transpose<T, num_elems_per_proc>(*cc_r, numproc, ns);
  rcv_buffer_inplace_transpose<T, num_elems_per_proc>(*dd_r, numproc, ns);
}
#endif /* ifndef INPLACE_TRANSPOSE */
