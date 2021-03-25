#include "cuda_timing.h"
#include <cassert>
#include <numeric>
#include <cmath>

#ifdef USE_MPI
#  include <mpi.h>
#endif

std::map<std::string, Timing::LoopData> Timing::loops;
std::vector<int> Timing::stack;
int Timing::counter = 0;


void Timing::pushRange(const std::string &_name) {
  nvtxRangePushA(_name.c_str());
}
void Timing::popRange() { nvtxRangePop(); }
void Timing::markStart(const std::string &_name) { nvtxMarkA(_name.c_str()); }

void Timing::startTimer(const std::string &_name) {
  pushRange(_name);
  if (loops.size() == 0) counter = 0;
  int parent           = stack.size() == 0 ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  int index;
  if (loops.find(fullname) != loops.end()) {
    loops[fullname].current = clock::now();
    index                   = loops[fullname].index;
  } else {
    index                  = counter;
    loops[fullname]        = LoopData(); // = {counter++, parent, 0.0, now, {}};
    loops[fullname].index  = counter++;
    loops[fullname].parent = parent;
    loops[fullname].current = clock::now();
  }
  stack.push_back(index);
}

void Timing::stopTimer(const std::string &_name) {
  auto now = clock::now();
  stack.pop_back();
  int parent           = stack.empty() ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  loops[fullname].time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - loops[fullname].current)
          .count();
  popRange();
}

void Timing::startTimerCUDA(const std::string &_name, cudaStream_t stream) {
  markStart(_name);
  if (loops.size() == 0) counter = 0;
  int parent           = stack.size() == 0 ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  cudaEvent_t start;
  cudaSafeCall(cudaEventCreate(&start));
  cudaSafeCall(cudaEventRecord(start, stream));
  int index;
  if (loops.find(fullname) != loops.end()) {
    loops[fullname].event_pairs.push_back(start);
    index = loops[fullname].index;
  } else {
    // loops[fullname] = {counter++, parent, 0.0, clock::now(), {start}};
    loops[fullname]        = LoopData(); // = {counter++, parent, 0.0, now, {}};
    loops[fullname].index  = counter++;
    loops[fullname].parent = parent;
    // loops[fullname].event_pairs = {start};
    loops[fullname].event_pairs.reserve(10);
    loops[fullname].event_pairs.push_back(start);
    index = counter - 1;
  }
  stack.push_back(index);
}

void Timing::stopTimerCUDA(const std::string &_name, cudaStream_t stream) {
  stack.pop_back();
  int parent           = stack.empty() ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  cudaEvent_t stop;
  cudaSafeCall(cudaEventCreate(&stop));
  cudaSafeCall(cudaEventRecord(stop, stream));
  loops[fullname].event_pairs.push_back(stop);
}

void Timing::reportWithParent(int parent, const std::string &indentation) {
  for (const auto &element : loops) {
    const LoopData &l = element.second;
    if (l.parent == parent) {
#ifdef USE_MPI
      int rank, nproc;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nproc);
      std::vector<double> times(nproc, 0);
      MPI_Gather(&l.time, 1, MPI_DOUBLE, times.data(), 1, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
      if (!rank) {
        double mean = 0.0;
        double max  = times[0];
        double min  = times[0];
        for (double t : times) {
          mean += t;
          max = std::max(max, t);
          min = std::min(min, t);
        }
        mean = mean / nproc;
        double stddev =
            std::accumulate(times.begin(), times.end(), 0.0,
                            [&](const double &sum, const double &time) {
                              return sum + (time - mean) * (time - mean);
                            });
        stddev = std::sqrt(stddev / nproc);

        std::cout << indentation + element.first + ": ";
        std::cout << min << "s; " << max << "s; " << mean << "s; " << stddev
                  << "s;\n";
      }
#else
      std::cout << indentation + element.first + ": "
                << std::to_string(l.time) + " seconds\n";
#endif
      reportWithParent(l.index, indentation + "  ");
    }
  }
}

void Timing::sumCudaEvents() {
  for (auto &element : loops) {
    LoopData &loop = element.second;
    assert(loop.event_pairs.size() % 2 == 0 &&
           "CUDA event measurement not closed!");
    for (int i = 0; 2 * i < loop.event_pairs.size(); ++i) {
      float milliseconds = 0;
      cudaSafeCall(cudaEventElapsedTime(&milliseconds, loop.event_pairs[2 * i],
                                        loop.event_pairs[2 * i + 1]));
      loop.time += milliseconds / 1000;
    }
    loop.event_pairs.clear();
  }
}

void Timing::report() {
  sumCudaEvents();
  reportWithParent(-1, "  ");
}
