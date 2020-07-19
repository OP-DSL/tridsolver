#include "cuda_timing.h"
#include <cassert>
#include <thread>

#ifdef USE_MPI
#  include <mpi.h>
#endif

std::map<std::string, Timing::LoopData> Timing::loops;
std::vector<int> Timing::stack;
int Timing::counter = 0;

void Timing::startTimer(const std::string &_name) {
  auto now = clock::now();
  if (loops.size() == 0) counter = 0;
  int parent           = stack.size() == 0 ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  int index;
  if (loops.find(fullname) != loops.end()) {
    loops[fullname].current = now;
    index                   = loops[fullname].index;
  } else {
    loops[fullname]        = LoopData(); // = {counter++, parent, 0.0, now, {}};
    loops[fullname].index  = counter++;
    loops[fullname].parent = parent;
    loops[fullname].current = now;
    index                   = counter - 1;
  }
  stack.push_back(index);
}

void Timing::stopTimer(const std::string &_name) {
  stack.pop_back();
  int parent           = stack.empty() ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  auto now             = clock::now();
  loops[fullname].time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - loops[fullname].current)
          .count();
}

void Timing::startTimerCUDA(const std::string &_name) {
  if (loops.size() == 0) counter = 0;
  int parent           = stack.size() == 0 ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  cudaEvent_t start;
  cudaSafeCall(cudaEventCreate(&start));
  cudaSafeCall(cudaEventRecord(start));
  int index;
  if (loops.find(fullname) != loops.end()) {
    loops[fullname].event_pairs.push_back(start);
    index = loops[fullname].index;
  } else {
    // loops[fullname] = {counter++, parent, 0.0, clock::now(), {start}};
    loops[fullname]        = LoopData(); // = {counter++, parent, 0.0, now, {}};
    loops[fullname].index  = counter++;
    loops[fullname].parent = parent;
    loops[fullname].event_pairs = {start};
    index                       = counter - 1;
  }
  stack.push_back(index);
}

void Timing::stopTimerCUDA(const std::string &_name) {
  stack.pop_back();
  int parent           = stack.empty() ? -1 : stack.back();
  std::string fullname = _name + "(" + std::to_string(parent) + ")";
  cudaEvent_t stop;
  cudaSafeCall(cudaEventCreate(&stop));
  cudaSafeCall(cudaEventRecord(stop));
  loops[fullname].event_pairs.push_back(stop);
}

void Timing::reportWithParent(int parent, const std::string &indentation) {
  for (const auto &element : loops) {
    const LoopData &l = element.second;
    if (l.parent == parent) {
      std::cout << indentation + element.first + ": " + std::to_string(l.time) +
                       " seconds\n";
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
#ifdef USE_MPI
  int rank, num_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  std::srand(rank);
  for (int i = 0; i < num_proc; ++i) {
    // Print the outputs
    if (i == rank) {
      std::cout << "##########################\n"
                << "Rank " << i << "\n"
                << "##########################\n";
#endif
      reportWithParent(-1, "  ");
#ifdef USE_MPI
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}
