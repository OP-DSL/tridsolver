#include "timing.h"
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
    loops[fullname] = {counter++, parent, 0.0, now};
    index           = counter - 1;
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

void Timing::report() {
#ifdef USE_MPI
  // For the debug prints
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
