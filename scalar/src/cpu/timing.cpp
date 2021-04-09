#include "timing.h"
#include <numeric>
#include <cmath>

#ifdef USE_MPI
#  include <mpi.h>
#endif

std::map<std::string, Timing::LoopData> Timing::loops;
std::vector<int> Timing::stack;
int Timing::counter  = 0;
bool Timing::measure = true;

void Timing::startTimer(const std::string &_name) {
  if (!measure) return;
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
  if (!measure) return;
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

void Timing::reset() {
  loops.clear();
  stack.clear();
  counter = 0;
}

void Timing::suspend_prof() { measure = false; }
void Timing::continue_prof() { measure = true; }


void Timing::report() { reportWithParent(-1, "  "); }
