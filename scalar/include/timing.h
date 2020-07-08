#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED
// Written by Istvan Reguly, Pazmany Peter Catholic University 2020
// With contributions from:
// Gabor Daniel Balogh, Pazmany Peter Catholic University 2020

#include <chrono>
#include <map>
#include <vector>
#include <iostream>

class Timing {
  using clock      = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  struct LoopData {
    int index   = 0;
    int parent  = 0;
    double time = 0.0;
    time_point current;
  };
  std::string name;

  static std::map<std::string, LoopData> loops;
  static std::vector<int> stack;
  static int counter;

  static void reportWithParent(int parent, const std::string &indentation);

public:
  explicit Timing(std::string &&name_p) : name(std::move(name_p)) {
    startTimer(name);
  }
  ~Timing() { stopTimer(name); }

  static void startTimer(const std::string &_name);
  static void stopTimer(const std::string &_name);
  static void report();
};

#if PROFILING
#  define BEGIN_PROFILING(name) Timing::startTimer(name)
#  define END_PROFILING(name)   Timing::stopTimer(name)

#  define PROFILE_SCOPE(name) Timing timer##__LINE__(name)
#  define PROFILE_FUNCTION()  PROFILE_SCOPE(__FUNCTION__)
#  define PROFILE_REPORT()    Timing::report()
#else
#  define BEGIN_PROFILING(name)
#  define PROFILE_SCOPE(name)
#  define PROFILE_FUNCTION()
#  define END_PROFILING(name)
#  define PROFILE_REPORT()
#endif

#endif /* ifndef TIMING_H_INCLUDED */
