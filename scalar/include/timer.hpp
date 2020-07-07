#ifndef TIMER_HPP_UAYKBI6U
#define TIMER_HPP_UAYKBI6U
#ifdef USE_TIMER_MACRO

#  define TIMER_DECL(t)   Timer t
#  define TIMER_START(t)  Timer t(false)
#  define TIMER_TOGGLE(t) t.toggle()
#  define TIMER_PRINT_MESSAGE(t, pre)                                          \
    do {                                                                       \
      std::cout << pre << " time: ";                                           \
      std::cout                                                                \
          << t.getTime<std::chrono::duration<double, std::milli>>().count()    \
          << " ms";                                                            \
      std::cout << std::endl;                                                  \
    } while (0)
#  define TIMER_PRINT(t) TIMER_PRINT_MESSAGE(t, #  t)

#else // don't use timer

#  define TIMER_START(t)
#  define TIMER_DECL(t)
#  define TIMER_TOGGLE(t)
#  define TIMER_PRINT_MESSAGE(t, pre)
#  define TIMER_PRINT(t)

#endif

#include <chrono>
#include <iostream>

class Timer {
  using clock      = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  time_point start;
  time_point paused_time;
  bool paused = true;

public:
  Timer() : start{clock::now()}, paused_time{clock::now()} {}
  explicit Timer(bool paused)
      : start{clock::now()}, paused_time{clock::now()}, paused{paused} {}

  template <typename duration> duration getTime() const {
    if (paused) {
      return std::chrono::duration_cast<duration>(paused_time - start);
    }
    time_point stop = clock::now();
    return std::chrono::duration_cast<duration>(stop - start);
  }

  long long getTimeMillis() const {
    return getTime<std::chrono::milliseconds>().count();
  }

  void printTimeMillis() const { std::cout << getTimeMillis() << " ms"; }

  void toggle() {
    if (!paused) {
      paused_time = clock::now();
    } else {
      time_point now = clock::now();
      start += now - paused_time;
    }
    paused = !paused;
  }
};
#endif /* end of include guard: TIMER_HPP_UAYKBI6U */
