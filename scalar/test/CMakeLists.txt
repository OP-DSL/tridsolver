if (BUILD_TESTING)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

  set(TEST_SOURCES
    catch_main.cpp)

  if (BUILD_FOR_CPU)
    set(TEST_SOURCES ${TEST_SOURCES} test_cpu.cpp)
  endif (BUILD_FOR_CPU)

  add_executable(tests ${TEST_SOURCES})
  target_link_libraries(tests Catch)

  if (BUILD_FOR_CPU)
    target_link_libraries(tests tridcpu)
    target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
  endif (BUILD_FOR_CPU)

  add_test(NAME catch_test
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif (BUILD_TESTING)
