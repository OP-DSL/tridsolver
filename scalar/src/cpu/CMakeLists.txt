if (USE_PROFILING)
  add_library(profiling ${BUILD_TYPE} ./timing.cpp)
  target_include_directories(profiling PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                              $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
  set_property(TARGET profiling PROPERTY POSITION_INDEPENDENT_CODE ON)
  if (BUILD_FOR_MPI)
    target_compile_definitions(profiling PRIVATE -DUSE_MPI)
    target_link_libraries(profiling MPI::MPI_CXX)
  endif (BUILD_FOR_MPI)
  target_compile_definitions(profiling PUBLIC -DPROFILING=1)
  set(${PROJECT_NAME}_components "${${PROJECT_NAME}_components} profiling")
  set(${PROJECT_NAME}_components
      ${${PROJECT_NAME}_components}
      PARENT_SCOPE)
  install(
    TARGETS profiling
    EXPORT profilingTarget
    LIBRARY DESTINATION ${PROJECT_NAME}/lib
    ARCHIVE DESTINATION ${PROJECT_NAME}/lib)
  install(FILES ${PROJECT_SOURCE_DIR}/include/timing.h DESTINATION ${PROJECT_NAME}/include)
  install(
    EXPORT profilingTarget
    FILE profilingTarget.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${PROJECT_NAME}/lib/cmake)
endif (USE_PROFILING)

set(CPU_LIBS "OpenMP::OpenMP_CXX;trid_common_options;trid_global_options")
set(CPU_INTERFACE_LIBS "trid_global_options")
if (USE_PROFILING)
  set(CPU_LIBS "${CPU_LIBS};profiling")
  set(CPU_INTERFACE_LIBS "${CPU_INTERFACE_LIBS};profiling")
endif (USE_PROFILING)

if (BUILD_FOR_CPU)
  if (BUILD_FOR_SN)
    set(${PROJECT_NAME}_components "${${PROJECT_NAME}_components} tridcpu")
    set(${PROJECT_NAME}_components
        ${${PROJECT_NAME}_components}
        PARENT_SCOPE)

    add_library(tridcpu ${BUILD_TYPE} ./trid_cpu.cpp)
    add_library(${PROJECT_NAME}::tridcpu ALIAS tridcpu)

    target_link_libraries(tridcpu PRIVATE ${CPU_LIBS})
    target_link_libraries(tridcpu PUBLIC ${CPU_INTERFACE_LIBS})

    set_property(TARGET tridcpu PROPERTY POSITION_INDEPENDENT_CODE ON)

    install(
      TARGETS tridcpu
      EXPORT tridcpuTarget
      LIBRARY DESTINATION ${PROJECT_NAME}/lib
      ARCHIVE DESTINATION ${PROJECT_NAME}/lib)
    install(FILES ${PROJECT_SOURCE_DIR}/include/trid_cpu.h ${PROJECT_SOURCE_DIR}/include/trid_common.h
                  ${PROJECT_SOURCE_DIR}/include/trid_simd.h DESTINATION ${PROJECT_NAME}/include)
    install(
      EXPORT tridcpuTarget
      FILE tridcpuTarget.cmake
      NAMESPACE ${PROJECT_NAME}::
      DESTINATION ${PROJECT_NAME}/lib/cmake)
  endif (BUILD_FOR_SN)
  if (BUILD_FOR_MPI)
    set(${PROJECT_NAME}_components "${${PROJECT_NAME}_components} tridcpu_mpi")
    set(${PROJECT_NAME}_components
        ${${PROJECT_NAME}_components}
        PARENT_SCOPE)
    add_library(tridcpu_sp_mpi OBJECT ./trid_mpi_cpu.cpp)
    add_library(tridcpu_dp_mpi OBJECT ./trid_mpi_cpu.cpp)
    add_library(tridcpu_mpi ${BUILD_TYPE} $<TARGET_OBJECTS:tridcpu_sp_mpi> $<TARGET_OBJECTS:tridcpu_dp_mpi>)
    add_library(${PROJECT_NAME}::tridcpu_mpi ALIAS tridcpu_mpi)
    target_link_libraries(
      tridcpu_sp_mpi
      PRIVATE ${CPU_LIBS}
      PUBLIC MPI::MPI_CXX)
    target_link_libraries(
      tridcpu_dp_mpi
      PRIVATE ${CPU_LIBS}
      PUBLIC MPI::MPI_CXX)
    target_link_libraries(tridcpu_mpi PUBLIC ${CPU_INTERFACE_LIBS} MPI::MPI_CXX)

    set_property(TARGET tridcpu_sp_mpi tridcpu_dp_mpi PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(tridcpu_sp_mpi PRIVATE -DFPPREC=0)
    target_compile_definitions(tridcpu_dp_mpi PRIVATE -DFPPREC=1)
    install(
      TARGETS tridcpu_mpi
      EXPORT tridcpuMPITarget
      LIBRARY DESTINATION ${PROJECT_NAME}/lib
      ARCHIVE DESTINATION ${PROJECT_NAME}/lib)
    install(FILES ${PROJECT_SOURCE_DIR}/include/trid_common.h ${PROJECT_SOURCE_DIR}/include/trid_mpi_cpu.h
                  ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp DESTINATION ${PROJECT_NAME}/include)
    install(
      EXPORT tridcpuMPITarget
      FILE tridcpu_mpiTarget.cmake
      NAMESPACE ${PROJECT_NAME}::
      DESTINATION ${PROJECT_NAME}/lib/cmake)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_CPU)
