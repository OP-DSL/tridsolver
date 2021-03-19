if (USE_PROFILING)
  add_library(profiling ${BUILD_TYPE} ./timing.cpp)
  target_include_directories(profiling PUBLIC ${PROJECT_SOURCE_DIR}/include)
  set_property(TARGET profiling PROPERTY POSITION_INDEPENDENT_CODE ON)
  if (BUILD_FOR_MPI)
    target_compile_definitions(profiling PRIVATE -DUSE_MPI)
    target_link_libraries(profiling MPI::MPI_CXX)
  endif (BUILD_FOR_MPI)
  target_compile_definitions(profiling PUBLIC -DPROFILING=1)
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

    add_library(tridcpu_sp OBJECT ./trid_cpu.cpp)
    add_library(tridcpu_dp OBJECT ./trid_cpu.cpp)
    add_library(tridcpu ${BUILD_TYPE} $<TARGET_OBJECTS:tridcpu_sp> $<TARGET_OBJECTS:tridcpu_dp>)
    add_library(${PROJECT_NAME}::tridcpu ALIAS tridcpu)

    target_link_libraries(tridcpu_dp PRIVATE ${CPU_LIBS})
    target_link_libraries(tridcpu_sp PRIVATE ${CPU_LIBS})
    target_link_libraries(tridcpu PUBLIC ${CPU_INTERFACE_LIBS})

    set_property(TARGET tridcpu_sp tridcpu_dp PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(tridcpu_sp PRIVATE -DFPPREC=0)
    target_compile_definitions(tridcpu_dp PRIVATE -DFPPREC=1)

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
    install(
      FILES ${PROJECT_SOURCE_DIR}/include/trid_common.h
            ${PROJECT_SOURCE_DIR}/include/trid_mpi_cpu.h
            ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp
      DESTINATION ${PROJECT_NAME}/include)
    install(
      EXPORT tridcpuMPITarget
      FILE tridcpu_mpiTarget.cmake
      NAMESPACE ${PROJECT_NAME}::
      DESTINATION ${PROJECT_NAME}/lib/cmake)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_CPU)

# cmake-format: off
if (BUILD_FOR_MIC AND CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	add_library(tridmic_offload_sp OBJECT ./trid_cpu.cpp)
	add_library(tridmic_offload_dp OBJECT ./trid_cpu.cpp)

	add_library(tridmic_native_sp OBJECT ./trid_cpu.cpp)
	add_library(tridmic_native_dp OBJECT ./trid_cpu.cpp)

	set(MICINCLUDES /software/mpss-3.5.2/src/glibc-2.14.1+mpss3.5.2/assert/ /software/mpss-3.5.2/src/glibc-2.14.1+mpss3.5.2/misc/sys/ /software/mpss-3.5.2/src/glibc-2.14.1+mpss3.5.2/include/ /software/mpss-3.5.2/src/glibc-2.14.1+mpss3.5.2/sys/ )

	target_include_directories(tridmic_offload_sp PUBLIC ${PROJECT_SOURCE_DIR}/include ./ PRIVATE ../ ${MICINCLUDES} )
	target_include_directories(tridmic_offload_dp PUBLIC ${PROJECT_SOURCE_DIR}/include ./ PRIVATE ../ ${MICINCLUDES} )

	target_include_directories(tridmic_native_sp  PUBLIC ${PROJECT_SOURCE_DIR}/include ./ PRIVATE ../ ${MICINCLUDES} )
	target_include_directories(tridmic_native_dp  PUBLIC ${PROJECT_SOURCE_DIR}/include ./ PRIVATE ../ ${MICINCLUDES} )

	target_compile_definitions(tridmic_offload_sp PRIVATE -DFPPREC=0)
	target_compile_definitions(tridmic_offload_dp PRIVATE -DFPPREC=1)
	#set_target_properties(tridmic_offload_sp PROPERTIES COMPILE_FLAGS "-fPIC -DFPPREC=0")
	#set_target_properties(tridmic_offload_dp PROPERTIES COMPILE_FLAGS "-fPIC -DFPPREC=1")

	target_compile_options(tridmic_offload_sp PRIVATE -xAVX -m64 -parallel -qopenmp -fPIC -offload -qopt-report-phase=par -opt-report-phase=offload -offload-attribute-target=mic -mkl -offload-option,mic,ld,"-L${INTEL_PATH}/mkl/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/mic -limf -lintlc -lsvml -lirng -liomp5 -loffload -lcilkrts" -I./include -I./libtrid  -D__OFFLOAD__ -L./libtrid/lib -limf -lintlc -lsvml -lirng)
	target_compile_options(tridmic_offload_dp PRIVATE -xAVX -m64 -parallel -qopenmp -fPIC -offload -qopt-report-phase=par -opt-report-phase=offload -offload-attribute-target=mic -mkl -offload-option,mic,ld,"-L${INTEL_PATH}/mkl/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/mic -limf -lintlc -lsvml -lirng -liomp5 -loffload -lcilkrts" -I./include -I./libtrid  -D__OFFLOAD__ -L./libtrid/lib -limf -lintlc -lsvml -lirng)


	#set_target_properties(tridcpu_sp PROPERTIES COMPILE_FLAGS "-DFPPREC=0")
	#set_target_properties(tridcpu_dp PROPERTIES COMPILE_FLAGS "-DFPPREC=1")

	target_compile_definitions(tridmic_native_sp PRIVATE -DFPPREC=0)
	target_compile_definitions(tridmic_native_dp PRIVATE -DFPPREC=1)
	target_compile_options(tridmic_native_sp PRIVATE -mmic -parallel -qopenmp -fPIC)
	target_compile_options(tridmic_native_dp PRIVATE -mmic -parallel -qopenmp -fPIC)

	add_library(tridmic_offload ${BUILD_TYPE} $<TARGET_OBJECTS:tridmic_offload_sp> $<TARGET_OBJECTS:tridmic_offload_dp>)
	set_target_properties(tridmic_offload PROPERTIES LINK_FLAGS -L./libtrid/lib -limf -lintlc -lsvml -lirng)

	add_library(tridmic_native ${BUILD_TYPE} $<TARGET_OBJECTS:tridmic_native_sp> $<TARGET_OBJECTS:tridmic_native_dp>)

	install(TARGETS tridmic_offload
		LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
		ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)
	install(TARGETS tridmic_native
		LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
		ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)
	install(FILES ${PROJECT_SOURCE_DIR}/include/trid_cpu.h
					${PROJECT_SOURCE_DIR}/include/trid_common.h
					${PROJECT_SOURCE_DIR}/include/trid_simd.h
					${PROJECT_SOURCE_DIR}/src/cpu/trid_mpi_cpu.hpp
					${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp
		DESTINATION ${CMAKE_BINARY_DIR}/include)
endif (BUILD_FOR_MIC AND CMAKE_CXX_COMPILER_ID MATCHES "Intel")
# cmake-format: on
