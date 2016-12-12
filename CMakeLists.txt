cmake_minimum_required(VERSION 3.6)
project(GaussianElimination)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    CMakeLists.txt
    GaussianElimination.c
    GaussianEliminationUsingMPI
    GaussianEliminationUsingMPI.c
        GaussianEliminationUsingOpenMP.c
    GaussianEliminationUsingPThread.c
    get_data.c
    get_data_new.c
    matrixNorm.cu
    mpi
    mpi.c
    mpi.o
    PThreadHelloWorld.c MatrixNormSerial.c MatrixNormalizationCuda.cu MatrixNormSowmya.co MatrixNormHarshBackUp.cu)

add_executable(GaussianElimination ${SOURCE_FILES})