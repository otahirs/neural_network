cmake_minimum_required(VERSION 3.10)

project(neural_network)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funroll-loops -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -std=c++17 ${OpenMP_CXX_FLAGS}")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(NEURAL_NETWORK_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(NEURAL_NETWORK_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(NEURAL_NETWORK_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(CSV_READER_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/csv_reader)
set(CSV_WRITER_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/csv_writer)
set(MODEL_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/model)
set(LOGGER_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/logger)
set(LINALG_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/linalg)

include_directories(${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
include_directories(${CSV_READER_HEADERS_DIR})
include_directories(${CSV_WRITER_HEADERS_DIR})
include_directories(${MODEL_HEADERS_DIR})
include_directories(${LOGGER_HEADERS_DIR})
include_directories(${LINALG_HEADERS_DIR})

add_subdirectory(src)
#add_subdirectory(test)


