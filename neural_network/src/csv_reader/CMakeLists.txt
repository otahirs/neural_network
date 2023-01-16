cmake_minimum_required(VERSION 3.10)
project(csv_reader C CXX)

set(SOURCE_FILES
    csv_reader.h
    csv_reader.cpp
)

add_library(csv_reader SHARED STATIC ${SOURCE_FILES})

install(TARGETS csv_reader DESTINATION ${NEURAL_NETWORK_INSTALL_LIB_DIR})
install(FILES csv_reader.h DESTINATION ${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
