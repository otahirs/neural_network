cmake_minimum_required(VERSION 3.10)
project(csv_writer C CXX)

set(SOURCE_FILES
    csv_writer.h
    csv_writer.cpp
)

add_library(csv_writer SHARED STATIC ${SOURCE_FILES})

install(TARGETS csv_writer DESTINATION ${NEURAL_NETWORK_INSTALL_LIB_DIR})
install(FILES csv_writer.h DESTINATION ${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
