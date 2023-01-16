cmake_minimum_required(VERSION 3.10)
project(linalg C CXX)

set(SOURCE_FILES
    linalg.h
    linalg.cpp
)

add_library(linalg SHARED STATIC ${SOURCE_FILES})

install(TARGETS linalg DESTINATION ${NEURAL_NETWORK_INSTALL_LIB_DIR})
install(FILES linalg.h DESTINATION ${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
