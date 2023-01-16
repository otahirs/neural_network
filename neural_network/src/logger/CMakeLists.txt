cmake_minimum_required(VERSION 3.10)
project(logger C CXX)

set(SOURCE_FILES
    logger.h
    logger.cpp
)

add_library(logger SHARED STATIC ${SOURCE_FILES})

install(TARGETS logger DESTINATION ${NEURAL_NETWORK_INSTALL_LIB_DIR})
install(FILES logger.h DESTINATION ${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
