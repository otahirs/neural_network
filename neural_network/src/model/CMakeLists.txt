cmake_minimum_required(VERSION 3.10)
project(model C CXX)

set(SOURCE_FILES
    model.h
    model.cpp
)

add_library(model SHARED STATIC ${SOURCE_FILES})

install(TARGETS model DESTINATION ${NEURAL_NETWORK_INSTALL_LIB_DIR})
install(FILES model.h DESTINATION ${NEURAL_NETWORK_INSTALL_INCLUDE_DIR})
