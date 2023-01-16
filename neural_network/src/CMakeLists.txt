cmake_minimum_required(VERSION 3.10)
project(neural_network)

add_subdirectory(csv_reader)
add_subdirectory(csv_writer)
add_subdirectory(model)
add_subdirectory(logger)
add_subdirectory(linalg)
set(SOURCE_FILES main.cpp)

add_executable(neural_network ${SOURCE_FILES})
target_link_libraries(neural_network csv_reader csv_writer model logger linalg)
install(TARGETS neural_network DESTINATION ${NEURAL_NETWORK_INSTALL_BIN_DIR})
