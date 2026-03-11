cmake_minimum_required(VERSION 3.18)
project(simple_ml_ext)

# Find Python and Pybind11 packages
execute_process(COMMAND CMD /c python3 -m pybind11 --cmakedir OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PyBind11 CMake Directory: ${PYBIND11_CMAKE_DIR}")
set(pybind11_DIR "${PYBIND11_CMAKE_DIR}") 
find_package(pybind11 CONFIG REQUIRED)

# Add the Pybind11 module
pybind11_add_module(simple_ml_ext MODULE src\\simple_ml_ext.cpp)