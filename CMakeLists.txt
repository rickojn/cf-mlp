cmake_minimum_required(VERSION 3.10)
project(MLP, CXX)


# set(CMAKE_C_FLAGS_DEBUG "-g -fopenmp -Wall -O2 -mavx -mavx2 -mfma")
# set(CMAKE_C_FLAGS_RELEASE "-fopenmp -Wall -O2 -DNDEBUG -mavx -mavx2 -mfma -march=native")

# Set C++ flags for AVX and optimizations
set(CMAKE_CXX_FLAGS_DEBUG "-g -fopenmp -Wall -O2 -mavx -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -Wall -O2 -DNDEBUG -mavx -mavx2 -mfma -march=native")

add_executable(mlp ./src/mlp.cpp)

target_compile_options(mlp PRIVATE -fopenmp -Wall -O2 -mavx -mavx2 -mfma -march=native)

target_link_libraries(mlp m)