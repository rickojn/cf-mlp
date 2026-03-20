export DATA_PATH=/home/xxxxxx/coding/mnist-data/
export MODELS_PATH=/home/xxxxxx/coding/cf-mlp/models/
export Torch_DIR=/home/xxxxxx/libtorch/libtorch

# to build debug version:
# cmake -DCMAKE_BUILD_TYPE=Debug -B build-debug
# cmake --build build-debug

# to build release version:
# cmake -DCMAKE_BUILD_TYPE=Release -B build-release
# cmake --build build-release

# ./build/inference mnist-data/t10k-images.idx3-ubyte mnist-data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte 
