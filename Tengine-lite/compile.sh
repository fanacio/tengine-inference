if [ ! -d "./build-aclm" ]; then
    mkdir build-aclm
fi
rm -r build-aclm/*
cd build-aclm
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DTENGINE_ENABLE_ACLM=ON
make -j32
make install
