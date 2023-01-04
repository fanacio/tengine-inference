if [ ! -d "./build-aclm" ]; then
    mkdir build-convert_tool
fi
rm -r build-convert_tool/*
cd build-convert_tool
cmake -DTENGINE_BUILD_CONVERT_TOOL=ON -DTENGINE_OPENMP=ON ..
make -j32
make install
