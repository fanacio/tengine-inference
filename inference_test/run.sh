cd build
rm -rf *
cmake ..
make
cd release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/firefly/data/pack/Tengine-tengine-lite/3rdparty/acl/lib
./shoot_v1.0 -m ../../models/shootmodelv2_v1.0.tmfile -i ../../images/0002.jpg
