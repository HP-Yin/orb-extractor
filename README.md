# orb-extractor
Keypoint extractor transfered from ORB-SLAM2

## Usage
```
mkdir build
cd build
cmake ..
make -j
./test_orb
```

Set the path of your image at test_orb.cpp
```
std::string img1str = "../kitti0_l.png";
```
