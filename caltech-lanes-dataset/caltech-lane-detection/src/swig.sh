g++ IPM.cpp InversePerspectiveMapping.cc mcv.cc -o a -std=c++11 `pkg-config --libs opencv`
g++ -c -fPIC IPM.cpp InversePerspectiveMapping.cc mcv.cc -std=c++11 `pkg-config --libs opencv`
swig -c++ -python IPM.i
g++ -c -fPIC IPM_wrap.cxx  -I/usr/include/python2.7 -I/usr/lib/python2.7 `pkg-config --libs opencv`
g++ -shared -Wl,-soname,_IPM.so -o _IPM.so IPM.o IPM_wrap.o InversePerspectiveMapping.o mcv.o `pkg-config --libs opencv`
