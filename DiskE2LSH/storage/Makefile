CPPFLAGS += -std=c++0x
LDFLAGS += -lleveldb -lboost_serialization -lboost_system
LIBS += -L ~/software/leveldb/ -L ~/software/boost/libs
INC += -I ~/software/leveldb/include/ -I ~/software/boost/include -I ../Eigen_3.2.4

all: main.bin read.bin

%.bin: %.cpp DiskVector.hpp
	$(CXX) \
		$(CPPFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2

