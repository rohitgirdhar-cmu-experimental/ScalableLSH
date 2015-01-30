CPPFLAGS += -std=c++0x
LDFLAGS += -lleveldb -lboost_serialization -lboost_system
LIBS += -L ~/Software/utils/leveldb/ -L ~/Software/utils/boost/lib/
INC += -I ~/Software/utils/leveldb/include/ -I ~/Software/utils/boost/include/ -I ../Eigen_3.2.4

all: main.bin read.bin

%.bin: %.cpp DiskVector.hpp
	$(CXX) \
		$(CPPFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2

