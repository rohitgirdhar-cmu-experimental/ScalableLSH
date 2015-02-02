CPPFLAGS += -std=c++0x
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_filesystem
LIBS += -L /exports/cyclops/software/utils/leveldb -L /exports/cyclops/software/cpp/boost_1_57_0/stage/lib
INC += -I ~/exports/cyclops/software/utils/leveldb/include/ -I /exports/cyclops/software/cpp/boost_1_57_0 -I Eigen/

all: main.bin read.bin

%.bin: %.cpp DiskVector.hpp
	$(CXX) \
		$(CPPFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2

