CPPFLAGS += -std=c++0x -fopenmp -g -O3
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options
LIBS += -L ~/software/leveldb/ -L ~/software/boost/libs/
INC += -I ~/software/leveldb/include/ -I ~/software/boost/include/ -I Eigen_3.2.4/

all: main.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc.hpp Table.hpp Resorter.hpp Makefile LSH.hpp
	$(CXX) \
		$(CPPFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS)

