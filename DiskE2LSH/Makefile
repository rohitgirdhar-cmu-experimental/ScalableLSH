CPPFLAGS += -std=c++0x -fopenmp -g -O3
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options
LIBS += -L /exports/cyclops/software/utils/leveldb -L /exports/cyclops/software/cpp/boost_1_57_0/stage/lib
INC += -I ~/exports/cyclops/software/utils/leveldb/include/ -I /exports/cyclops/software/cpp/boost_1_57_0 -I Eigen/

all: main.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc.hpp Table.hpp Resorter.hpp Makefile LSH.hpp
	$(CXX) $(CPPFLAGS) $(INC) $< -o $@ $(LIBS) $(LDFLAGS)

