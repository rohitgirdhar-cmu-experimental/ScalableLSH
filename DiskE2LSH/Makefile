CPPFLAGS += -std=c++0x -fopenmp -g -O2 -llmdb
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options -lboost_filesystem -lglog
LIBS += -L ~/Software/utils/leveldb/ -L /home/rgirdhar/data/Software/cpp/boost_1_57_0/stage/lib/
INC += -I ~/Software/utils/leveldb/include/ -I /home/rgirdhar/data/Software/cpp/boost_1_57_0/ -I Eigen/

all: main.bin buildIndex.bin buildDiskIndex.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc.hpp Table.hpp Resorter.hpp Makefile LSH.hpp
	$(CXX) $(CPPFLAGS) $(INC) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	rm *.bin

