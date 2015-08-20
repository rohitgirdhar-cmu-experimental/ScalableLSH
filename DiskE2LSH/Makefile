CPPFLAGS += -std=c++0x -fopenmp -g -O2 -llmdb
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options -lboost_filesystem -lglog -lz
LIBS += -L ~/Software/utils/leveldb/ -L /home/rgirdhar/data/Software/cpp/boost_1_57_0/stage/lib/
INC += -I ~/Software/utils/leveldb/include/ -I /home/rgirdhar/data/Software/cpp/boost_1_57_0/ -I Eigen/

all: buildIndex.bin main.bin computePairwiseSim.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc_ITQ.hpp Table.hpp Resorter.hpp LSH.hpp config.hpp storage/DiskVectorLMDB.hpp
	$(CXX) $(CPPFLAGS) $(INC) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	rm *.bin

