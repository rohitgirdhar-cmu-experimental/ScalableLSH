CPPFLAGS += -std=c++0x -fopenmp -g -O2 -llmdb
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options -lboost_filesystem -lglog -lz -lzmq -llmdb
LIBS += -L/mnt/data/Softwares/CPP/boost/boost_1_57_0/stage/lib/ -L/mnt/data/Softwares/CPP/zmq/install/lib/ -L/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/
INC += -I/mnt/data/Softwares/CPP/boost/boost_1_57_0/ -IEigen/ -I/mnt/data/Softwares/CPP/zmq/install/include/ -I/mnt/data/Softwares/CPP/cppzmq/ -I/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/

all: buildIndex.bin main.bin computePairwiseSim.bin convertHashesToTxt.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc_ITQ.hpp Table.hpp Resorter.hpp LSH.hpp config.hpp storage/DiskVectorLMDB.hpp
	$(CXX) $(CPPFLAGS) $(INC) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	rm *.bin

