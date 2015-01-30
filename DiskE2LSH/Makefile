CPPFLAGS += -std=c++0x
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options
LIBS += -L ~/Software/utils/leveldb/ -L ~/Software/utils/boost/lib/
INC += -I ~/Software/utils/leveldb/include/ -I ~/Software/utils/boost/include/ -I Eigen_3.2.4/

all: main.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc.hpp
	$(CXX) \
		$(CPPFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2

