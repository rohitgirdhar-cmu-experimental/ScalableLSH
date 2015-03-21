# Set the following:
#CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
CAFFE_ROOT := /home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN/external/caffe_dev_MemLayerWithMat
CUDA_ROOT := /usr/local/cuda
CPU_ONLY := 0

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
LIBS += -L$(CAFFE_ROOT)/build/lib -L/usr/local/lib/ -L/home/rgirdhar/data/Software/cpp/boost_1_57_0/stage/lib/
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -llmdb
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/ -I/usr/local/include/opencv2/ -I/home/rgirdhar/data/Software/cpp/boost_1_57_0/

# for the search code
SEARCH_CODE_PATH=/home/rgirdhar/data/Work/Code/0002_Retrieval/ScalableLSH/DiskE2LSH
INC += -I$(SEARCH_CODE_PATH) -I$(SEARCH_CODE_PATH)/Eigen

ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CPPFLAGS += -DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile utils.hpp
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
