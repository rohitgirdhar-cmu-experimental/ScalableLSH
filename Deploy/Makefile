# Segent the following:
#CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
#CAFFE_ROOT := /home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN/external/caffe_dev_MemLayerWithMat
CAFFE_ROOT := ../../ComputeFeatures/Features/CNN/external/caffe
FEAT_EXT_CODE_ROOT := ../../ComputeFeatures/Features/CNN/ver2
CUDA_ROOT := /usr/local/cuda
CPU_ONLY := 0

# for the zmq
INC += -I/mnt/data/Softwares/CPP/zmq/install/include/
INC += -I/mnt/data/Softwares/CPP/cppzmq
LIBS += -L/mnt/data/Softwares/CPP/zmq/install/lib/
LDFLAGS += -lzmq -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
LIBS += -L$(CAFFE_ROOT)/build/lib -L/usr/local/lib/ -L/mnt/data/Softwares/Vision/opencv/install/lib/ -L/mnt/data/Softwares/CPP/boost/boost_1_57_0/stage/lib -L/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/ -L /usr/lib/x86_64-linux-gnu
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -lboost_thread -llmdb -lz -lcurl
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/ -I/mnt/data/Softwares/Vision/opencv/install/include -I/mnt/data/Softwares/CPP/boost/boost_1_57_0 -I/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/ -I /usr/include/

# for the feature extraction utils.hpp
INC += -I$(FEAT_EXT_CODE_ROOT)

# for the search code
SEARCH_CODE_PATH=../DiskE2LSH
INC += -I$(SEARCH_CODE_PATH) -I$(SEARCH_CODE_PATH)/Eigen



ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
