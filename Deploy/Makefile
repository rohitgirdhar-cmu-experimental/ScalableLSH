# Set the following:
#CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
#CAFFE_ROOT := /home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN/external/caffe_dev_MemLayerWithMat
CAFFE_ROOT := /home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN/external/caffe
FEAT_EXT_CODE_ROOT := /srv2/rgirdhar/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN/ver2
CUDA_ROOT := /usr/local/cuda
CPU_ONLY := 0

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
LIBS += -L$(CAFFE_ROOT)/build/lib -L/usr/local/lib/ -L/home/rgirdhar/data/Software/cpp/boost_1_57_0/stage/lib/ `pkg-config --libs /srv2/rgirdhar/Software/vision/opencv3_install/lib/pkgconfig/opencv.pc` 
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -llmdb -lz -lcurl
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/ -I/srv2/rgirdhar/Software/vision/opencv3_install/include -I/home/rgirdhar/data/Software/cpp/boost_1_57_0/ `pkg-config --cflags /srv2/rgirdhar/Software/vision/opencv3_install/lib/pkgconfig/opencv.pc`

# for the feature extraction utils.hpp
INC += -I$(FEAT_EXT_CODE_ROOT)

# for the search code
SEARCH_CODE_PATH=../DiskE2LSH
INC += -I$(SEARCH_CODE_PATH) -I$(SEARCH_CODE_PATH)/Eigen

# for the zmq
INC += -I/home/rgirdhar/data/Software/cpp/zeromq/install/include/
INC += -I/home/rgirdhar/data/Software/cpp/zeromq/cppzmq/
LIBS += -L/home/rgirdhar/data/Software/cpp/zeromq/install/lib/
LDFLAGS += -lzmq


ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif

CXXFLAGS += -DUSE_OPENCV

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
