BASE_PATH=/home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASE_PATH}/external/caffe_dev_MemLayerWithMat/build/lib/
./computeFeatAndSearch.bin \
    -n deploy.prototxt \
    -m /home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -i ~/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/search_models/search_250bit.index

