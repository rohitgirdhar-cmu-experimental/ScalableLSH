/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include <boost/archive/binary_oarchive.hpp>
#include "caffe/caffe.hpp"
#include "utils.hpp"
// from the search code
#include "LSH.hpp"
#include "Resorter.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int
main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  Caffe::set_phase(Caffe::TEST); // important, else will give random features

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("network-path,n", po::value<string>()->required(),
     "Path to the prototxt file")
    ("model-path,m", po::value<string>()->required(),
     "Path to corresponding caffemodel")
    ("layer,l", po::value<string>()->default_value("pool5"),
     "CNN layer to extract features from")
    ("index,i", po::value<string>()->required(),
     "Path to load search index from")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    LOG(INFO) << desc;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  
  fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
  fs::path MODEL_PATH = 
    fs::path(vm["model-path"].as<string>());
  string LAYER = vm["layer"].as<string>();

  NetParameter test_net_params;
  ReadProtoFromTextFile(NETWORK_PATH.string(), &test_net_params);
  Net<float> caffe_test_net(test_net_params);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(MODEL_PATH.string(), &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
  int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

  string IMPATH = "/home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Images/corpus/ImagesTexas/Texas_2012_10_10_1349841918000_4_0.jpg";
  vector<Mat> Is;
  Mat I = imread(IMPATH);
  if (!I.data) {
    LOG(ERROR) << "Unable to read " << IMPATH;
    return -1;
  }
  vector<Rect> bboxes;
  bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
  resize(I, I, Size(256, 256));
  Is.push_back(I);
  vector<vector<float>> output;
  computeFeatures<float>(caffe_test_net, Is, LAYER, BATCH_SIZE, output);
  l2NormalizeFeatures(output);

  // Do the search
  ifstream ifs(vm["index"].as<string>(), ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  std::shared_ptr<LSH> l(new LSH(0,0,0));
  ia >> *l;
  ifs.close();

  return 0;
}

