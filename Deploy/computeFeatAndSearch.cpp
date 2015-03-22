/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
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
// for server
#include <zmq.h>

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void readFromURL(char*, Mat&);

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
    ("featstor,s", po::value<string>()->required(),
     "Path to feature store")
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

  
  // Read the search index
  LOG(INFO) << "Reading the search index...";
  ifstream ifs(vm["index"].as<string>(), ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  std::shared_ptr<LSH> l(new LSH(0,0,0));
  ia >> *l;
  ifs.close();
  LOG(INFO) << "Done.";

  LOG(INFO) << "Setting up the server...";
  auto featstor = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
      new DiskVectorLMDB<vector<float>>(vm["featstor"].as<string>(), 1));

  //  Socket to talk to clients
  void *context = zmq_ctx_new();
  void *responder = zmq_socket(context, ZMQ_REP);
  int rc = zmq_bind(responder, "tcp://*:5555");
  assert (rc == 0);

  LOG(INFO) << "Server Ready";

  char buffer[1000], outbuf[1000];
  ostringstream oss;
  while (true) {
    oss.clear();
    zmq_recv (responder, buffer, 1000, 0);
    LOG(INFO) << "Recieved: " << buffer;

    vector<Mat> Is;
    Mat I;
    readFromURL(buffer, I);
    if (!I.data) {
      oss << "Unable to read " << buffer;
      zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
      continue;
    }
    vector<Rect> bboxes;
    bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    resize(I, I, Size(256, 256));
    Is.push_back(I);
    vector<vector<float>> feats;
    computeFeatures<float>(caffe_test_net, Is, LAYER, BATCH_SIZE, feats);
    l2NormalizeFeatures(feats);

    unordered_set<int> init_matches;
    vector<pair<float, int>> res;
    l->search(feats[0], init_matches);
    Resorter::resort_multicore(init_matches, featstor, feats[0], res);
    for (int i = 0; i < min(res.size(), (size_t) 20); i++) {
      oss << res[i].first << ":" << res[i].second << ',';
    }
    zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
  }
  return 0;
}

void readFromURL(char* url, Mat& I) {
  VideoCapture cap(url);
  if (cap.isOpened()) {
    cap >> I;
  }
}

