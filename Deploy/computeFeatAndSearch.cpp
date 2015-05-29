/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
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

#define MAXFEATPERIMG 10000

using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void readFromURL(const string&, Mat&);
string convertToFname(long long idx, const vector<fs::path>& imgslist);

int
main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif

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
    ("imgslist,q", po::value<string>()->required(),
     "File with images list")
    ("port-num,p", po::value<string>()->default_value("5555"),
     "Port to run the service on")
    ("seg-img,g", po::value<string>()->default_value(""),
     "Path to read the segmentation image from. Keep empty for full image search. "
     "On setting this, system will pool features from bg boxes")
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
  fs::path SEG_IMG_PATH = fs::path(vm["seg-img"].as<string>());
  vector<string> layers = {LAYER};
  vector<fs::path> imgslist;
  readList(vm["imgslist"].as<string>(), imgslist);

  Net<float> caffe_test_net(NETWORK_PATH.string(), caffe::TEST);
  caffe_test_net.CopyTrainedLayersFrom(MODEL_PATH.string());
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
  int rc = zmq_bind(responder, (string("tcp://*:")
        + vm["port-num"].as<string>()).c_str());
  assert (rc == 0);

  LOG(INFO) << "Server Ready";

  while (true) {
    char buffer[1000], outbuf[1000];
    ostringstream oss;
    zmq_recv (responder, buffer, 1000, 0);
    LOG(INFO) << "Recieved: " << buffer;
    high_resolution_clock::time_point st = high_resolution_clock::now();

    vector<Mat> Is;
    Mat I;
    readFromURL(string(buffer), I);
    if (!I.data) {
      oss << "Unable to read " << buffer;
      zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
      continue;
    }

    vector<Rect> bboxes;
    if (SEG_IMG_PATH.string().length() > 0) {
      Mat S; // not really used
      CNNFeatureUtils::genSlidingWindows(I.size(), bboxes);
      CNNFeatureUtils::pruneBboxesWithSeg(I.size(), SEG_IMG_PATH, bboxes, S);
    } else {
      bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    }

    for (int i = 0; i < bboxes.size(); i++) {
      Mat Itemp = I(bboxes[i]);
      resize(Itemp, Itemp, Size(256, 256));
      Is.push_back(Itemp);
    }

    high_resolution_clock::time_point read = high_resolution_clock::now();

    vector<vector<vector<float>>> feats;
    CNNFeatureUtils::computeFeaturesPipeline<float>(caffe_test_net, Is, 
        layers, BATCH_SIZE, feats, true, "avg", true);

    high_resolution_clock::time_point feat = high_resolution_clock::now();

    unordered_set<long long int> init_matches;
    vector<pair<float, long long int>> res;
    l->search(feats[0][0], init_matches);
    LOG(INFO) << "Re-sorting " << init_matches.size() << " matches";
    Resorter::resort_multicore(init_matches, featstor, feats[0][0], res);

    high_resolution_clock::time_point search = high_resolution_clock::now();

    for (int i = 0; i < min(res.size(), (size_t) 20); i++) {
      oss << res[i].first << ":" << convertToFname(res[i].second, imgslist) << ',';
    }
    zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
    LOG(INFO) << "Time taken: " << endl
              << " Read : " << duration_cast<milliseconds>(read - st).count() << "ms" << endl
              << " Ext Feat : " << duration_cast<milliseconds>(feat - read).count() << "ms" << endl
              << " Search : " << duration_cast<milliseconds>(search - feat).count() << "ms" << endl;
  }
  return 0;
}

void readFromURL(const string& url, Mat& I) {
  string temppath = "/tmp/temp-img.jpg";
  system((string("wget ") + url + " -O " + temppath).c_str());
  I = imread(temppath.c_str());
}

string convertToFname(long long idx, const vector<fs::path>& imgslist) {
  return imgslist[idx / MAXFEATPERIMG - 1].filename().string();
}

