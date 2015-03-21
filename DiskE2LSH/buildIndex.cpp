#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "storage/DiskVectorLMDB.hpp"
#include "LSH.hpp"
#include "Resorter.hpp"
#include "utils.hpp"
#include "lock.hpp"
#include "config.hpp"

using namespace std;
using namespace std::chrono;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

long long getIndex(long long, int); // both must be 1 indexed

int main(int argc, char* argv[]) {
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>()->required(),
     "Path to LMDB where the data is stored")
    ("dimgslist,n", po::value<string>()->required(),
     "File with list of all images")
    ("featcount,c", po::value<string>()->default_value(""),
     "File with list of number of features in each image")
    ("save,s", po::value<string>(),
     "Path to save the hash table")
    ("nbits,b", po::value<int>()->default_value(250),
     "Number of bits in the representation")
    ("ntables,t", po::value<int>()->default_value(15),
     "Number of random proj tables in the representation")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    cerr << desc << endl;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    cerr << e.what() << endl;
    return -1;
  }
  
  // read the list of images to hash
  vector<fs::path> imgslst;
  readList(vm["dimgslist"].as<string>(), imgslst);
  vector<int> featcounts(imgslst.size(), 1); // default: 1 feat/image
  if (vm["featcount"].as<string>().length() > 0) {
    featcounts.clear();
    readList(vm["featcount"].as<string>(), featcounts);
  }
  
  std::shared_ptr<LSH> l(new LSH(vm["nbits"].as<int>(), vm["ntables"].as<int>(), 9216));
  vector<float> feat;
  
  high_resolution_clock::time_point pivot = high_resolution_clock::now();
  DiskVectorLMDB<vector<float>> tree(vm["datapath"].as<string>(), 1);
  for (int i = 0; i < imgslst.size(); i++) {
    for (int j = 0; j < featcounts[i]; j++) {
      long long idx = getIndex(i+1, j+1);
      if (!tree.Get(idx, feat)) break;
      l->insert(feat, idx);
    }
    if (i % 100 == 0) {
      high_resolution_clock::time_point pivot2 = high_resolution_clock::now();
      cout << "Done for " << i + 1  << "/" << imgslst.size()
           << " in " 
           << duration_cast<milliseconds>(pivot2 - pivot).count()
           << "ms" <<endl;
      pivot = pivot2;
    }
  }

  if (vm.count("save")) {
    cout << "Saving model to " << vm["save"].as<string>() << "...";
    cout.flush();
    ofstream ofs(vm["save"].as<string>(), ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << *l;
    cout << "done." << endl;
  }

  return 0;
}

long long getIndex(long long imid, int pos) { // imid and pos must be 1 indexed
  return imid * MAXFEATPERIMG + pos;
}

