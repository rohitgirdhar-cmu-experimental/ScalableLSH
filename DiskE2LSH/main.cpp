#include <iostream>
#include <boost/program_options.hpp>
#include "storage/DiskVector.hpp"
#include "Table.hpp"
#include "LSHFunc.hpp"
#include "LSH.hpp"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>()->required(),
     "Path to leveldb where the normalized data is stored")
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
  string OUTDIR = vm["datapath"].as<string>();

  DiskVector<vector<float>> temp(OUTDIR);
  LSHFunc L(20, 9216);
  vector<float> feat;
  temp.Get(5, feat);
  
  LSH l(20, 10, 9216);
  l.insert(feat, 1);
  unordered_set<int> t2;
  l.search(feat, t2);
  for (auto it = t2.begin(); it != t2.end(); it++) {
    cout << *it << " ";
  }

  return 0;
}

