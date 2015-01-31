#include <iostream>
#include <boost/program_options.hpp>
#include "storage/DiskVector.hpp"
#include "LSH.hpp"
#include "Resorter.hpp"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

#define RESDIR "tempdata/matches/"

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
  vector<float> feat;
  
  LSH l(200, 15, 9216);
  for (int i = 0; i < 9000; i++) {
    temp.Get(i, feat);
    l.insert(feat, i);
    if (i % 1000 == 0)
      cout << "done for " << i << endl;
  }
  unordered_set<int> t2;
  DiskVector<vector<float>> q("storage/marked_feats_normalized");
  for (int i = 0; i < 20; i++) {
    q.Get(i, feat);
    l.search(feat, t2);
    vector<pair<float, int>> res = Resorter::resort(t2, temp, feat);
    ofstream fout(string(RESDIR) + "/" + to_string(i + 1) + ".txt");
    for (auto it = res.begin(); it != res.end(); it++) {
      fout << it->second + 1 << endl; 
//      cout << i << "/" << it->second + 1 << " : " << it->first << endl;
    }
    fout.close();
  }

  return 0;
}

