#include "DiskVector.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

#define FPATH "/home/rgirdhar/Work/Projects/001_DetectionRetrieval/BgMatchesObjDet/tempdata/selsearch_feats_all_normalized.txt"

void readAndIndex(fs::path fpath) {
  DiskVector<vector<float>> d("selsearch_feats_normalized");
  ifstream ifs(fpath.string().c_str(), ios::in);
  string line;
  float el;
  int i = 0;
  while (getline(ifs, line)) {
    vector<float> feat;
    istringstream iss(line);
    while (iss >> el) {
      feat.push_back(el);
    }
    d.Put(i, feat);
    i++;
    cout << "done for " << i << endl;
  }
}

int main() {
  readAndIndex(FPATH);
  return 0;
}

