#include "DiskVector.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

#define FPATH "/home/rgirdhar/Work/Projects/001_DetectionRetrieval/BgMatchesObjDet/tempdata/selsearch_feats_all_normalized.txt"

void readAndPrint(fs::path fpath) {
  DiskVector<vector<float>> d("selsearch_feats_normalized");
  vector<float> temp;
  d.Get(10, temp);
  cout << temp.size();
  for (int i = 0; i < temp.size(); i++) {
    cout << temp[i] << " ";
  }
}

int main() {
  readAndPrint(FPATH);
  return 0;
}

