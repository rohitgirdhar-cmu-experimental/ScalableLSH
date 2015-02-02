#include "DiskVector.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

#define FPATH "/exports/cyclops/work/003_Backpage/dataset/backpage/features/structured/"
#define IMGSLST "/exports/cyclops/work/003_Backpage/dataset/backpage/TrainSet.txt"
#define OUTDIR "nevada_feat_normed_stor"

void normalize(vector<float>& feat) {
  float norm = 0;
  for (auto it = feat.begin(); it != feat.end(); it++) {
    norm += *it * (*it);
  }
  norm = sqrt(norm);
  for (auto it = feat.begin(); it != feat.end(); it++) {
    *it = *it / norm; 
  }

}

void readAndIndex(fs::path fpath, fs::path imgsLst) {
  DiskVector<vector<float>> d(OUTDIR);
  ifstream ifs(imgsLst.string(), ios::in);
  string line;
  float el;
  int i = 0;
  while (getline(ifs, line)) {
    fs::path featpath = fpath / fs::change_extension(fs::path(line), fs::path(".txt"));
    vector<float> feat;
    ifstream ifs2(featpath.string());
    if (!ifs2.is_open()) {
      cerr << "Unable to open " << featpath.string() << ". Exitting.." << endl;
      return;
    }
    string line;
    getline(ifs2, line);
    istringstream iss(line);
    while (iss >> el) {
      feat.push_back(el);
    }
    normalize(feat);
    d.Put(i, feat);
    i++;
    if (i % 100 == 0) {
      cout << "done for " << i << endl;
    }
  }
}

int main() {
  readAndIndex(fs::path(FPATH), fs::path(IMGSLST));
  return 0;
}

