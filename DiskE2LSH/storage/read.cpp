#include "DiskVector.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

void readAndPrint() {
  DiskVector<vector<float>> d("selsearch_feats_normalized");
  vector<float> temp;
  d.Get(0, temp);
  cout << temp.size();
  for (int i = 0; i < temp.size(); i++) {
    cout << temp[i] << " ";
  }
}

int main() {
  readAndPrint();
  return 0;
}

