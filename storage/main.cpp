#include "DiskVector.hpp"
#include <iostream>

using namespace std;

int main() {
  DiskVector<float> d("temp");
  vector<float> a;
  a.push_back(1);
  //d.Put(0, a);
  vector<float> b;
  d.Get(0, b);
  cout << b[0];
  return 0;
}

