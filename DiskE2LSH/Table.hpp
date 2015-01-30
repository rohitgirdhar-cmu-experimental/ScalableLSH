#ifndef TABLE_HPP
#define TABLE_HPP

#include "LSHFunc.hpp"
#include <vector>
#include <map>
#include <unordered_set>

class Table {
  LSHFunc lshFunc;
  map<vector<int>, unordered_set<int>> index;
public:
  Table(int k, int dim) : lshFunc(k, dim) {}
  void insert(const vector<float>& feat, int val) {
    vector<int> hash;
    lshFunc.computeHash(feat, hash);
    auto pos = index.find(hash);
    if (pos == index.end()) {
      unordered_set<int> lst; 
      lst.insert(val);
      index[hash] = lst;
    } else {
      pos->second.insert(val);
    }
  }
  bool search(const vector<float>& feat, unordered_set<int>& output) {
    vector<int> hash;
    lshFunc.computeHash(feat, hash);
    auto pos = index.find(hash);
    if (pos != index.end()) {
      output = pos->second;
      return true;
    }
    return false;
  }
};

#endif

