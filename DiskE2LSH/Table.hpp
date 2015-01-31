#ifndef TABLE_HPP
#define TABLE_HPP

#include "LSHFunc.hpp"
#include <vector>
#include <map>
#include <unordered_set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp> 
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/unordered_set.hpp> 

class Table {
  friend class boost::serialization::access;
  LSHFunc lshFunc;
  map<vector<int>, unordered_set<int>> index;
public:
  Table(int k, int dim) : lshFunc(k, dim) {}
  Table() {} // used for serializing
  void insert(const vector<float>& feat, int label) {
    vector<int> hash;
    lshFunc.computeHash(feat, hash);

    auto pos = index.find(hash);
    if (pos == index.end()) {
      unordered_set<int> lst; 
      lst.insert(label);
      index[hash] = lst;
    } else {
      pos->second.insert(label);
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
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & lshFunc;
    ar & index;
  }
};

#endif

