#ifndef TABLE_HPP
#define TABLE_HPP

#include "LSHFunc_ITQ.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp> 
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/unordered_set.hpp>
#include <boost/functional/hash.hpp>

class Table {
  friend class boost::serialization::access;
  LSHFunc_ITQ lshFunc;
  unordered_map<vector<bool>, unordered_set<long long int>, hash<vector<bool>>> index;
public:
  Table(int k) : lshFunc(k) {}
  Table() {} // used for serializing
  template <typename T>
  void train(const vector<vector<T>>& sampleData) {
    lshFunc.train(sampleData);
  }
  void insert(const vector<float>& feat, long long int label) {
    vector<bool> hash;
    lshFunc.computeHash(feat, hash);

    auto pos = index.find(hash);
    if (pos == index.end()) {
      unordered_set<long long int> lst; 
      lst.insert(label);
      index[hash] = lst;
    } else {
      pos->second.insert(label);
    }
  }
  bool search(const vector<float>& feat, unordered_set<long long int>& output) const {
    vector<bool> hash;
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

