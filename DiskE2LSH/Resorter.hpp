#ifndef RESORTER_HPP
#define RESORTER_HPP

#include "storage/DiskVector.hpp"
#include <Eigen/Dense>
#include <algorithm>

class Resorter {
public:
  vector<pair<float, int>> static resort(const unordered_set<int>& matches, 
      const DiskVector<vector<float>>& feats,
      const vector<float>& qfeat) {
    Eigen::MatrixXf qfeat_mat = Eigen::VectorXf::Map(&qfeat[0], qfeat.size());
    vector<vector<float>> feats_vec;
    for (auto match = matches.begin(); 
        match != matches.end(); match++) {
      vector<float> temp;
      feats.Get(*match, temp);
      feats_vec.push_back(temp);
    }
    Eigen::MatrixXf feats_mat(matches.size(), qfeat.size());
    for (int i = 0; i < feats_vec.size(); i++) {
      feats_mat.row(i) = Eigen::VectorXf::Map(&feats_vec[i][0], feats_vec[i].size());
    }
    Eigen::MatrixXf cos_scores = qfeat_mat.transpose() * feats_mat.transpose();

    vector<pair<float, int>> res;
    int i = 0;
    for (auto match = matches.begin(); match != matches.end(); match++, i++) {
      res.push_back(make_pair(cos_scores(i), *match));
    }
    sort(res.begin(), res.end());
    return res;
  }
};

#endif
