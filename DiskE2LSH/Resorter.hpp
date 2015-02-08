#ifndef RESORTER_HPP
#define RESORTER_HPP

#include "storage/DiskVector.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#define MAX_RESORT_BATCH_SIZE 10000 // resort these many at a time

class Resorter {
public:
  float static computeDot(vector<float> a, vector<float> b) {
    float ans = 0;
    for (auto it = a.begin(), it2 = b.begin(); 
        it != a.end() && it2 != b.end(); it++, it2++) {
      ans += (*it) * (*it2);
    }
    return ans;
  }
  void static resort(const unordered_set<int>& matches, 
      const DiskVector<vector<float>>& feats,
      const vector<float>& qfeat,
      vector<pair<float, int>>& res) {
    res.clear();
    Eigen::MatrixXf qfeat_mat = Eigen::VectorXf::Map(&qfeat[0], qfeat.size());
    
    // Batch process the scoring
    int nMatches = matches.size();
    int nBatches = ceil(nMatches * 1.0f / MAX_RESORT_BATCH_SIZE);
    auto match = matches.begin();
    vector<vector<float>> feats_vec;
    vector<pair<float, int>> res_batch;
    
    for (int batch = 0; batch < nBatches; batch++) {
      feats_vec.clear();
      res_batch.clear();
      int batchSize = 0;
      for (; match != matches.end() && batchSize < MAX_RESORT_BATCH_SIZE;
          match++, batchSize++) {
        // TODO: Avoid this, use pre-alloc of memory
        vector<float> temp;
        feats.Get(*match, temp);
        feats_vec.push_back(temp);
        // for output
        res_batch.push_back(make_pair(0.0f, *match));
      }
      Eigen::MatrixXf feats_mat(matches.size(), qfeat.size());
      for (int i = 0; i < feats_vec.size(); i++) {
        feats_mat.row(i) = Eigen::VectorXf::Map(&feats_vec[i][0], feats_vec[i].size());
      }
      Eigen::MatrixXf cos_scores = qfeat_mat.transpose() * feats_mat.transpose();

      for (int i = 0; i < res_batch.size(); i++) {
        res_batch[i].first = cos_scores(i);
      }
      res.insert(res.end(), res_batch.begin(), res_batch.end());
    }
    sort(res.begin(), res.end());
    reverse(res.begin(), res.end());
  }
};

#endif
