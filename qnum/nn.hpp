#pragma once
#include "common.hpp"

template<typename T>
struct nn_t {
  using mat = MatrixXtvar<T>;
  using vec = VectorXtvar<T>;
  using var = autodiff::reverse::Variable<T>;

  /// to be filled in instance ctor
  std::vector<var*> params;

  void register_params(vec& v) {
    for(int i=0;i<v.size(); ++i) {
      params.push_back(&v(i));
    }
  }

  void register_params(mat& m) {
    for(int r = 0; r < m.rows(); ++r) {
      for(int c = 0; c < m.cols(); ++c) {
        params.push_back(&m(r,c));
      }
    }
  }

  void save(const char* name) {
    FILE* fp = fopen(name, "wb");

    std::vector<T> v(params.size());
    for(int i=0; i<params.size(); ++i) {
      v[i] = params[i]->expr->val;
    }

    fwrite(v.data(), v.size() * sizeof(T), 1, fp);
    fclose(fp);
  }

  void load(const char* name) { 
    FILE* fp = fopen(name, "rb");

    std::vector<T> v(params.size());

    fread(v.data(), v.size() * sizeof(T), 1, fp);
    fclose(fp);

    for(int i=0; i<params.size(); ++i) {
      params[i]->expr->val = v[i];
    }
  }

  void backward(const var& loss) {
    // first check for poisonous loss values
    if constexpr(is_qnum<T>::value) {
      if(loss.expr->val.saturated()) {
        return;
      }
    } else if constexpr(is_std_float<T>::value) {
      if(!std::isnormal(loss.expr->val)) {
        return;
      }
    } else if constexpr(is_flexfloat<T>::value) {
      if(!std::isnormal((double)loss.expr->val)) {
        return;
      }
    }

    //cout << "rewrite" << endl;
    loss.expr->rewrite();
    std::vector<autodiff::reverse::Expr<T>*> vec;
    loss.expr->topology_sort(vec);
    loss.expr->grad = T(1.0);
    for(auto it = vec.rbegin(); it != vec.rend(); ++it) {
      (*it)->propagate_step();
    }
    //loss.expr->propagate(T(1.0));
  }

  void seed() {
    for (var* x : params) {
      x->seed();
    }
  }

  void learn(const T& rate) {
    for (var* x : params) {
      x->expr->val -= x->grad() * rate;
    }
  }

  void check_histogram() {
    constexpr int nhist = 20;
    std::vector<int> histogram(nhist);
    std::vector<T> bucket_bounds(nhist);
    double incr = 2.0 / nhist;
    T cur = -1.0;
    for(int i=0;i<nhist;++i) {
      bucket_bounds[i] = cur;
      cur += incr;
    }
    bucket_bounds[nhist/2] = 0;

    for(auto &x: params) {
      auto &e = *x;
      for(int k=0;k<nhist;++k) {
        if (e.expr->val < bucket_bounds[k]) {
          histogram[k]++;
          break;
        }
      }
    }
    std::cout << "[DEBUG] Histogram :" ;
    for(int i=0;i<nhist;++i) {
      std::cout << " " << std::setw(5) << histogram[i];
    }
    std::cout << std::endl;
  }

  void check_saturation() {

    if constexpr(is_qnum<T>::value) {
      int ntotal = 0;
      int nsat = 0;
      int nsat_grad = 0;

      for(auto &x: params) {
        auto &e = *x;
        ++ntotal;
        if (e.expr->val.saturated()) ++ nsat;
        if (e.grad().saturated()) ++ nsat_grad;
      }

      std::cout << "[DEBUG] Saturation: " << nsat << " / " << nsat_grad << " / " << ntotal << std::endl;
    }
  }

  void dump_weights() {
    for(auto &x: params) {
      auto &e = *x;
      std::cout << "[DUMP] " << e.expr->val << std::endl;
    }
  }

  virtual vec forward(const vec& x) = 0;
};
