#pragma once
#include "common.hpp"

template<typename T>
struct mlp_t {
  using mat = MatrixXtvar<T>;
  using vec = VectorXtvar<T>;
  mat w1, w2;
  int sz_input, sz_hidden, sz_output;

  mlp_t(int ninput, int nhidden, int noutput) :
    sz_input(ninput), sz_hidden(nhidden), sz_output(noutput),
    w1(mat::Random(nhidden, ninput + 1) * 0.05),
    w2(mat::Random(noutput, nhidden + 1) * 0.05) { }

  void save(const char* name) {
    FILE* fp = fopen(name, "wb");

    std::vector<T> v1(sz_hidden * (sz_input+1));
    std::vector<T> v2(sz_output * (sz_hidden + 1));

    int idx = 0;
    for(int r = 0; r < sz_hidden; ++r) {
      for(int c = 0; c < sz_input + 1; ++c) {
        v1[idx++] = w1(r, c).expr->val;
      }
    }
    idx = 0;
    for(int r = 0; r < sz_output; ++r) {
      for(int c = 0; c < sz_hidden + 1; ++c) {
        v2[idx++] = w2(r, c).expr->val;
      }
    }
    fwrite(v1.data(), v1.size() * sizeof(T), 1, fp);
    fwrite(v2.data(), v2.size() * sizeof(T), 1, fp);

    fclose(fp);
  }

  void load(const char* name) {
    FILE* fp = fopen(name, "rb");

    std::vector<T> v1(sz_hidden * (sz_input+1));
    std::vector<T> v2(sz_output * (sz_hidden + 1));

    fread(v1.data(), v1.size() * sizeof(T), 1, fp);
    fread(v2.data(), v2.size() * sizeof(T), 1, fp);
    fclose(fp);

    int idx = 0;
    for(int r = 0; r < sz_hidden; ++r) {
      for(int c = 0; c < sz_input + 1; ++c) {
        w1(r, c) = v1[idx++];
      }
    }
    idx = 0;
    for(int r = 0; r < sz_output; ++r) {
      for(int c = 0; c < sz_hidden + 1; ++c) {
        w2(r, c) = v2[idx++];
      }
    }
  }

  vec forward(const vec& x) {
    auto bx = withb(x);
    auto hx = withb(fc_layer(bx, w1, act_relu));
    auto ox = fc_layer(hx, w2, act_softmax);
    return ox;
  }

  void backward(const Variable<T>& loss) {
    // first check for poisonous loss values
    if constexpr(!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
      if(loss.expr->val.saturated()) {
        return;
      }
    } else {
      if(!std::isnormal(loss.expr->val)) {
        return;
      }
    }

    //cout << "rewrite" << endl;
    loss.expr->rewrite();
    std::deque<autodiff::reverse::Expr<T>*> vec;
    loss.expr->topology_sort(vec);
    loss.expr->grad = T(1.0);
    for(auto &x : vec) {
      x->propagate_step();
    }
  }

  void learn(const T& rate) {
    for (int r = 0; r < sz_hidden; ++r) {
      for (int c = 0; c < sz_input + 1; ++c) {
        auto& e = w1(r, c);
        e = (e - e.grad() * rate)->val;
        e.seed();
      }
    }
    for (int r = 0; r < sz_output; ++r) {
      for (int c = 0; c < sz_hidden + 1; ++c) {
        auto& e = w2(r, c);
        e = (e - e.grad() * rate)->val;
        e.seed();
      }
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

    for (int r = 0; r < sz_hidden; ++r) {
      for (int c = 0; c < sz_input; ++c) {
        auto& e = w1(r, c+1);
        for(int k=0;k<nhist;++k) {
          if (e.expr->val < bucket_bounds[k]) {
            histogram[k]++;
            break;
          }
        }
      }
    }
    for (int r = 0; r < sz_output; ++r) {
      for (int c = 0; c < sz_hidden; ++c) {
        auto& e = w2(r, c + 1);
        for(int k=0;k<nhist;++k) {
          if (e.expr->val < bucket_bounds[k]) {
            histogram[k]++;
            break;
          }
        }
      }
    }
    cout << "[DEBUG] Histogram :" ;
    for(int i=0;i<nhist;++i) {
      cout << " " << setw(5) << histogram[i];
    }
    cout << endl;
  }

  void check_saturation() {

    if constexpr(!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
      int ntotal = 0;
      int nsat = 0;
      int nsat_grad = 0;

      for (int j = 0; j < sz_hidden; ++j) {
        for (int i = 0; i < sz_input; ++i) {
          auto& e = w1(j, i + 1);
          ++ntotal;
          if (e.expr->val.saturated()) ++ nsat;
          if (e.grad().saturated()) ++ nsat_grad;
        }
      }
      for (int j = 0; j < sz_output; ++j) {
        for (int i = 0; i < sz_hidden; ++i) {
          auto& e = w2(j, i + 1);
          ++ntotal;
          if (e.expr->val.saturated()) ++ nsat;
          if (e.grad().saturated()) ++ nsat_grad;
        }
      }

      cout << "[DEBUG] Saturation: " << nsat << " / " << nsat_grad << " / " << ntotal << endl;
    }
  }

  void dump_weights() {
    for (int j = 0; j < sz_hidden; ++j) {
      for (int i = 0; i < sz_input; ++i) {
        auto& e = w1(j, i + 1);
        cout << "[DUMP] " << e.expr->val << endl;
      }
    }
    for (int j = 0; j < sz_output; ++j) {
      for (int i = 0; i < sz_hidden; ++i) {
        auto& e = w2(j, i + 1);
        cout << "[DUMP] " << e.expr->val << endl;
      }
    }
  }
};

