#pragma once
#include "common.hpp"

template<typename T>
struct nn_t {
  using mat = MatrixXtvar<T>;
  using vec = VectorXtvar<T>;

  virtual void save(const char* name) = 0;
  virtual void load(const char* name) = 0;
  virtual vec forward(const vec& x) = 0;

  void backward(const autodiff::reverse::Variable<T>& loss) {
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
  }

  virtual void learn(const T& rate) = 0;
  virtual void check_histogram() = 0;
  virtual void check_saturation() = 0;
  virtual void dump_weights() = 0;
};
