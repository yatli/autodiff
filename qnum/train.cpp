#include "common.hpp"
#include "data.hpp"
#include <thread>

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

  vec forward(const vec& x) {
    auto bx = withb(x);
    auto hx = withb(fc_layer(bx, w1, act_relu));
    auto ox = fc_layer(hx, w2, act_softmax);
    return ox;
  }

  void backward(const Variable<T>& loss) {
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
    for (int j = 0; j < sz_hidden; ++j) {
      for (int i = 0; i < sz_input; ++i) {
        auto& e = w1(j, i + 1);
        e = (e - e.grad() * rate)->val;
        e.seed();
      }
    }
    for (int j = 0; j < sz_output; ++j) {
      for (int i = 0; i < sz_hidden; ++i) {
        auto& e = w2(j, i + 1);
        e = (e - e.grad() * rate)->val;
        e.seed();
      }
    }
  }
};

constexpr bool parallel = true;

template<typename T> void train() {
  cout << "loading data..." << endl;
  auto ptrain = load_train<T>();
  auto ptest = load_test<T>();
  cout << "initializing network..." << endl;
  mlp_t<T> net(28*28, 128, 10);

  for (int epoch = 0;; ++epoch) {
    cout << "epoch " << epoch << endl;
    auto samples = ptrain->shuffle();
    auto batch_size = 1;
    if constexpr(parallel) {
      batch_size = 8;
    }
    auto run = [&](const VectorXtvar<T> &img, 
                  const VectorXtvar<T> &label, 
                  double& loss_store,
                  int& correct_store,
                  bool backward) {
      auto label_predict = net.forward(img);
      auto loss = loss_crossent(label, label_predict);
      loss_store = static_cast<double>(loss.expr->val);
      correct_store = (argmax(label) == argmax(label_predict));
      if (backward) {
        net.backward(loss);
      }
    };

    std::vector<double> losses(batch_size);
    std::vector<int> corrects(batch_size);
    int total_correct = 0;
    double total_loss = 0.0;

    for (auto i = 0; i < ptrain->size(); i += batch_size) {
      if constexpr(parallel) {
        std::vector<std::thread> threads;
        for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
          threads.emplace_back([&](auto idx){
            run(ptrain->imgs[idx], ptrain->labels[idx], losses[idx - i], corrects[idx - i], true);
          }, i+j);
        }
        for(auto &t: threads) {
          t.join();
        }
      } else {
        run(ptrain->imgs[i], ptrain->labels[i], losses[0], corrects[0], true);
      }

      net.learn(T(0.01));

      // update & print stats
      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }

      auto current_acc = total_correct / (double)i;
      auto current_loss = total_loss / i;

      cout 
        << "smploss = " << setw(10) << losses[0] 
        << ", avgloss = " << setw(10) << current_loss
        << ", acc = " << setw(10) << current_acc
        << ", sample " << setw(5) << i << "/60000" 
        << endl;
    }
  }
}

int main(int argc, char* argv[]) {
  std::string type = argv[1];

  if(type == "q8") train<qnum8_t>();
  else if(type == "q16") train<qnum16_t>();
  else if (type == "q32") train<qnum32_t>();
  else if (type == "q64") train<qnum64_t>();
  else if (type == "f32") train<float>();
  else if (type == "f64") train<double>();
  else {
    cout << "unknown data type " << type << "." << endl;
  }

  return 0;
}
