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
    loss.expr->propagate(0.01);
  }

  void learn() {
    for (int i = 0; i < sz_input; ++i) {
      for (int j = 0; j < sz_hidden; ++j) {
        auto& e = w1(j, i + 1);
        e = (e - e.grad())->val;
        e.seed();
      }
    }
    for (int i = 0; i < sz_hidden; ++i) {
      for (int j = 0; j < sz_output; ++j) {
        auto& e = w1(j, i + 1);
        e = (e - e.grad())->val;
        e.seed();
      }
    }
  }
};

template<typename T> void train() {
  cout << "loading data..." << endl;
  auto ptrain = load_train<T>();
  cout << "initializing network..." << endl;
  mlp_t<T> net(28*28, 128, 10);

  for (int epoch = 0;; ++epoch) {
    cout << "epoch " << epoch << endl;
    auto samples = ptrain->shuffle();
    auto batch_size = 8;
    for (auto i = 0; i < ptrain->size(); i += batch_size) {
      std::vector<std::thread> threads;
      std::vector<T> losses(batch_size);
      for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
        threads.emplace_back([&](auto idx){
          auto img = ptrain->imgs[idx];
          auto label = ptrain->labels[idx];
          auto label_predict = net.forward(img);
          auto loss = loss_crossent(label, label_predict);
          losses[idx - i] = loss.expr->val;
          net.backward(loss);
        }, i+j);
      }

      for(auto &t: threads) {
        t.join();
      }
      net.learn();
      cout << "loss = " << setw(10) << losses[0] << ", sample " << setw(5) << i << "/60000" << endl;
    }
  }
}

int main(int argc, char* argv[]) {
  std::string type = argv[1];

  if(type == "qnum16") train<qnum16_t>();
  else if (type == "qnum32") train<qnum32_t>();
  else if (type == "f32") train<float>();
  else if (type == "f64") train<double>();
  else {
    cout << "unknown data type " << type << "." << endl;
  }

  return 0;
}
