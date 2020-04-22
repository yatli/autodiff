#include "common.hpp"
#include "data.hpp"

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
    //loss.expr->rewrite();
    loss.expr->propagate(0.01);
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

int main(int argc, char* argv[]) {
  cout << "loading data..." << endl;
  auto ptrain = load_train<qnum8_t>();
  cout << "initializing network..." << endl;
  mlp_t<qnum8_t> net(28*28, 128, 10);

  for (int epoch = 0;; ++epoch) {
    cout << "epoch " << epoch << endl;
    int smp = 0;
    for (auto i : ptrain->shuffle()) {
      auto img = ptrain->imgs[i];
      auto label = ptrain->labels[i];
      auto label_predict = net.forward(img);
      auto loss = loss_crossent(label, label_predict);
      //debug_dump(label);
      //debug_dump(label_predict);
      cout << "loss = " << setw(10) << loss << ", sample " << setw(5) << smp << "/60000" << endl;
      smp++;
      net.backward(loss);
    }
  }

  return 0;
}
