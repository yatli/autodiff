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

    fwrite(v1.data(), v1.size() * sizeof(T), 1, fp);
    fwrite(v2.data(), v2.size() * sizeof(T), 1, fp);
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
};

template<typename T> void train(double lr, int nhidden, const string& type) {
  cout << "loading data..." << endl;
  auto ptrain = load_train<T>();
  auto ptest = load_test<T>();
  cout << "initializing network..." << endl;
  mlp_t<T> net(28*28, nhidden, 10);

  for (int epoch = 0;; ++epoch) {
    cout << "[TRAIN] epoch " << epoch << endl;
    auto samples = ptrain->shuffle();
    auto batch_size = 8;
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
    auto smpidx = ptrain->shuffle();

    for (auto i = 0; i < ptrain->size(); i += batch_size) {

      if (i % 10000 == 0) {
        char buf[256];
        sprintf(buf, "%s-h%d-lr%f-epoch-%d-step-%d.dmp", type.data(), nhidden, lr, epoch, i);
        net.save(buf);
      }

      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
        threads.emplace_back([&](auto idx){
          run(ptrain->imgs[smpidx[idx]], ptrain->labels[smpidx[idx]], losses[idx - i], corrects[idx - i], true);
        }, i+j);
      }
      for(auto &t: threads) {
        t.join();
      }
      net.learn(T(lr));

      if ((i/batch_size) % 10 == 0) {
        net.check_histogram();
        // TODO check saturation, but on all nodes, not just weights
      }

      // update & print stats
      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }

      auto current_acc = total_correct / ((double)i + batch_size);
      auto current_loss = total_loss / (i+batch_size);

      cout 
        << "[TRAIN] smploss = " << setw(12) << losses[0] 
        << ", avgloss = "       << setw(12) << current_loss
        << ", acc = "           << setw(12) << current_acc
        << ", sample "          << setw(5)  << i << "/" << ptrain->size()
        << endl;

    }

    total_correct = 0;
    total_loss = 0.0;
    cout << "[TEST] epoch " << epoch << endl;
    for (auto i = 0; i < ptest->size(); i += batch_size) {
      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
        threads.emplace_back([&](auto idx){
          run(ptest->imgs[idx], ptest->labels[idx], losses[idx - i], corrects[idx - i], false);
        }, i+j);
      }
      for(auto &t: threads) {
        t.join();
      }
      // update & print stats
      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }
    }
    cout 
      << "[TEST] avgloss = " << setw(12) << total_loss / (double)ptest->size()
      << ", acc = "          << setw(12) << total_correct / (double)ptest->size()
      << endl;
  }
}

template<typename T, typename ... Args> void train_wrap(int E, Args... args)
{
  switch (E) {
    case 1:
      train<qspace_number_t<T, 1>>(args...);
      break;
    case 2:
      train<qspace_number_t<T, 2>>(args...);
      break;
    case 3:
      train<qspace_number_t<T, 3>>(args...);
      break;
    case 4:
      train<qspace_number_t<T, 4>>(args...);
      break;
    case 5:
      train<qspace_number_t<T, 5>>(args...);
      break;
    case 6:
      train<qspace_number_t<T, 6>>(args...);
      break;
    case 7:
      train<qspace_number_t<T, 7>>(args...);
      break;
    case 8:
      train<qspace_number_t<T, 8>>(args...);
      break;
    case 9:
      train<qspace_number_t<T, 9>>(args...);
      break;
    case 10:
      train<qspace_number_t<T, 10>>(args...);
      break;
    case 11:
      train<qspace_number_t<T, 11>>(args...);
      break;
    case 12:
      train<qspace_number_t<T, 12>>(args...);
      break;
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  std::string type = argv[1];
  int E = atoi(argv[2]);
  double lr = atof(argv[3]);
  int nhidden = atoi(argv[4]);

  if(type == "q8") train_wrap<int8_t>(E, lr, nhidden, type);
  else if(type == "q16") train_wrap<int16_t>(E, lr, nhidden, type);
  else if (type == "q32") train_wrap<int32_t>(E, lr, nhidden, type);
  else if (type == "f32") train<float>(lr, nhidden, type);
  else if (type == "f64") train<double>(lr, nhidden, type);
  else {
    cout << "unknown data type " << type << "." << endl;
  }

  return 0;
}
