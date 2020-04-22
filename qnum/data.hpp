#pragma once

#include "common.hpp"
#include <execution>
#include <algorithm>
#include <random>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr auto c_mnist_dir = "F:/data/mnist";
constexpr auto c_mnist_train_img_file = "train-images.idx3-ubyte";
constexpr auto c_mnist_train_label_file = "train-labels.idx1-ubyte";
constexpr auto c_mnist_train_size = 60000;
constexpr auto c_mnist_imgh = 28;
constexpr auto c_mnist_imgw = 28;

typedef uint8_t mnist_img_t[c_mnist_imgh * c_mnist_imgw];
typedef uint8_t mnist_label_t;

template<typename T>
struct mnist_enumerable {
  const VectorXvar<T> *pimgs, *plabels;
  int len;
  vector<int> idx;
  mnist_enumerable(const VectorXvar<T>* imgs, const VectorXvar<T>* labels, const int len):
    pimgs(imgs), plabels(labels), len(len), idx(len)
  { 
    for (int i = 0; i < len; ++i) {
      idx[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
  }
  vector<int>::iterator begin() { return idx.begin(); }
  vector<int>::iterator end() { return idx.end(); }
};

template<typename T>
struct mnist_train_t {
  VectorXvar<T> imgs[c_mnist_train_size];
  VectorXvar<T> labels[c_mnist_train_size];
  mnist_enumerable<T> shuffle() const {
    return mnist_enumerable(imgs, labels, c_mnist_train_size);
  }
};

template<typename T>
const mnist_train_t<T>* load_train() {
  char buf[256];
  sprintf(buf, "%s/%s", c_mnist_dir, c_mnist_train_img_file);
  FILE* img_fp = fopen(buf, "rb");
  sprintf(buf, "%s/%s", c_mnist_dir, c_mnist_train_label_file);
  FILE* label_fp = fopen(buf, "rb");

  if (!img_fp || !label_fp) {
    printf("cannot read data.\n");
    exit(-1);
  }

  auto p = new mnist_train_t<T>();

  fread(buf, 16, 1, img_fp);
  fread(buf, 8, 1, label_fp);

  vector<mnist_img_t> imgs(c_mnist_train_size);
  vector<mnist_label_t> labels(c_mnist_train_size);

  fread(imgs.data(), sizeof(mnist_img_t), c_mnist_train_size, img_fp);
  fread(labels.data(), sizeof(mnist_label_t), c_mnist_train_size, label_fp);

  fclose(img_fp);
  fclose(label_fp);

  vector<int> idx(c_mnist_train_size);
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto i) {
    p->imgs[i] = VectorXvar<T>::Zero(c_mnist_imgh * c_mnist_imgw);
    p->labels[i] = VectorXvar<T>::Zero(10);
    for (int j = 0; j < 10; ++j) {
      if (labels[i] == j) {
        p->labels[i][j] = T(1.0);
      }
      else {
        p->labels[i][j] = T(0.0);
      }
    }
    for (int j = 0; j < c_mnist_imgh * c_mnist_imgw; ++j) {
      p->imgs[i][j] = T(imgs[i][j]);
    }
  });

  return p;
}
