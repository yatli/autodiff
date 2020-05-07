#include "common.hpp"
using namespace std;
using namespace Eigen;

int main() {
  VectorXvar v;
  MatrixXvar m1;
  v.resize(3, 1);
  m1.resize(3, 3);
  //MatrixXvar m2 = MatrixXvar::Random(3, 3);

  auto p1 = m1 * v;
  //auto prod = m2 * p1;

  //auto aggregate = prod.sum();
  auto aggregate = p1.sum();

  for(int i=0;i<3;++i) {
    for(int j=0;j<3;++j) {
      m1(i,j).seed();
      //m2(i,j).seed();
    }
  }

  aggregate.expr->print(0);
  aggregate.rewrite();
  cout << "-----------------------------" << endl;
  aggregate.expr->print(0);
  aggregate.expr->propagate(1.0);

  return 0;
}
