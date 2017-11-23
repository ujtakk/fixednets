#ifdef _MNIST_HPP_

MNIST::MNIST()
{
#if defined _MLP
  model.Load("../data/mnist/mlp");
#elif defined _LENET
  model.Load("../data/mnist/lenet");
#endif
}

MNIST::~MNIST()
{
}

#include <random>
#include <algorithm>
#include <map>
// TODO: introduce batch training
void MNIST::train()
{
  // const float alpha = 0.001;
  std::mt19937 mt(42);

  std::vector<std::pair<int, int>> batch(CLASS*SAMPLE);
  for (int i = 0; i < CLASS; ++i) {
    for (int j = 0; j < SAMPLE; ++j) {
      batch[SAMPLE*i+j] = std::make_pair(i, j);
    }
  }

  for (int i = 0; i < 10; ++i) {
    std::shuffle(batch.begin(), batch.end(), mt);
    for (auto target : batch) {
      const int label = target.first;
      const int sample = target.second;
      const std::string filename = data(label, sample);

      // printf("%d: ", label);
      model.Forward(filename);
      model.Backward(label);
      model.Update();
    }
    test();
  }
}

int MNIST::predict(int label, int sample)
{
  std::string filename = data(label, sample);
  int ans = model.calc(filename, 0, 0);

  // std::cout << filename << ": answer is " << ans << std::endl;

  return ans;
}

void MNIST::test()
{
  int ans[SAMPLE];

  double total = 0.0;
  for (int label = 0; label < CLASS; label++) {
    for (int sample = 0; sample < SAMPLE; sample++)
      ans[sample] = predict(label, sample);

    int count = 0;
    for (int sample=0; sample < SAMPLE; sample++)
      if(ans[sample] == label) count++;

    double prob = count / (double)SAMPLE;
    printf("%d: %.8e\n", label, prob);

    total += prob;
  }

  printf("result: %.3f\n", total/(double)CLASS);
}

#endif
