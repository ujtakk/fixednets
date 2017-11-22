#ifdef _CIFAR10_HPP_

CIFAR10::CIFAR10()
{
  model.Load("/home/work/takau/cifar10_data/src/cifar10_data/test");
}

CIFAR10::~CIFAR10()
{
}

int CIFAR10::predict(int label, int sample)
{
  std::string filename = data(label, sample);
  int ans = model.calc(filename, 0, 0);

  std::cout << filename << ": answer is " << ans << std::endl;

  return ans;
}

void CIFAR10::test()
{
  int ans[SAMPLE];

  double total = 0.0;
  for (int label = 0; label < CLASS; label++) {
    for (int i=0; i < SAMPLE; i++)
      ans[i] = predict(label, i);

    int count = 0;
    for (int i=0; i < SAMPLE; i++)
      if(ans[i] == label) count++;

    double prob = count / (double)SAMPLE;
    printf("%d: %.8e\n", label, prob);

    total += prob;
  }

  printf("result: %.8e\n", total/(double)CLASS);
}

#endif
