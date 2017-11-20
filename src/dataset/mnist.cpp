#ifdef _MNIST_HPP_

MNIST::MNIST()
{
#if defined _MLP
  // mlp.Load("/home/work/takau/binary_net/mlp");
  mlp.Load("/ldisk/takau/tf_tutorial/tf_mlp");
#elif defined _LENET
  model.Load("../data/mnist/lenet");
#endif
}

MNIST::~MNIST()
{
}

int MNIST::predict(int label, int sample)
{
  string filename = data(label, sample);
  int ans = model.calc(filename, 0, 0);

  std::cout << filename << ": answer is " << ans << std::endl;

  return ans;
}

void MNIST::test()
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
