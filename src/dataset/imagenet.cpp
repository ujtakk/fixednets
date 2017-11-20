#ifdef _MNIST_HPP_

ImageNet::ImageNet()
{
#if defined _ALEX
  model.Load("/home/work/takau/imagenet_data/alexbn");
#elif defined _VGG
  model.Load("/home/work/takau/imagenet_data/vgg12");
#endif
}

ImageNet::~ImageNet()
{
}

int ImageNet::predict(int label, int sample)
{
  string filename = data(label, sample);
  std::vector<int> ans(5);
  ans = model.calc(filename, 0, 0);

  printf(
    "%s: answer is %3d - %3d - %3d - %3d - %3d.\n",
    filename[i], ans[i][0], ans[i][1], ans[i][2], ans[i][3], ans[i][4]
  );

  return ans;
}

void ImageNet::test()
{
  std::vector<std::vector<int>> ans(SAMPLE, std::vector<int>(5));

  double total = 0.0;
  for (int label = 0; label < CLASS; label++) {
    for (int i=0; i < SAMPLE; i++)
      ans[i] = predict(label, i);

    int count = 0;
    for (int i=0; i < SAMPLE; i++) {
      /* Top 5 Answer */
           if(ans[i][0] == number) count++;
      else if(ans[i][1] == number) count++;
      else if(ans[i][2] == number) count++;
      else if(ans[i][3] == number) count++;
      else if(ans[i][4] == number) count++;
    }

    double prob = count / (double)SAMPLE;
    printf("%d: %.8e\n", label, prob);

    total += prob;
  }

  printf("result: %.8e\n", total/(double)CLASS);
}

#endif
