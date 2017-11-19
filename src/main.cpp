#define _LENET
#define _EAGER
#define CLASS  10
#define SAMPLE 100

#include <string>

#include "network.hpp"

using std::string;
using std::to_string;

#if defined _MLP
  MLP<Q_TYPE>     mlp;
#elif defined _LENET
  LeNet<Q_TYPE>   lenet;
#elif defined _CIFAR
  CIFAR<Q_TYPE>   cifar;
#elif defined _ALEX
  AlexNet<Q_TYPE> alex;
#elif defined _VGG
  VGGNet<Q_TYPE>  vgg;
#endif

void classify(int number, int sample)
{
  string filename;

#if defined _ALEX || defined _VGG
  std::vector<int> ans(5);
#else
  int ans;
#endif

  auto data = [](string base, int number, int sample) {
    return base + to_string(number) + "/data" + to_string(sample) + ".txt";
  };

#if defined _MLP
    filename = "/home/work/takau/2.mlearn/mnist_data/input/"
             + to_string(number)+"/data"+to_string(sample)+".txt";
    ans = mlp.calc(filename);
#elif defined _LENET
    filename = data("../data/mnist/input/", number, sample);
    ans = lenet.calc(filename, 0, 0);
#elif defined _CIFAR
    filename = "/home/work/takau/2.mlearn/cifar10_data/pro_input/"
             + to_string(number)+"/data"+to_string(sample+1)+".txt";
    ans = cifar.calc(filename, 0, 0);
#elif defined _ALEX
    filename = "/home/work/takau/2.mlearn/imagenet_data/input/"
             + to_string(number)+"/data"+to_string(sample)+".txt";
    ans = alex.calc(filename);
#elif defined _VGG
    filename = "/home/work/takau/2.mlearn/imagenet_data/input224/"
             + to_string(number)+"/data"+to_string(sample)+".txt";
    ans = vgg.calc(filename);
#endif

    std::cout << filename << ": answer is " << ans << std::endl;
}

double accuracy(int number, int which, int amount)
{
  std::vector<string> filename(SAMPLE);
#if defined _ALEX || defined _VGG
  std::vector<std::vector<int>> ans(SAMPLE, std::vector<int>(5));
#else
  int ans[SAMPLE];
#endif
  int count = 0;
  double pnum;

  auto data = [](string base, int number, int sample) {
    return base + to_string(number) + "/data" + to_string(sample) + ".txt";
  };

  for (int i=0; i < SAMPLE; i++) {
#if defined _MLP
    filename[i] = "/home/work/takau/2.mlearn/mnist_data/input/"
                + to_string(number)+"/data"+to_string(i)+".txt";
    ans[i] = mlp.calc(filename[i]);
#elif defined _LENET
    filename[i] = data("../data/mnist/input/", number, i);
    ans[i] = lenet.calc(filename[i], 0, 0);
#elif defined _CIFAR
    filename[i] = "/home/work/takau/2.mlearn/cifar10_data/pro_input/"
                + to_string(number)+"/data"+to_string(i+1)+".txt";
    ans[i] = cifar.calc(filename[i], 0, 0);
#elif defined _ALEX
    filename[i] = "/home/work/takau/2.mlearn/imagenet_data/input/"
                + to_string(number)+"/data"+to_string(i)+".txt";
    ans[i] = alex.calc(filename[i]);
#elif defined _VGG
    filename[i] = "/home/work/takau/2.mlearn/imagenet_data/input224/"
                + to_string(number)+"/data"+to_string(i)+".txt";
    ans[i] = vgg.calc(filename[i]);
#endif

#if defined _ALEX || defined _VGG
    printf(
      "%s: answer is %3d - %3d - %3d - %3d - %3d.\n",
      filename[i], ans[i][0], ans[i][1], ans[i][2], ans[i][3], ans[i][4]
    );
  }

  /* Top 5 Answer */
  for (int i=0; i < SAMPLE; i++) {
         if(ans[i][0] == number) count++;
    else if(ans[i][1] == number) count++;
    else if(ans[i][2] == number) count++;
    else if(ans[i][3] == number) count++;
    else if(ans[i][4] == number) count++;
  }
#else
    std::cout << filename[i] << ": answer is " << ans[i] << std::endl;
  }

  for (int i=0; i < SAMPLE; i++)
    if(ans[i] == number) count++;
#endif

  pnum = count / (double)SAMPLE;

  return pnum;
}

void graph()
{
  double pnum = 0.0;
  double total = 0.0;

  // string output;
  // string filename;
  // FILE *fp;
  // FILE *gp;

  for (int which=0; which < 1; which++) {
    // filename = "./eval_cifar/Type"+which+".dat";
    // gp = fopen(filename, "w");
    for (int amount=0; amount < 1; amount++) {
      // output = "./eval_cifar/Type"+which+"/Type"+which+"x"+amount+".dat";
      // fp = fopen(output, "w");

      for (int number=0; number < CLASS; number++) {
        pnum = accuracy(number, which, amount);
        total += pnum;
        // fprintf(fp, "%d %.8e\n", number, pnum);
      }

      printf("%d %.8e\n", amount, total/(double)CLASS);
      // fprintf(gp, "%d %.8e\n", amount, total/(double)CLASS);

      total = 0.0;
      // fclose(fp);
    }
    // fclose(gp);
  }
}

int main(int argc, char **argv)
{
#if defined _MLP
  // mlp.Load("/home/work/takau/binary_net/mlp");
  mlp.Load("/ldisk/takau/tf_tutorial/tf_mlp");
#elif defined _LENET
  lenet.Load("../data/mnist/lenet");
  // lenet.Load("../../models_chainer/lenetbn");
#elif defined _CIFAR
  cifar.Load("/home/work/takau/cifar10_data/src/cifar10_data/test");
#elif defined _ALEX
  alex.Load("/home/work/takau/imagenet_data/alexbn");
#elif defined _VGG
  vgg.Load("/home/work/takau/imagenet_data/vgg12");
#endif

  if (argv[1] && argv[2]) {
    int number = 0;
    int sample = 0;

    number = std::stoi(argv[1]);
    sample = std::stoi(argv[2]);
    std::cout << "number: " << number << ", "
              << "sample: " << sample << std::endl;
    try {
      classify(number, sample);
    }
    catch (const string e) {
      std::cerr << e << std::endl;
    }
  }
  else {
    try {
      graph();
    } catch (const string e) {
      std::cerr << e << std::endl;
    }
  }

  return 0;
}

