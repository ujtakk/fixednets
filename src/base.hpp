#ifndef _BASE_HPP_
#define _BASE_HPP_

#include <string>

template <typename T>
class Layer
{
private:

public:
  virtual void forward(T &input, T &output) = 0;
  virtual void backward(T &output, T &input) = 0;
};

template <typename T>
class ParamLayer : public Layer<T>
{
private:
public:
  virtual void load(std::string path) = 0;
  virtual void save(std::string path) = 0;

  virtual void forward(T &input, T &output) = 0;
  virtual void backward(T &output, T &input) = 0;

  virtual void update() = 0;
};

template <typename T, typename Rt>
class Network
{
private:
public:
  virtual void Load(std::string path) = 0;
  virtual void Save(std::string path) = 0;

  virtual void Forward(std::string data) = 0;
  virtual void Backward(Rt label) = 0;

  virtual void Update() = 0;

  virtual Rt calc(std::string data, int which, int amount) = 0;
};

class Dataset
{
private:
public:
  virtual int predict(int label, int sample) = 0;
  virtual void test() = 0;
};

#endif
