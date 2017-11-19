#ifdef _LENET_HPP_

template <typename T>
LeNet<T>::LeNet()
  :  conv1{N_F1,    1, FHEI, FWID, 1, 0}
  ,  conv2{N_F2, N_F1, FHEI, FWID, 1, 0}
  // :  conv1{N_F1,    1, FHEI, FWID}
  // ,  conv2{N_F2, N_F1, FHEI, FWID}
  ,  pool1{PHEI, PWID}
  ,  pool2{PHEI, PWID}
  ,  full3{N_HL, N_F2*pm2hei*pm2wid}
  ,  full4{LABEL, N_HL}
{
  input = zeros<T>(1, IMHEI, IMWID);
  fmap1 = zeros<T>(N_F1, IMHEI-FHEI+1, IMWID-FWID+1);
  amap1 = zeros<T>(N_F1, IMHEI-FHEI+1, IMWID-FWID+1);
  pmap1 = zeros<T>(N_F1, pm1hei, pm1wid);
  fmap2 = zeros<T>(N_F2, pm1hei-FHEI+1, pm1wid-FWID+1);
  amap2 = zeros<T>(N_F2, pm1hei-FHEI+1, pm1wid-FWID+1);
  pmap2 = zeros<T>(N_F2, pm2hei, pm2wid);
  pmap2_flat = zeros<T>(N_F2*pm2hei*pm2wid);
  hunit = zeros<T>(N_HL);
  aunit = zeros<T>(N_HL);
  output = zeros<T>(LABEL);
}

template <typename T>
LeNet<T>::~LeNet()
{
}

template <typename T>
void LeNet<T>::Load(string path)
{
#if 1
  conv1.load(path+"/conv1");
  conv2.load(path+"/conv2");
  full3.load(path+"/full3");
  full4.load(path+"/full4");
#else
  conv1.load(path+"/wb_1");
  conv2.load(path+"/wb_2");
  full3.load(path+"/wb_3");
  full4.load(path+"/wb_4");
#endif
}

template <typename T>
void LeNet<T>::Save(string path)
{
  conv1.save(path+"/conv0");
  conv2.save(path+"/conv1");
  full3.save(path+"/full2");
  full4.save(path+"/full3");
}

template <typename T>
void LeNet<T>::Forward(string data)
{
  load_image(data, input);

  conv1.forward(input, fmap1);
  pool1.forward(fmap1, pmap1);
  relu1.forward(pmap1, amap1);

  conv2.forward(pmap1, fmap2);
  pool2.forward(fmap2, pmap2);
  relu2.forward(pmap2, amap2);

  flatten(pmap2, pmap2_flat, N_F2, pm2hei, pm2wid);

  full3.forward(pmap2_flat, hunit);
  relu3.forward(hunit, aunit);

  full4.forward(hunit, output);
  //output4.forward(ioutput);
}

template <typename T>
void LeNet<T>::Backward(int label)
{
  //output4.backward(output);
  full4.backward(output, hunit);

  relu3.backward(aunit, hunit);
  full3.backward(hunit, pmap2_flat);

  reshape(pmap2_flat, pmap2, N_F2, pm2hei, pm2wid);

  relu2.backward(amap2, pmap2);
  pool2.backward(pmap2, fmap2);
  conv2.backward(fmap2, pmap1);

  relu1.backward(amap1, pmap1);
  pool1.backward(pmap1, fmap1);
  conv1.backward(fmap1, input);
}

template <typename T>
void LeNet<T>::Update()
{
  conv1.update();
  conv2.update();
  full3.update();
  full4.update();
}

template <typename T>
int LeNet<T>::calc(string data, int which, int amount)
{
  load_image(data, input);

  conv1.forward(input, fmap1);
  relu1.forward(fmap1, amap1);
  pool1.forward(amap1, pmap1);

  conv2.forward(pmap1, fmap2);
  relu2.forward(fmap2, amap2);
  pool2.forward(amap2, pmap2);

  flatten(pmap2, pmap2_flat);

  full3.forward(pmap2_flat, hunit);
  relu3.forward(hunit, aunit);

  full4.forward(aunit, output);

  int number = -1;
  T temp = std::numeric_limits<T>::min();
  for (int i=0; i<LABEL; i++) {
    if (temp < output[i]) {
      temp = output[i];
      number = i;
    }
  }

  return number;
}

#endif
