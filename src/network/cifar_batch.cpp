#ifdef _DEEPER_CIFAR_HPP_

template <typename T>
VGG_CIFAR<T>::VGG_CIFAR()
  : conv1{N_F1, 3, FHEI, FWID}
  , conv2{N_F1, N_F1, FHEI, FWID}
  , pool1{PHEI,PWID}
  //, convpool1{N_F1, N_F1, FHEI, FWID, PHEI, PWID}
  , conv3{N_F2, N_F1, FHEI, FWID}
  , conv4{N_F2, N_F2, FHEI, FWID}
  , pool2{PHEI,PWID}
  //, convpool2{N_F2, N_F2, FHEI, FWID, PHEI, PWID}
  , conv5{N_F3, N_F2, FHEI, FWID}
  , conv6{N_F3, N_F3, FHEI, FWID}
  , conv7{N_F3, N_F3, FHEI, FWID}
  , conv8{N_F3, N_F3, FHEI, FWID}
  , pool3{PHEI,PWID}
  //, convpool3{N_F3, N_F3, FHEI, FWID, PHEI, PWID}
  , full9{N_H1, N_F3*pm3hei*pm3wid}
  , full10{N_H2, N_H1}
  , full11{LABEL, N_H2}

{
  input = zeros<T>(3, IMHEI, IMWID);

  fmap1 = zeros<T>(N_F1, IMHEI, IMWID);
  amap1 = zeros<T>(N_F1, IMHEI, IMWID);
  fmap2 = zeros<T>(N_F1, IMHEI, IMWID);
  pmap1 = zeros<T>(N_F1, pm1hei, pm1wid);
  amap2 = zeros<T>(N_F1, pm1hei, pm1wid);

  fmap3 = zeros<T>(N_F2, pm1hei, pm1wid);
  amap3 = zeros<T>(N_F2, pm1hei, pm1wid);
  fmap4 = zeros<T>(N_F2, pm1hei, pm1wid);
  pmap2 = zeros<T>(N_F2, pm2hei, pm2wid);
  amap4 = zeros<T>(N_F2, pm2hei, pm2wid);

  fmap5 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap5 = zeros<T>(N_F3, pm2hei, pm2wid);
  fmap6 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap6 = zeros<T>(N_F3, pm2hei, pm2wid);
  fmap7 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap7 = zeros<T>(N_F3, pm2hei, pm2wid);
  fmap8 = zeros<T>(N_F3, pm2hei, pm2wid);
  pmap3 = zeros<T>(N_F3, pm3hei, pm3wid);
  amap8 = zeros<T>(N_F3, pm3hei, pm3wid);

  amap8_flat = zeros<T>(N_F3*pm3hei*pm3wid);

  hunit1 = zeros<T>(N_H1);
  aunit1 = zeros<T>(N_H1);
  hunit2 = zeros<T>(N_H2);
  aunit2 = zeros<T>(N_H2);
  output = zeros<T>(LABEL);
}

template <typename T>
VGG_CIFAR<T>::~VGG_CIFAR()
{
}

template <typename T>
void VGG_CIFAR<T>::Load(char *path)
{
  conv1.load(path+"/wb_1");
  conv2.load(path+"/wb_2");
  //convpool1.load(path+"/wb_2");
  conv3.load(path+"/wb_3");
  conv4.load(path+"/wb_4");
  //convpool2.load(path+"/wb_4");
  conv5.load(path+"/wb_5");
  conv6.load(path+"/wb_6");
  conv7.load(path+"/wb_7");
  conv8.load(path+"/wb_8");
  //convpool3.load(path+"/wb_8");
  full9.load(path+"/wb_9");
  full10.load(path+"/wb_10");
  full11.load(path+"/wb_11");
}

template <typename T>
int VGG_CIFAR<T>::calc(char *data, int which, int amount)
{
  int number = -1;
  int temp = INT_MIN;

  load_image(data, input);

  conv1.forward(input, fmap1);
  relu1.forward(fmap1, amap1);
  conv2.forward(amap1, fmap2);
  pool1.forward(fmap2, pmap1);
  //convpool1.forward(fmap1, pmap1, which, amount);
  relu2.forward(pmap1, amap2);

  conv3.forward(amap2, fmap3);
  relu3.forward(fmap3, amap3);
  conv4.forward(amap3, fmap4);
  pool2.forward(fmap4, pmap2);
  //convpool2.forward(fmap2, pmap2, which, amount);
  relu4.forward(pmap2, amap4);

  conv5.forward(amap4, fmap5);
  relu5.forward(fmap5, amap5);
  conv6.forward(amap5, fmap6);
  relu6.forward(fmap6, amap6);
  conv7.forward(amap6, fmap7);
  relu7.forward(fmap7, amap7);
  conv8.forward(amap7, fmap8);
  pool3.forward(fmap8, pmap3);
  //convpool3.forward(fmap5, pmap3, which, amount);
  relu8.forward(pmap3, amap8);

  flatten(amap8, amap8_flat, N_F3, pm3hei, pm3wid);

  full9.forward(amap8_flat, hunit1);
  relu9.forward(hunit1, aunit1);

  full10.forward(aunit1, hunit2);
  relu10.forward(hunit2, aunit2);

  full11.forward(aunit2, output);

  for(int i=0;i<LABEL;i++) {
    if(temp < output[i]) {
      temp = output[i];
      number = i;
    }
  }

  return number;
}

#endif
