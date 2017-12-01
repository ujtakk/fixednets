#ifdef _SQUEEZE_DET_HPP_

template <typename T>
SqueezeDet<T>::SqueezeDet()
  : conv1  {64, 3, 3, 3, 2, 1}
  , pool1  {3, 3, 2, 1}
  , fire2  {16, 64, 64, 64}
  , fire3  {16, 64, 64, 128}
  , pool3  {3, 3, 2, 1}
  , fire4  {32, 128, 128, 128}
  , fire5  {32, 128, 128, 256}
  , pool5  {3, 3, 2, 1}
  , fire6  {48, 192, 192, 256}
  , fire7  {48, 192, 192, 384}
  , fire8  {64, 256, 256, 384}
  , fire9  {64, 256, 256, 512}
  , fire10 {96, 384, 384, 512}
  , fire11 {96, 384, 384, 768}
  , conv12 {72, 768, 3, 3, 1, 1}
{
}

template <typename T>
SqueezeDet<T>::~SqueezeDet()
{
}

template <typename T>
void SqueezeDet<T>::configure(DetConfig& conf)
{
  ANCHORS = conf.ANCHORS;
  ANCHOR_PER_GRID = conf.ANCHOR_PER_GRID;
  IMAGE_WIDTH = conf.IMAGE_WIDTH;
  IMAGE_HEIGHT = conf.IMAGE_HEIGHT;
  CLASSES = conf.CLASSES;
  ANCHOR_BOX = conf.ANCHOR_BOX;
}

template <typename T>
void SqueezeDet<T>::Load(std::string path)
{
  conv1.load(path+"/conv1");
  fire2.load(path+"/fire2");
  fire3.load(path+"/fire3");
  fire4.load(path+"/fire4");
  fire5.load(path+"/fire5");
  fire6.load(path+"/fire6");
  fire7.load(path+"/fire7");
  fire8.load(path+"/fire8");
  fire9.load(path+"/fire9");
  fire10.load(path+"/fire10");
  fire11.load(path+"/fire11");
  conv12.load(path+"/conv12");
}

template <typename T>
void SqueezeDet<T>::Save(std::string path)
{
}

template <typename T>
void SqueezeDet<T>::Forward(std::string data)
{
}

template <typename T>
void SqueezeDet<T>::Backward(BBoxMask label)
{
}

template <typename T>
void SqueezeDet<T>::Update()
{
}

template <typename T>
auto SqueezeDet<T>::merge_box_delta(Mat2D<T>& anchor, Mat2D<T>& delta)
{
  auto bbox_transform = [](auto cx, auto cy, auto w, auto h) {
    auto out_box = zeros<T>(4, cx.size());

    auto half_w = w / (float)2.0;
    auto half_h = h / (float)2.0;
    out_box[0] = cx - half_w;
    out_box[1] = cy - half_h;
    out_box[2] = cx + half_w;
    out_box[3] = cy + half_h;

    return out_box;
  };

  auto bbox_transform_inv = [](auto xmin, auto ymin, auto xmax, auto ymax) {
    auto width  = xmax - xmin;
    width = width + static_cast<T>(1.0);
    auto height = ymax - ymin;
    height = height + static_cast<T>(1.0);
    auto out_box = zeros<T>(4, xmin.size());

    auto half_w = static_cast<T>(0.5) * width;
    auto half_h = static_cast<T>(0.5) * height;
    out_box[0]  = xmin + half_w;
    out_box[1]  = ymin + half_h;
    out_box[2]  = width;
    out_box[3]  = height;

    return transpose(out_box);
  };

  _("transpose");
  auto delta_t = transpose(delta);
  auto delta_x = delta_t[0];
  auto delta_y = delta_t[1];
  auto delta_w = delta_t[2];
  auto delta_h = delta_t[3];

  _("transpose");
  auto anchor_t = transpose(anchor);
  auto anchor_x = anchor_t[0];
  auto anchor_y = anchor_t[1];
  auto anchor_w = anchor_t[2];
  auto anchor_h = anchor_t[3];

  const float EXP_THRESH = 1.0;
  auto center_x = delta_x * anchor_w;
  center_x = anchor_x + center_x;
  auto center_y = delta_y * anchor_h;
  center_y = anchor_y + center_y;

  _("transpose");
  auto width = safe_exp(delta_w, EXP_THRESH);
  width = anchor_w * width;
  auto height = safe_exp(delta_h, EXP_THRESH);
  height = anchor_h * height;

  auto boxes = bbox_transform(center_x, center_y, width, height);

  auto xmins = clip<T>(boxes[0], 0.0, IMAGE_WIDTH-1.0);
  auto ymins = clip<T>(boxes[1], 0.0, IMAGE_HEIGHT-1.0);
  auto xmaxs = clip<T>(boxes[2], 0.0, IMAGE_WIDTH-1.0);
  auto ymaxs = clip<T>(boxes[3], 0.0, IMAGE_HEIGHT-1.0);

  auto det_boxes = bbox_transform_inv(xmins, ymins, xmaxs, ymaxs);

  return det_boxes;
}

template <typename T>
Mat1D<float> SqueezeDet<T>::safe_exp(Mat1D<float>& x, float thresh)
{
  // TODO: implement
  return x;
}

template <typename T>
BBoxMask SqueezeDet<T>::interpret(Mat3D<T> preds)
{
  BBoxMask mask;

  const int num_class_probs = ANCHOR_PER_GRID * CLASSES;
  const int num_confidence_scores = ANCHOR_PER_GRID + num_class_probs;
  const int num_box_delta = preds.size();

  const int out_h = preds[0].size();
  const int out_w = preds[0][0].size();

  auto pred_class = zeros<float>(ANCHOR_PER_GRID * CLASSES, out_h, out_w);
  auto pred_confidence = zeros<float>(ANCHOR_PER_GRID, out_h, out_w);
  auto pred_box = zeros<float>(num_box_delta-num_confidence_scores, out_h, out_w);
  std::cout << num_box_delta-num_confidence_scores << std::endl;
  std::cout << ANCHORS << " " << ANCHOR_PER_GRID << " " << out_h << " " << out_w << std::endl;
  for (int i = 0; i < num_class_probs; ++i)
    pred_class[i] = preds[i];
  for (int i = num_class_probs; i < num_confidence_scores; ++i)
    pred_confidence[i-num_class_probs] = preds[i];
  for (int i = num_confidence_scores; i < num_box_delta; ++i)
    pred_box[i-num_confidence_scores] = preds[i];

  // auto pred_class_flat = zeros<T>(ANCHORS, CLASSES);
  // auto pred_class_probs = zeros<T>(ANCHORS, CLASSES);
  // f(reshape<T>(pred_class_flat, pred_class));
  // f(softmax(pred_class_probs, pred_class_flat));
  auto pred_class_flat = zeros<float>(ANCHORS*CLASSES);
  auto pred_class_ = zeros<float>(ANCHORS, CLASSES);
  auto pred_class_probs = zeros<float>(ANCHORS, CLASSES);
  f(flatten(pred_class_flat, pred_class));
  f(reshape(pred_class_, pred_class_flat));
  for (int i = 0; i < ANCHORS; ++i) {
    softmax(pred_class_probs[i], pred_class_[i]);
  }

  auto pred_confidence_flat = zeros<float>(ANCHORS);
  auto pred_confidence_scores = zeros<float>(ANCHORS);
  assert(ANCHORS == ANCHOR_PER_GRID*out_h*out_w);
  f(flatten(pred_confidence_flat, pred_confidence));
  f(sigmoid(pred_confidence_scores, pred_confidence_flat));

  auto pred_box_flat = zeros<float>(ANCHORS*4);
  // auto pred_box_ = zeros<float>(ANCHORS, 4);
  auto pred_box_delta = zeros<float>(ANCHORS, 4);
  f(flatten(pred_box_flat, pred_box));
  f(reshape(pred_box_delta, pred_box_flat));
  // f(reshape<float>(pred_box_delta, pred_box));

  f(mask.det_boxes = merge_box_delta(ANCHOR_BOX, pred_box_delta));

  auto probs = zeros<float>(ANCHORS, CLASSES);
  for (int i = 0; i < ANCHORS; ++i)
    // scalar * vector
    probs[i] = pred_confidence_scores[i] * pred_class_probs[i];

  mask.det_probs = zeros<float>(ANCHORS);
  mask.det_class = zeros<int>(ANCHORS);
  for (int i = 0; i < ANCHORS; ++i) {
    mask.det_probs[i] = max(probs[i]);
    mask.det_class[i] = argmax(probs[i]);
  }

  // return det_boxes, det_probs, det_class;
  return mask;
}

template <typename T>
BBoxMask SqueezeDet<T>::calc(std::string data)
{
  auto show = [](auto fmap) {
    // std::cout << fmap.size() << " "
    //           << fmap[0].size() << " "
    //           << fmap[0][0].size() << std::endl;
  };
  load_img(input, data);
  show(input);

  f(conv1.forward(fmap1, input));
  show(fmap1);
  f(pool1.forward(pmap1, fmap1));
  show(pmap1);
  f(fire2.forward(fmap2, pmap1));
  f(fire3.forward(fmap3, fmap2));
  f(pool3.forward(pmap3, fmap3));
  show(pmap3);
  f(fire4.forward(fmap4, pmap3));
  f(fire5.forward(fmap5, fmap4));
  f(pool5.forward(pmap5, fmap5));
  show(pmap5);
  f(fire6.forward(fmap6, pmap5));
  f(fire7.forward(fmap7, fmap6));
  f(fire8.forward(fmap8, fmap7));
  f(fire9.forward(fmap9, fmap8));
  f(fire10.forward(fmap10, fmap9));
  f(fire11.forward(fmap11, fmap10));
  show(fmap11);
  f(conv12.forward(fmap12, fmap11));
  show(fmap12);

  return interpret(fmap12);
}


#endif
