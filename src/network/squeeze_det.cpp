#ifdef _SQUEEZE_DET_HPP_

template <typename T>
SqueezeDet<T>::SqueezeDet(DetConfig& conf)
{
  ANCHORS = conf.H * conf.W * conf.B;
  ANCHOR_PER_GRID = conf.ANCHOR_PER_GRID;
  IMAGE_WIDTH = conf.IMAGE_WIDTH;
  IMAGE_HEIGHT = conf.IMAGE_HEIGHT;
  CLASSES = conf.CLASSES;
  ANCHOR_BOX = conf.ANCHOR_BOX;
}

template <typename T>
SqueezeDet<T>::~SqueezeDet()
{
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
void SqueezeDet<T>::Backward(int label)
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

    out_box[0] = cx - w / 2;
    out_box[1] = cy - h / 2;
    out_box[2] = cx + w / 2;
    out_box[3] = cy + h / 2;

    return out_box;
  };

  auto bbox_transform_inv = [](auto xmin, auto ymin, auto xmax, auto ymax) {
    auto width  = xmax - xmin + 1.0;
    auto height = ymax - ymin + 1.0;
    auto out_box = zeros<T>(4, xmin.size());

    out_box[0]  = xmin + 0.5 * width ;
    out_box[1]  = ymin + 0.5 * height;
    out_box[2]  = width;
    out_box[3]  = height;

    return transpose(out_box);
  };

  auto delta_t = transpose(delta);
  auto delta_x = delta_t[0];
  auto delta_y = delta_t[1];
  auto delta_w = delta_t[2];
  auto delta_h = delta_t[3];

  auto anchor_t = transpose(anchor);
  auto anchor_x = anchor_t[0];
  auto anchor_y = anchor_t[1];
  auto anchor_w = anchor_t[2];
  auto anchor_h = anchor_t[3];

  const float EXP_THRESH = 1.0;
  auto center_x = anchor_x + delta_x * anchor_w;
  auto center_y = anchor_y + delta_y * anchor_h;
  auto width = anchor_w * safe_exp(delta_w, EXP_THRESH);
  auto height = anchor_h * safe_exp(delta_h, EXP_THRESH);

  auto boxes = bbox_transform(center_x, center_y, width, height);

  auto xmins = clip(boxes[0], 0.0, IMAGE_WIDTH-1.0);
  auto ymins = clip(boxes[1], 0.0, IMAGE_HEIGHT-1.0);
  auto xmaxs = clip(boxes[2], 0.0, IMAGE_WIDTH-1.0);
  auto ymaxs = clip(boxes[3], 0.0, IMAGE_HEIGHT-1.0);

  auto det_boxes = bbox_transform_inv(xmins, ymins, xmaxs, ymaxs);

  return det_boxes;
}

template <typename T>
auto SqueezeDet<T>::set_anchors()
{
}

template <typename T>
Mat1D<float> SqueezeDet<T>::safe_exp(Mat1D<float>& x, float thresh)
{
  // TODO: implement
  return x;
}

template <typename T>
BBoxMask<T> SqueezeDet<T>::interpret(Mat3D<T> preds)
{
  BBoxMask<T> mask;

  const int num_class_probs = ANCHOR_PER_GRID * CLASSES;
  const int num_confidence_scores = ANCHOR_PER_GRID + num_class_probs;
  const int num_box_delta = preds.size();

  const int out_h = preds[0].size();
  const int out_w = preds[0][0].size();

  auto pred_class = zeros<T>(num_class_probs, out_h, out_w);
  auto pred_confidence = zeros<T>(num_confidence_scores, out_h, out_w);
  auto pred_box = zeros<T>(num_box_delta, out_h, out_w);
  for (int i = 0; i < num_class_probs; ++i)
    pred_class[i] = preds[i];
  for (int i = num_class_probs; i < num_confidence_scores; ++i)
    pred_confidence[i-num_class_probs] = preds[i];
  for (int i = num_confidence_scores; i < num_box_delta; ++i)
    pred_box[i-num_confidence_scores] = preds[i];

  auto pred_class_flat = zeros<T>(ANCHORS, CLASSES);
  auto pred_class_probs = zeros<T>(ANCHORS, CLASSES);
  reshape(pred_class_flat, pred_class);
  softmax(pred_class_probs, pred_class_flat);


  auto pred_confidence_flat = zeros<T>(ANCHORS);
  auto pred_confidence_scores = zeros<T>(ANCHORS);
  reshape(pred_confidence_flat, pred_confidence);
  sigmoid(pred_confidence_scores, pred_confidence_flat);

  auto pred_box_delta = zeros<T>(ANCHORS, 4);
  reshape(pred_box_delta, pred_box);

  mask.det_boxes = merge_box_delta(ANCHOR_BOX, pred_box_delta);

  auto probs = zeros<T>(ANCHORS, CLASSES);
  for (int i = 0; i < ANCHORS; ++i)
    // scalar * vector
    probs[i] = pred_confidence_scores[i] * pred_class_probs[i];

  mask.det_probs = zeros<T>(ANCHORS);
  mask.det_class = zeros<int>(ANCHORS);
  for (int i = 0; i < ANCHORS; ++i) {
    mask.det_probs[i] = max(probs[i]);
    mask.det_class[i] = argmax(probs[i]);
  }

  // return det_boxes, det_probs, det_class;
  return mask;
}

template <typename T>
BBoxMask<T> SqueezeDet<T>::calc(std::string data)
{
  conv1.forward(fmap1, input);
  pool1.forward(pmap1, fmap1);
  fire2.forward(fmap2, pmap1);
  fire3.forward(fmap3, fmap2);
  pool3.forward(pmap3, fmap3);
  fire4.forward(fmap4, pmap3);
  fire5.forward(fmap5, fmap4);
  pool5.forward(pmap5, fmap5);
  fire6.forward(fmap6, pmap5);
  fire7.forward(fmap7, fmap6);
  fire8.forward(fmap8, fmap7);
  fire9.forward(fmap9, fmap8);
  fire10.forward(fmap10, fmap9);
  fire11.forward(fmap11, fmap10);
  conv12.forward(fmap12, fmap11);

  BBoxMask<T> bboxes = interpret(fmap12);

  return bboxes;
}


#endif
