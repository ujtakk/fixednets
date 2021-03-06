#ifdef _SQUEEZE_DET_HPP_

template <typename T>
SqueezeDet<T>::SqueezeDet(bool quantized)
  : conv1  {64, 3, 3, 3, 2, 1, quantized}
  , pool1  {3, 3, 2, 1}
  , fire2  {16, 64, 64, 64, quantized}
  , fire3  {16, 64, 64, 128, quantized}
  , pool3  {3, 3, 2, 1}
  , fire4  {32, 128, 128, 128, quantized}
  , fire5  {32, 128, 128, 256, quantized}
  , pool5  {3, 3, 2, 1}
  , fire6  {48, 192, 192, 256, quantized}
  , fire7  {48, 192, 192, 384, quantized}
  , fire8  {64, 256, 256, 384, quantized}
  , fire9  {64, 256, 256, 512, quantized}
  , fire10 {96, 384, 384, 512, quantized}
  , fire11 {96, 384, 384, 768, quantized}
  , conv12 {72, 768, 3, 3, 1, 1, quantized}
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
  TOP_N_DETECTION = conf.TOP_N_DETECTION;
  NMS_THRESH = conf.NMS_THRESH;
  PROB_THRESH = conf.PROB_THRESH;

  input = zeros<T>(3, IMAGE_HEIGHT, IMAGE_WIDTH);
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
auto SqueezeDet<T>::merge_box_delta(Mat2D<float>& anchor, Mat2D<float>& delta)
{
  auto bbox_transform =
  [](Mat1D<float> cx, Mat1D<float> cy, Mat1D<float> w, Mat1D<float> h) {
    Mat2D<float> out_box = zeros<float>(4, cx.size());

    Mat1D<float> half_w = w / (float)2.0;
    Mat1D<float> half_h = h / (float)2.0;
    out_box[0] = cx - half_w;
    out_box[1] = cy - half_h;
    out_box[2] = cx + half_w;
    out_box[3] = cy + half_h;

    return out_box;
  };

  auto bbox_transform_inv =
  [](Mat1D<float> xmin, Mat1D<float> ymin, Mat1D<float> xmax, Mat1D<float> ymax) {
    Mat2D<float> out_box = zeros<float>(4, xmin.size());

    Mat1D<float> width  = xmax - xmin;
    width = width + static_cast<float>(1.0);

    Mat1D<float> height = ymax - ymin;
    height = height + static_cast<float>(1.0);

    Mat1D<float> half_w = static_cast<float>(0.5) * width;
    Mat1D<float> half_h = static_cast<float>(0.5) * height;
    out_box[0]  = xmin + half_w;
    out_box[1]  = ymin + half_h;
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
  auto center_x = delta_x * anchor_w;
  center_x = anchor_x + center_x;
  auto center_y = delta_y * anchor_h;
  center_y = anchor_y + center_y;

  auto width = safe_exp(delta_w, EXP_THRESH);
  width = anchor_w * width;

  auto height = safe_exp(delta_h, EXP_THRESH);
  height = anchor_h * height;

  auto boxes = bbox_transform(center_x, center_y, width, height);

  auto xmins = clip<float>(boxes[0], 0.0, IMAGE_WIDTH-1.0);
  auto ymins = clip<float>(boxes[1], 0.0, IMAGE_HEIGHT-1.0);
  auto xmaxs = clip<float>(boxes[2], 0.0, IMAGE_WIDTH-1.0);
  auto ymaxs = clip<float>(boxes[3], 0.0, IMAGE_HEIGHT-1.0);

  auto det_boxes = bbox_transform_inv(xmins, ymins, xmaxs, ymaxs);

  return det_boxes;
}

template <typename T>
Mat1D<float> SqueezeDet<T>::safe_exp(Mat1D<float>& w, float thresh)
{
  const int len = w.size();

  Mat1D<float> out(len);
  for (int i = 0; i < len; ++i) {
    auto x = w[i];
    auto y = 0.0;

    if (x > thresh)
      y = exp(thresh) * (x - thresh + 1.0);
    else
      y = exp(x);

    out[i] = y;
  }

  return out;
}

template <typename T>
BBoxMask SqueezeDet<T>::interpret(Mat3D<T> preds)
{
  BBoxMask mask;
  Mat3D<float> preds_float = float_of_T(preds);
  // save_txt("now_preds.txt", preds_float);
  // exit(0);

  const int num_class_probs = ANCHOR_PER_GRID * CLASSES;
  const int num_confidence_scores = ANCHOR_PER_GRID + num_class_probs;
  const int num_box_delta = preds.size();

  const int out_h = preds[0].size();
  const int out_w = preds[0][0].size();

  auto pred_class = zeros<float>(out_h, out_w, ANCHOR_PER_GRID * CLASSES);
  auto pred_confidence = zeros<float>(out_h, out_w, ANCHOR_PER_GRID);
  auto pred_box = zeros<float>(out_h, out_w,
                               num_box_delta-num_confidence_scores);
  for (int j = 0; j < out_h; ++j) {
    for (int k = 0; k < out_w; ++k) {
      // convert to tensorflow encoding
      for (int i = 0; i < num_class_probs; ++i)
        pred_class[j][k][i] = preds_float[i][j][k];

      for (int i = num_class_probs; i < num_confidence_scores; ++i)
        pred_confidence[j][k][i-num_class_probs] = preds_float[i][j][k];

      for (int i = num_confidence_scores; i < num_box_delta; ++i)
        pred_box[j][k][i-num_confidence_scores] = preds_float[i][j][k];
    }
  }

  auto pred_class_flat = zeros<float>(ANCHORS*CLASSES);
  auto pred_class_ = zeros<float>(ANCHORS, CLASSES);
  auto pred_class_probs = zeros<float>(ANCHORS, CLASSES);
  flatten(pred_class_flat, pred_class);
  reshape(pred_class_, pred_class_flat);
  for (int i = 0; i < ANCHORS; ++i) {
    softmax(pred_class_probs[i], pred_class_[i]);
  }

  auto pred_confidence_flat = zeros<float>(ANCHORS);
  auto pred_confidence_scores = zeros<float>(ANCHORS);
  flatten(pred_confidence_flat, pred_confidence);
  sigmoid(pred_confidence_scores, pred_confidence_flat);

  auto pred_box_flat = zeros<float>(ANCHORS*4);
  auto pred_box_delta = zeros<float>(ANCHORS, 4);
  flatten(pred_box_flat, pred_box);
  reshape(pred_box_delta, pred_box_flat);

  mask.det_boxes = merge_box_delta(ANCHOR_BOX, pred_box_delta);

  Mat2D<float> probs = zeros<float>(ANCHORS, CLASSES);
  for (int i = 0; i < ANCHORS; ++i)
    // scalar * vector
    probs[i] = pred_confidence_scores[i] * pred_class_probs[i];

  mask.det_probs = zeros<float>(ANCHORS);
  mask.det_class = zeros<int>(ANCHORS);
  for (int i = 0; i < ANCHORS; ++i) {
    mask.det_probs[i] = max(probs[i]);
    mask.det_class[i] = argmax(probs[i]);
  }

  return mask;
}

template <typename T>
BBoxMask SqueezeDet<T>::filter(BBoxMask mask)
{
  auto boxes = mask.det_boxes;
  auto probs = mask.det_probs;
  auto _clas = mask.det_class;

  std::vector<int> whole(probs.size());
  std::iota(whole.begin(), whole.end(), 0);
  std::vector<int> order;

  if (0 < TOP_N_DETECTION && TOP_N_DETECTION < (int)probs.size()) {
    std::sort(whole.begin(), whole.end(), [&](int i, int j) {
      return probs[i] > probs[j];
    });
    order.assign(whole.begin(), whole.begin()+TOP_N_DETECTION);
  }
  else {
    std::copy_if(whole.begin(), whole.end(), order.begin(), [&](int i) {
      return probs[i] > PROB_THRESH;
    });
  }

  Mat2D<float>  new_boxes;
  Mat1D<float>  new_probs;
  Mat1D<int>    new_class;
  for (int i : order) {
    new_boxes.emplace_back(boxes[i]);
    new_probs.emplace_back(probs[i]);
    new_class.emplace_back(_clas[i]);
  }

  Mat2D<float>  final_boxes;
  Mat1D<float>  final_probs;
  Mat1D<int>    final_class;

  for (int c = 0; c < CLASSES; ++c) {
    Mat2D<float>  cand_boxes;
    Mat1D<float>  cand_probs;
    for (int i = 0; i < (int)new_class.size(); ++i) {
      if (new_class[i] == c) {
        cand_boxes.emplace_back(new_boxes[i]);
        cand_probs.emplace_back(new_probs[i]);
      }
    }

    auto keep = nms(cand_boxes, cand_probs, NMS_THRESH);
    for (int i = 0; i < (int)keep.size(); ++i) {
      if (keep[i]) {
        final_boxes.emplace_back(cand_boxes[i]);
        final_probs.emplace_back(cand_probs[i]);
        final_class.emplace_back(c);
      }
    }
  }

  BBoxMask filtered_mask;
  filtered_mask.det_boxes = final_boxes;
  filtered_mask.det_probs = final_probs;
  filtered_mask.det_class = final_class;

  return filtered_mask;
}

#define show(fmap) \
  printf("%s:\t%4d %4d %4d\n", \
         #fmap, (int)fmap.size(), (int)fmap[0][0].size(), (int)fmap[0].size());
template <typename T>
BBoxMask SqueezeDet<T>::calc(std::string data)
{
  auto scales = load_img(input, data);
  // Mat3D<float> fim = float_of_T(input);
  // save_txt("now_image.txt", fim);

  _DO_(conv1.forward(fmap1, input));
  _DO_(pool1.forward(pmap1, fmap1));
  _DO_(fire2.forward(fmap2, pmap1));
  _DO_(fire3.forward(fmap3, fmap2));
  _DO_(pool3.forward(pmap3, fmap3));
  _DO_(fire4.forward(fmap4, pmap3));
  _DO_(fire5.forward(fmap5, fmap4));
  _DO_(pool5.forward(pmap5, fmap5));
  _DO_(fire6.forward(fmap6, pmap5));
  _DO_(fire7.forward(fmap7, fmap6));
  _DO_(fire8.forward(fmap8, fmap7));
  _DO_(fire9.forward(fmap9, fmap8));
  _DO_(fire10.forward(fmap10, fmap9));
  _DO_(fire11.forward(fmap11, fmap10));
  _DO_(conv12.forward(fmap12, fmap11));

  // Mat3D<float> ff = float_of_T(pmap1);
  // save_txt("now_pool1.txt", ff);
  // _("done");
  // exit(0);

  // show(input);
  // show(fmap1);
  // show(pmap1);
  // show(fmap2);
  // show(fmap3);
  // show(pmap3);
  // show(fmap4);
  // show(fmap5);
  // show(pmap5);
  // show(fmap6);
  // show(fmap7);
  // show(fmap8);
  // show(fmap9);
  // show(fmap10);
  // show(fmap11);
  // show(fmap12);
  // exit(0);
  BBoxMask mask;

  _DO_(mask = interpret(fmap12));
  mask.scales = scales;

  return mask;
}

#endif
