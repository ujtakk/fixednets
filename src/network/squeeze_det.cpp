#ifdef _SQUEEZE_DET_HPP_

template <typename T>
SqueezeDet<T>::SqueezeDet(DetConf config)
{
}

template <typename T>
SqueezeDet<T>::~SqueezeDet()
{
}

template <typename T>
void Load(std::string path)
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
void Save(std::string path)
{
}

template <typename T>
void Forward(std::string data)
{
}

template <typename T>
void Backward(int label)
{
}

template <typename T>
void Update()
{
}

template <typename T>
void interpret(Mat3D<T> preds)
{
  const int ANCHOR_PER_GRID;
  const int CLASSES;
  auto num_class_probs = ANCHOR_PER_GRID * CLASSES;

  const int ANCHORS;
  auto pred_class_probs =
    reshape(softmax(reshape(preds[:, :, :num_class_probs],
            [-1, CLASSES])), [ANCHORS, CLASSES]);

  auto num_confidence_scores = ANCHOR_PER_GRID + num_class_probs;
  auto pred_conf =
    sigmoid(reshape(pred[:, :, num_class_probs:num_confidence_scores],
            [ANCHORS]);

  auto pred_box_delta =
    reshape(pred[:, :, num_confidence_scores:], [ANCHORS, 4]);

  auto input_mask;
  auto num_objects = reduce_sum(input_mask);

  delta_x, delta_y, delta_w, delta_h = tf.unstack(
      self.pred_box_delta, axis=2)

  auto ANCHOR_BOX;
  auto anchor_x = ANCHOR_BOX[:, 0];
  auto anchor_y = ANCHOR_BOX[:, 1];
  auto anchor_w = ANCHOR_BOX[:, 2];
  auto anchor_h = ANCHOR_BOX[:, 3];

  auto box_center_x = anchor_x + delta_x * anchor_w;
  auto box_center_y = anchor_y + delta_y * anchor_h;
  auto box_width = anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH);
  auto box_height = anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH);

  xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
      [box_center_x, box_center_y, box_width, box_height])

  xmins = clip(xmins, 0.0, IMAGE_WIDTH-1.0);
  ymins = clip(xmins, 0.0, IMAGE_HEIGHT-1.0);
  xmaxs = clip(xmins, 0.0, IMAGE_WIDTH-1.0);
  ymaxs = clip(xmins, 0.0, IMAGE_HEIGHT-1.0);

  self.det_boxes = tf.transpose(
      tf.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
      (1, 2, 0), name='bbox'
  )

  probs = tf.multiply(
      self.pred_class_probs,
      tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
      name='final_class_prob'
  )

  self.det_probs = tf.reduce_max(probs, 2, name='score')
  self.det_class = tf.argmax(probs, 2, name='class_idx')

  return det_boxes, det_probs, det_class;
}

template <typename T>
BBoxMask<T> calc(std::string data, int which, int amount)
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
