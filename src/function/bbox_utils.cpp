#ifdef _BBOX_UTILS_HPP_

#include "transform.hpp"

Mat1D<float> bbox_transform(float cx, float cy, float w, float h)
{
  auto out_box = zeros<float>(4);

  out_box[0] = cx - w / 2;
  out_box[1] = cy - h / 2;
  out_box[2] = cx + w / 2;
  out_box[3] = cy + h / 2;

  return out_box;
}

Mat1D<float> bbox_transform_inv(float xmin, float ymin, float xmax, float ymax)
{
  auto out_box = zeros<float>(4);

  auto width  = xmax - xmin + 1.0;
  auto height = ymax - ymin + 1.0;

  out_box[0]  = xmin + 0.5 * width ;
  out_box[1]  = ymin + 0.5 * height;
  out_box[2]  = width;
  out_box[3]  = height;

  return out_box;
}

Mat1D<float> batch_iou(Mat2D<float> boxes, Mat1D<float> box)
{
  // const int len = boxes.size();

  // auto left   = zeros<T>(len);
  // auto right  = zeros<T>(len);
  // auto bottom = zeros<T>(len);
  // auto top    = zeros<T>(len);
  //
  // for (int i = 0; i < len; ++i) {
  //   if (box_left < boxes_left[i])
  //     left[i] = boxes_left[i];
  //   else
  //     left[i] = box_left;
  //
  //   if (boxes_right[i] < box_right)
  //     right[i] = boxes_right[i];
  //   else
  //     right[i] = box_right;
  //
  //   if (box_top < boxes_top[i])
  //     top[i] = boxes_top[i];
  //   else
  //     top[i] = box_top;
  //
  //   if (boxes_bottom[i] < box_bottom)
  //     bottom[i] = boxes_bottom[i];
  //   else
  //     bottom[i] = box_bottom;
  // }

  auto boxes_t = transpose(boxes);

  auto half_width = ((float)0.5 * boxes_t[2]);
  auto half_height = ((float)0.5 * boxes_t[3]);
  auto boxes_right  = boxes_t[0] + half_width;
  auto boxes_left   = boxes_t[0] - half_width;
  auto boxes_bottom = boxes_t[1] + half_height;
  auto boxes_top    = boxes_t[1] - half_height;
  auto box_right    = box[0] + ((float)0.5 * box[2]);
  auto box_left     = box[0] - ((float)0.5 * box[2]);
  auto box_bottom   = box[1] + ((float)0.5 * box[3]);
  auto box_top      = box[1] - ((float)0.5 * box[3]);

  using std::numeric_limits;

  auto left   = clip(boxes_left,    box_left, max(boxes_left));
  auto right  = clip(boxes_right,   min(boxes_right), box_right);
  auto top    = clip(boxes_top,     box_top, max(boxes_top));
  auto bottom = clip(boxes_bottom,  min(boxes_bottom), box_bottom);

  auto left_right = clip(right-left, (float)0.0, numeric_limits<float>::max());
  auto top_bottom = clip(bottom-top, (float)0.0, numeric_limits<float>::max());

  auto intersection_area = left_right * top_bottom;
  auto base_area = boxes_t[2] * boxes_t[3];
  auto whole_area = base_area + box[2]*box[3];
  auto union_area = whole_area - intersection_area;

  return intersection_area / union_area;
}

Mat1D<bool> nms(Mat2D<float> boxes, Mat1D<float> probs, float thresh)
{
  const int len = probs.size();

  std::vector<int> order(len);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [probs](int i, int j) {
    return probs[i] > probs[j];
  });

  Mat1D<bool> keep(len, true);
  for (int i = 0; i < len-1; ++i) {
    // auto sub_boxes = boxes[order[i+1:]]
    Mat2D<float> sub_boxes(boxes.begin()+order[i+1], boxes.end());
    auto ovps = batch_iou(sub_boxes, boxes[order[i]]);
    for (int j = 0; j < (int)ovps.size(); ++j) {
      if (ovps[j] > thresh)
        keep[order[j+i+1]] = false;
    }
  }

  return keep;
}

#endif
