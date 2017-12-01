#ifndef _BBOX_UTILS_HPP_
#define _BBOX_UTILS_HPP_

#include "matrix.hpp"

Mat1D<float> bbox_transform(float cx, float cy, float w, float h);
Mat1D<float> bbox_transform_inv(float xmin, float ymin, float xmax, float ymax);
//
Mat1D<float> batch_iou(Mat2D<float> boxes, Mat1D<float> box);

Mat1D<bool> nms(Mat2D<float> boxes, Mat1D<float> probs, float thresh);

#include "bbox_utils.cpp"
#endif
