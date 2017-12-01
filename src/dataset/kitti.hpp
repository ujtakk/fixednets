#ifndef _KITTI_HPP_
#define _KITTI_HPP_

#include <unordered_map>

#include "network.hpp"

#if 0
// temp
struct BBoxMask
{
  Mat2D<float> det_boxes;
  Mat1D<float> det_probs;
  Mat1D<int> det_class;
};

// temp
struct DetConfig
{
  int H, W, B;
  int ANCHOR_PER_GRID;
  int CLASSES;
  int IMAGE_HEIGHT;
  int IMAGE_WIDTH;
  float PLOT_PROB_THRESH;
  float NMS_THRESH;
  float PROB_THRESH;
  int TOP_N_DETECTION;

  int BATCH_SIZE;
  bool LOAD_PRETRAINED_MODEL;

  bool DATA_AUGMENTATION;
  int DRIFT_X;
  int DRIFT_Y;
  bool EXCLUDE_HARD_EXAMPLES;

  Mat2D<float> ANCHOR_BOX;
  int ANCHORS;
};
#endif

class KITTI : Dataset
{
private:
  SqueezeDet<Q_TYPE> model;

  // temp
  std::string image_set = "val";
  // temp
  std::vector<std::string> _classes;
  DetConfig conf;

  int num_classes;

  std::string data_root_path = "../data/kitti";
  std::string image_path = data_root_path + "/training" + "/image_2";
  std::string label_path = data_root_path + "/training" + "/label_2";

  std::string eval_dir = "../data/kitti/eval";
  std::string eval_tool = "../utils/kitti-eval/cpp/evaluate_object";

  std::vector<std::string> image_idx;
  std::unordered_map<std::string, Mat2D<float>> rois;
  std::unordered_map<std::string, int> class_to_idx;

  Mat2D<float> set_anchors();
  std::vector<std::string> load_image_set_idx();
  std::unordered_map<std::string, Mat2D<float>> load_kitti_annotation();

  std::pair<std::vector<float>, std::vector<std::string>>
  evaluate_detections(std::string eval_dir, Mat4D<float> all_boxes);

  auto analyze_detections(std::string detection_file_dir, std::string det_error_file);

  auto do_detection_analysis_in_eval(std::string eval_dir);

public:
  KITTI();
  ~KITTI();

  BBoxMask predict(int sample);
  void test();
};

#include "kitti.cpp"
#endif
