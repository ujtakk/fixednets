#ifndef _KITTI_HPP_
#define _KITTI_HPP_

#include "network.hpp"

class KITTI : Dataset
{
private:
  SqueezeDet<Q_TYPE> model;

  const std::string class_map[] = {"car", "pedestrian", "cyclist"};
  const int CLASSES = 3;
  const int IMAGE_WIDTH  = 1248;
  const int IMAGE_HEIGHT = 384;

  mc.PLOT_PROB_THRESH      = 0.4;
  mc.NMS_THRESH            = 0.4;
  mc.PROB_THRESH           = 0.005;
  mc.TOP_N_DETECTION       = 64;

  mc.DATA_AUGMENTATION     = True;
  mc.DRIFT_X               = 150;
  mc.DRIFT_Y               = 100;
  mc.EXCLUDE_HARD_EXAMPLES = False;

  ANCHOR_BOX      = set_anchors(mc);
  const int ANCHORS         = len(mc.ANCHOR_BOX);
  const int ANCHOR_PER_GRID = 9;

  auto set_anchors();

public:
  KITTI();
  ~KITTI();

  void test();
};

#include "kitti.cpp"
#endif
