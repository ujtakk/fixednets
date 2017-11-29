#ifndef _KITTI_HPP_
#define _KITTI_HPP_

#include "network.hpp"

class KITTI : Dataset
{
private:
  SqueezeDet<Q_TYPE> model;

  const std::string class_map[] = {"car", "pedestrian", "cyclist"};
  DetConfig conf;

  auto set_anchors();

public:
  KITTI();
  ~KITTI();

  void test();
};

#include "kitti.cpp"
#endif
