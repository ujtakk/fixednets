#ifdef _KITTI_HPP_

KITTI::KITTI()
{
  conf.PLOT_PROB_THRESH       = 0.4;
  conf.NMS_THRESH             = 0.4;
  conf.PROB_THRESH            = 0.005;
  conf.TOP_N_DETECTION        = 64;

  conf.DATA_AUGMENTATION      = true;
  conf.DRIFT_X                = 150;
  conf.DRIFT_Y                = 100;
  conf.EXCLUDE_HARD_EXAMPLES  = false;

  conf.ANCHOR_BOX             = set_anchors( c);
  conf.ANCHORS                = len(mc.ANCHOR_BOX);
  conf.ANCHOR_PER_GRID        = 9;

  model.Load("../data/kitti/squeezeDet");
}

KITTI::~KITTI()
{
}

auto KITTI::set_anchors()
{
  const int H = 24, W = 78, B = 9;

  auto model_anchor = 
  anchor_shapes = np.reshape(
      [np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(IMAGE_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors;
}

KITTI::test()
{
}

#endif
