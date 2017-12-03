#ifdef _KITTI_HPP_

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <libgen.h>

#include "function.hpp"

static inline std::string dirname(std::string path)
{
  return std::string(dirname(const_cast<char *>(path.c_str())));
}

KITTI::KITTI()
  : _classes{"car", "pedestrian", "cyclist"}
{
  conf.CLASSES                = 3;
  conf.IMAGE_WIDTH            = 1248;
  conf.IMAGE_HEIGHT           = 384;

  conf.NMS_THRESH             = 0.4;
  conf.PROB_THRESH            = 0.005;
  conf.TOP_N_DETECTION        = 64;

  conf.EXCLUDE_HARD_EXAMPLES  = false;

  conf.ANCHOR_BOX             = set_anchors();
  conf.ANCHORS                = conf.ANCHOR_BOX.size();
  conf.ANCHOR_PER_GRID        = 9;

  image_idx = load_image_set_idx();
  rois = load_kitti_annotation();

  num_classes = _classes.size();
  for (int i = 0; i < num_classes; ++i)
    class_to_idx[_classes[i]] = i;

  model.configure(conf);
  model.Load("../data/kitti/squeezeDet");
}

KITTI::~KITTI()
{
}

std::vector<std::string> KITTI::load_image_set_idx()
{
  auto image_set_file = data_root_path + "/ImageSets" + "/"+image_set+".txt";
  std::ifstream ifs(image_set_file);
  if (!ifs.is_open())
    throw "load_image_set_idx: File does not exist";

  std::vector<std::string> image_idx;
  std::string line;
  int idx = 0;
  const int len = 1;
  while (ifs >> line) {
    image_idx.emplace_back(line);
    ++idx;
    if (idx == len) break;
  }

  return image_idx;
}

std::unordered_map<std::string, Mat2D<float>> KITTI::load_kitti_annotation()
{
  auto _get_obj_level = [](auto obj) {
    auto height = std::stof(obj[7]) - std::stof(obj[5]) + 1;
    auto truncation = std::stof(obj[1]);
    auto occlusion = std::stof(obj[2]);

    if (height >= 40 && truncation <= 0.15 && occlusion <= 0)
        return 1;
    else if (height >= 25 && truncation <= 0.3 && occlusion <= 1)
        return 2;
    else if (height >= 25 && truncation <= 0.5 && occlusion <= 2)
        return 3;
    else
        return 4;
  };

  std::unordered_map<std::string, Mat2D<float>> idx2annotation;
  for (auto index : image_idx) {
    auto filename = label_path + "/"+index+".txt";
    std::string line;
    std::ifstream ifs(filename);
    Mat2D<float> bboxes;
    while (std::getline(ifs, line)) {
      std::string tmp;
      std::vector<std::string> obj;
      std::istringstream strm(line);
      while (strm >> tmp)
        obj.emplace_back(tmp);

      float cls;
      // try {
        // cls = self._class_to_idx[obj[0].lower().strip()]
        std::string c;
        std::transform(obj[0].begin(), obj[0].end(), c.begin(), tolower);
        cls = class_to_idx[c];
      // }
      // catch (std::string& e) {
      //   continue;
      // }

      if (conf.EXCLUDE_HARD_EXAMPLES && _get_obj_level(obj) > 3)
        continue;

      auto xmin = std::stof(obj[4]);
      auto ymin = std::stof(obj[5]);
      auto xmax = std::stof(obj[6]);
      auto ymax = std::stof(obj[7]);
      auto bbox = bbox_transform_inv(xmin, ymin, xmax, ymax);
      bboxes.push_back(Mat1D<float>{bbox[0], bbox[1], bbox[2], bbox[3], cls});
    }

    idx2annotation[index] = bboxes;
  }

  return idx2annotation;
}

std::pair<std::vector<float>, std::vector<std::string>>
KITTI::evaluate_detections(std::string eval_dir, Mat4D<float> all_boxes)
{
  auto det_file_dir = eval_dir + "/detection_files" + "/data";

  // TODO: needed
  // if not os.path.isdir(det_file_dir):
  //   os.makedirs(det_file_dir)

  for (auto im_idx = 0; im_idx < (int)image_idx.size(); ++im_idx) {
    auto index = image_idx[im_idx];
    auto filename = det_file_dir+"/"+index+".txt";
    auto fp = fopen(filename.c_str(), "w");
    for (auto cls_idx = 0; cls_idx < (int)_classes.size(); ++cls_idx) {
      std::string cls = _classes[cls_idx];
      std::transform(cls.begin(), cls.end(), cls.begin(), tolower);

      Mat2D<float> dets = all_boxes[cls_idx][im_idx];
      for (auto det : dets) {
        fprintf(fp,
          "%s -1 -1 0.0 %.2f %.2f %.2f %.2f 0.0 0.0 0.0 0.0 0.0 "
          "0.0 0.0 %.3f\n",
          cls.c_str(), det[0], det[1], det[2], det[3], det[4]
        );
      }
    }
    fclose(fp);
  }

  std::stringstream cmd;
  cmd << eval_tool << " "
      << data_root_path << "/training" << " "
      << data_root_path << "/ImageSets" << "/"+image_set+".txt" << " "
      << dirname(det_file_dir) << " "
      << image_idx.size();

  printf("Running: %s\n", cmd.str().c_str());
  system(cmd.str().c_str());

  std::vector<float> aps;
  std::vector<std::string> names;
  for (auto cls : _classes) {
    auto det_file_name = dirname(det_file_dir) + "/stats_"+cls+"_ap.txt";
    std::ifstream ifs(det_file_name);
    if (ifs.is_open()) {
      // with open(det_file_name, "r") as f:
      //   lines = f.readlines()

      for (int i = 0; i < 3; ++i) {
        float val;
        std::string lval;
        std::getline(ifs, lval, '=');
        ifs >> val;
        aps.emplace_back(val);
      }

      // aps.emplace_back(float(lines[0].split("=")[1].strip()));
      // aps.emplace_back(float(lines[1].split("=")[1].strip()));
      // aps.emplace_back(float(lines[2].split("=")[1].strip()));
    }
    else {
      // aps.extend([0.0, 0.0, 0.0])
      aps.assign({0.0, 0.0, 0.0});
    }

    names.emplace_back(cls+"_easy");
    names.emplace_back(cls+"_medium");
    names.emplace_back(cls+"_hard");
  }

  // return aps, names;
  return std::make_pair(aps, names);
}

auto KITTI::analyze_detections(std::string detection_file_dir, std::string det_error_file)
{
  auto _save_detection = [&](FILE* fp, std::string idx, std::string err_type,
                             Mat1D<float> det, float score) {
    fprintf(fp,
      "%s %s %.1f %.1f %.1f %.1f %s %.3f\n",
      idx.c_str(), err_type.c_str(),
      det[0]-det[2]/2., det[1]-det[3]/2.,
      det[0]+det[2]/2., det[1]+det[3]/2.,
      _classes[(int)det[4]].c_str(),
      score
    );
  };

  // // load detections
  std::unordered_map<std::string, Mat2D<float>> det_rois;
  for (auto idx : image_idx) {
    auto det_file_name = detection_file_dir+"/"+idx+".txt";
    std::ifstream ifs(det_file_name);
    std::string line;
    Mat2D<float> bboxes;
    while (std::getline(ifs, line)) {
      std::string tmp;
      std::vector<std::string> obj;
      std::istringstream strm(line);
      while (strm >> tmp)
        obj.emplace_back(tmp);

      float xmin = std::stof(obj[4]);
      float ymin = std::stof(obj[5]);
      float xmax = std::stof(obj[6]);
      float ymax = std::stof(obj[7]);
      float score = std::stof(obj.back());
      float cls; {
        std::string c;
        std::transform(obj[0].begin(), obj[0].end(), c.begin(), tolower);
        cls = class_to_idx[c];
      }

      auto bbox = bbox_transform_inv(xmin, ymin, xmax, ymax);
      bboxes.emplace_back(
          Mat1D<float>{bbox[0], bbox[1], bbox[2], bbox[3], cls, score});
    }
    // bboxes.sort(key=lambda x: x[-1], reverse=True);
    std::sort(bboxes.begin(), bboxes.end(), [&](auto i, auto j) {
      return i.back() > j.back();
    });
    det_rois[idx] = bboxes;
  }

  // do error analysis
  auto num_correct         = 0.0;
  auto num_objs            = 0.0;
  auto num_dets            = 0.0;
  auto num_loc_error       = 0.0;
  auto num_cls_error       = 0.0;
  auto num_bg_error        = 0.0;
  auto num_repeated_error  = 0.0;
  auto num_detected_obj    = 0.0;

  auto fp = fopen(det_error_file.c_str(), "w");
  for (auto idx : image_idx) {
    auto gt_bboxes = rois[idx];
    num_objs += gt_bboxes.size();
    std::vector<bool> detected(gt_bboxes.size(), false);

    auto det_bboxes = det_rois[idx];
    if (gt_bboxes.size() < 1)
      continue;

    for (int i = 0; i < (int)det_bboxes.size(); ++i) {
      auto det = det_bboxes[i];
      if (i < (int)gt_bboxes.size())
        num_dets += 1;
      // auto ious = batch_iou(gt_bboxes[:, :4], det[:4]);
      auto ious = batch_iou(gt_bboxes, det);
      // Mat1D<float> ious;
      auto max_iou = max(ious);
      auto gt_idx = argmax(ious);
      if (max_iou > 0.1) {
        if ((int)gt_bboxes[gt_idx][4] == (int)det[4]) {
          if (max_iou >= 0.5) {
            if (i < (int)gt_bboxes.size()) {
              if (!detected[gt_idx]) {
                num_correct += 1;
                detected[gt_idx] = true;
              }
              else {
                num_repeated_error += 1;
              }
            }
          }
          else if (i < (int)gt_bboxes.size()) {
            num_loc_error += 1;
            _save_detection(fp, idx, "loc", det, det[5]);
          }
        }
        else if (i < (int)gt_bboxes.size()) {
          num_cls_error += 1;
          _save_detection(fp, idx, "cls", det, det[5]);
        }
      }
      else if (i < (int)gt_bboxes.size()) {
        num_bg_error += 1;
        _save_detection(fp, idx, "bg", det, det[5]);
      }
    }

    for (int i = 0; i < (int)gt_bboxes.size(); ++i) {
      if (!detected[i])
        _save_detection(fp, idx, "missed", gt_bboxes[i], -1.0);
    }
    num_detected_obj += std::count(detected.begin(), detected.end(), true);
  }
  fclose(fp);

  printf(
    "Detection Analysis:\n"
    "    Number of detections: %f\n"
    "    Number of objects: %f\n"
    "    Percentage of correct detections: %f\n"
    "    Percentage of localization error: %f\n"
    "    Percentage of classification error: %f\n"
    "    Percentage of background error: %f\n"
    "    Percentage of repeated detections: %f\n"
    "    Recall: %f\n"
    , num_dets, num_objs, num_correct/num_dets
    , num_loc_error/num_dets, num_cls_error/num_dets
    , num_bg_error/num_dets, num_repeated_error/num_dets
    , num_detected_obj/num_objs
  );

  std::unordered_map<std::string, float> out;
  // out["num of detections"]      = num_dets;
  // out["num of objects"]         = num_objs;
  // out["% correct detections"]   = num_correct/num_dets;
  // out["% localization error"]   = num_loc_error/num_dets;
  // out["% classification error"] = num_cls_error/num_dets;
  // out["% background error"]     = num_bg_error/num_dets;
  // out["% repeated error"]       = num_repeated_error/num_dets;
  // out["% recall"]               = num_detected_obj/num_objs;

  return out;
}

auto KITTI::do_detection_analysis_in_eval(std::string eval_dir)
{
  auto det_file_dir  = eval_dir + "/detection_files" + "/data";
  auto det_error_dir = eval_dir + "/detection_files" + "/error_analysis";
  auto det_error_file = det_error_dir + "det_error_file.txt";

  // if not os.path.exists(det_error_dir):
  //   os.makedirs(det_error_dir)

  auto stats = analyze_detections(det_file_dir, det_error_file);

  return stats;
}

Mat2D<float> KITTI::set_anchors()
{
  const int H = 24, W = 78, B = 9;

  float anchor_shapes[B][2] = {
    {  36.,  37.}, { 366., 174.}, { 115.,  59.},
    { 162.,  87.}, {  38.,  90.}, { 258., 173.},
    { 224., 108.}, {  78., 170.}, {  72.,  43.}
  };

  auto center_x = zeros<float>(W);
  for (int i = 0; i < W; ++i) {
    center_x[i] = static_cast<float>(i+1)/(W+1) * conf.IMAGE_WIDTH;
  }

  auto center_y = zeros<float>(H);
  for (int i = 0; i < H; ++i) {
    center_y[i] = static_cast<float>(i+1)/(H+1) * conf.IMAGE_HEIGHT;
  }

  auto anchors = zeros<float>(H*W*B, 4);
  int idx = 0;
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      for (int k = 0; k < B; ++k) {
        anchors[idx][0] = center_x[j];
        anchors[idx][1] = center_y[i];
        anchors[idx][2] = anchor_shapes[k][0];
        anchors[idx][3] = anchor_shapes[k][1];
        ++idx;
      }
    }
  }

  return anchors;
}

BBoxMask KITTI::predict(int sample)
{
  std::ostringstream filename;
  filename << std::setfill('0') << std::setw(6)
           << image_path << "/" << image_idx[sample] << ".png";

  // cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
  auto mask = model.calc(filename.str());

  return mask;
}

void KITTI::test()
{
  std::vector<std::string> ap_names;
  for (auto cls : _classes) {
    ap_names.emplace_back(cls+"_easy");
    ap_names.emplace_back(cls+"_medium");
    ap_names.emplace_back(cls+"_hard");
  }

  auto num_images = image_idx.size();

  // auto all_boxes = zeros<float>(num_classes, num_images, 1);
  Mat4D<float> all_boxes(num_classes);
  for (auto& b : all_boxes)
    b.resize(num_images);

  auto num_detection = 0.0;
  for (int i = 0; i < (int)num_images; ++i) {
    std::cout << i+1 << " / " << num_images << std::endl;
    // images, scales = imdb.read_image_batch(shuffle=False)

    auto mask = predict(i);
    auto scales = mask.scales;
    auto det_boxes = mask.det_boxes;
    auto det_probs = mask.det_probs;
    auto det_class = mask.det_class;

    // for (int j = 0; j < (int)det_boxes.size(); ++j) {
      // det_boxes[j, :, 0::2] /= scales[j][0]
      // det_boxes[j, :, 1::2] /= scales[j][1]
      for (int j = 0; j < (int)det_boxes.size(); ++j) {
        det_boxes[j][0] /= scales[1];
        det_boxes[j][1] /= scales[0];
        det_boxes[j][2] /= scales[1];
        det_boxes[j][3] /= scales[0];
      }

      // det_bbox, score, det_class = model.filter_prediction(
      //     det_boxes[j], det_probs[j], det_class[j]);
      auto filtered_mask = model.filter(mask);
      Mat2D<float> _bbox  = filtered_mask.det_boxes;
      Mat1D<float> _score = filtered_mask.det_probs;
      Mat1D<int>   _class = filtered_mask.det_class;
      // save_txt("now_det_boxes.txt", filtered_mask.det_boxes);
      // save_txt("now_det_probs.txt", filtered_mask.det_probs);
      // save_txt("now_det_class.txt", filtered_mask.det_class);

      const int mask_len = _bbox.size();
      // for (auto& b : all_boxes)
      //   b[i].resize(mask_len);
      for (int k = 0; k < mask_len; ++k) {
        auto c = _class[k];
        auto b = _bbox[k];
        auto s = _score[k];
        auto bbox = bbox_transform(b[0], b[1], b[2], b[3]);
        all_boxes[c][i].emplace_back(
            Mat1D<float>{bbox[0], bbox[1], bbox[2], bbox[3], s});
        // all_boxes[c][i][k] =
        //     Mat1D<float>{bbox[0], bbox[1], bbox[2], bbox[3], s};
      }

      num_detection += mask_len;
    // }
  }

  printf("Evaluating detections...\n");

  std::vector<float> aps;
  std::vector<std::string> names;
  std::tie(aps, ap_names) =
    evaluate_detections(eval_dir, all_boxes);

  printf("Evaluation summary:\n");

  printf("  Average number of detections per image: %f:\n",
    num_detection/num_images);

  printf("  Average precisions:\n");
  for (int i = 0; i < (int)aps.size(); ++i) {
    printf("    %s: %.3f\n", ap_names[i].c_str(), aps[i]);
  }

  printf("    Mean average precision: %.3f\n",
      std::accumulate(aps.begin(), aps.end(), 0.0)/aps.size());
  // feed_dict[eval_summary_phs["num_det_per_image"]] = num_detection/num_images

  printf("Analyzing detections...\n");
  auto stats = do_detection_analysis_in_eval(eval_dir);

  return;
}

#endif
