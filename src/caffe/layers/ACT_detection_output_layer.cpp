#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/ACT_detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACTDetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ACTDetectionOutputParameter& act_detection_output_param =
      this->layer_param_.act_detection_output_param();
  // length of input sequence
  sequence_length_= act_detection_output_param.sequence_length(); 
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;
  // ACT-detector: option for saving full tubes and scores 
  save_full_ = act_detection_output_param.save_full();
  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  eta_ = detection_output_param.nms_param().eta();
  CHECK_GT(eta_, 0.);
  CHECK_LE(eta_, 1.);
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
  const SaveOutputParameter& save_output_param =
      detection_output_param.save_output_param();
  output_directory_ = save_output_param.output_directory();
  if (!output_directory_.empty()) {
    if (boost::filesystem::is_directory(output_directory_)) {
      boost::filesystem::remove_all(output_directory_);
    }
    if (!boost::filesystem::create_directories(output_directory_)) {
        LOG(FATAL) << "Failed to create directory: " << output_directory_;
    }
  }
  output_name_prefix_ = save_output_param.output_name_prefix();
  need_save_ = output_directory_ == "" ? false : true;
  output_format_ = save_output_param.output_format();
  if (save_output_param.has_label_map_file()) {
    string label_map_file = save_output_param.label_map_file();
    if (label_map_file.empty()) {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
      need_save_ = false;
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  } else {
    need_save_ = false;
  }
  if (save_output_param.has_name_size_file()) {
    string name_size_file = save_output_param.name_size_file();
    if (name_size_file.empty()) {
      // Ignore saving if there is no name_size_file provided.
      LOG(WARNING) << "Provide name_size_file if output results to files.";
      need_save_ = false;
    } else {
      std::ifstream infile(name_size_file.c_str());
      CHECK(infile.good())
          << "Failed to open name size file: " << name_size_file;
      // The file is in the following format:
      //    name height width
      //    ...
      string name;
      int height, width;
      while (infile >> name >> height >> width) {
        names_.push_back(name);
        sizes_.push_back(std::make_pair(height, width));
      }
      infile.close();
      if (save_output_param.has_num_test_image()) {
        num_test_image_ = save_output_param.num_test_image();
      } else {
        num_test_image_ = names_.size();
      }
      CHECK_LE(num_test_image_, names_.size());
    }
  } else {
    need_save_ = false;
  }
  has_resize_ = save_output_param.has_resize_param();
  if (has_resize_) {
    resize_param_ = save_output_param.resize_param();
  }
  name_count_ = 0;
  visualize_ = detection_output_param.visualize();
  if (visualize_) {
    visualize_threshold_ = 0.6;
    if (detection_output_param.has_visualize_threshold()) {
      visualize_threshold_ = detection_output_param.visualize_threshold();
    }
    data_transformer_.reset(
        new DataTransformer<Dtype>(this->layer_param_.transform_param(),
                                   this->phase_));
    data_transformer_->InitRand();
    save_file_ = detection_output_param.save_file();
  }
  bbox_preds_.ReshapeLike(*(bottom[0]));
  if (!share_location_) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }
  conf_permute_.ReshapeLike(*(bottom[1]));
}

// reshape 
template <typename Dtype>
void ACTDetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (need_save_) {
    CHECK_LE(name_count_, names_.size());
    if (name_count_ % num_test_image_ == 0) {
      // Clean all outputs.
      if (output_format_ == "VOC") {
        boost::filesystem::path output_directory(output_directory_);
        for (map<int, string>::iterator it = label_to_name_.begin();
             it != label_to_name_.end(); ++it) {
          if (it->first == background_label_id_) {
            continue;
          }
          std::ofstream outfile;
          boost::filesystem::path file(
              output_name_prefix_ + it->second + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          outfile.open(out_file.string().c_str(), std::ofstream::out);
        }
      }
    }
  }
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  if (bbox_preds_.num() != bottom[0]->num() ||
      bbox_preds_.count(1) != bottom[0]->count(1)) {
    bbox_preds_.ReshapeLike(*(bottom[0]));
  }
  if (!share_location_ && (bbox_permute_.num() != bottom[0]->num() ||
      bbox_permute_.count(1) != bottom[0]->count(1))) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }
  if (conf_permute_.num() != bottom[1]->num() ||
      conf_permute_.count(1) != bottom[1]->count(1)) {
    conf_permute_.ReshapeLike(*(bottom[1]));
  }
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(num_priors_ * num_loc_classes_ * 4 * sequence_length_ , bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // ACT-detector: each row is a [3 dimensional vector + sequence_length vector x 4]
  // eg. if sequence_length = 6, each row is 27D
  // the 3D vector is: [image_id, label, confidence] 
  // sequence_length x the coordinates: [ xmin, ymin, xmax, ymax]  
  if(save_full_){ 
    top_shape.push_back(3 + 4* sequence_length_); 
  } else {
    top_shape.push_back(1 + 4*sequence_length_ + num_classes_); 
  }
  top[0]->Reshape(top_shape);
}

// forward
template <typename Dtype>
void ACTDetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();

  // Retrieve all location predictions.
  vector<ACTLabelTube> all_loc_preds;
  ACTGetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds, sequence_length_);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores); 

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<ACTNormalizedTube> prior_tubes;
  vector<vector<float> > prior_variances;
  ACTGetPriorTubes(prior_data, num_priors_, &prior_tubes, &prior_variances, sequence_length_);

  // Decode all loc predictions to bboxes.
  vector<ACTLabelTube> all_decode_tubes;
  const bool clip_bbox = false;
  ACTDecodeTubesAll(all_loc_preds, prior_tubes, prior_variances, num,
                  share_location_, num_loc_classes_, background_label_id_,
                  code_type_, variance_encoded_in_target_, clip_bbox,
                  &all_decode_tubes);  

  // ACT-detector: save all tubes/scores
  if( save_full_){
    vector<int> top_shape(2, 1);
    // ACT-detector: num_priors are num_prior tubes
    top_shape.push_back(num_priors_ * num ); 
    // ACT-detector: first is the index of the image in the batch, then tube coordinates, then class scores
    int ncols = 1 + 4* sequence_length_ + num_classes_; 
    top_shape.push_back(ncols); 
    top[0]->Reshape(top_shape);
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    // ACT-detector: for each image in the batch
    for (int i = 0; i < num; ++i) { 
      CHECK(share_location_) << "ACT-detector: save full tubes option is implemented only with share location";      
      const map<int, vector<float> >& conf_scores = all_conf_scores[i];
      const vector<ACTNormalizedTube>& tubes = all_decode_tubes[i].find(-1)->second;
      
      for(int j=0 ; j<num_priors_ ; ++j){
        const int start_idx = i*num_priors_*ncols + j*ncols;
        const ACTNormalizedTube tube = tubes[j];
        // ACT-detector: batch idx
        top_data[ start_idx ] = i;
        // ACT-detector: for each box in the sequence
        for (int jj=0; jj< sequence_length_ ; ++jj){
            top_data[start_idx + 4 * jj + 1 + 0] = tube[jj].xmin();
            top_data[start_idx + 4 * jj + 1 + 1] = tube[jj].ymin();
            top_data[start_idx + 4 * jj + 1 + 2] = tube[jj].xmax();
            top_data[start_idx + 4 * jj + 1 + 3] = tube[jj].ymax();
        }
        // ACT-detector: scoring
        for(int c=0 ; c<num_classes_ ; ++c){
          top_data[start_idx + 1 + 4*sequence_length_ + c] = conf_scores.find(c)->second[j];
        }
      }
      
    }
    return; // ACT-detector: finished saving all tubes/scores
  }
                    
  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const ACTLabelTube& decode_tubes = all_decode_tubes[i];
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      const vector<float>& scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_tubes.find(label) == decode_tubes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<ACTNormalizedTube>& tubes = decode_tubes.find(label)->second;
      ACTApplyNMSTubeFast(tubes, scores, confidence_threshold_, nms_threshold_, eta_,
          top_k_, &(indices[c])); 
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for " << label;
          continue;
        }
        const vector<float>& scores = conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(std::make_pair(
                  scores[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  } 

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  // ACT-detector: dimensionality of each row
  top_shape.push_back(3 + 4* sequence_length_); 
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      // ACT-detector: dimensionality of each row for the whole sequence
      top_data += 3 + 4* sequence_length_; 
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }

  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const ACTLabelTube& decode_tubes = all_decode_tubes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_tubes.find(loc_label) == decode_tubes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      const vector<ACTNormalizedTube>& tubes =
          decode_tubes.find(loc_label)->second;
      vector<int>& indices = it->second;
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
          << "Cannot find label: " << label << " in the label map.";
        CHECK_LT(name_count_, names_.size());
      } 
      // ACT-detector: dimensionality of each row
      int row_size = 3 + 4* sequence_length_; 
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * row_size] = i; 
        top_data[count * row_size + 1] = label;
        top_data[count * row_size + 2] = scores[idx];
        const ACTNormalizedTube& tube = tubes[idx];
        for (int jj =0; jj<sequence_length_ ; ++jj) {
            top_data[count * row_size + 4 * jj + 3] = tube[jj].xmin();
            top_data[count * row_size + 4 * jj + 4] = tube[jj].ymin();
            top_data[count * row_size + 4 * jj + 5] = tube[jj].xmax();
            top_data[count * row_size + 4 * jj + 6] = tube[jj].ymax(); 
        }
        if (need_save_) {
          CHECK(false) << "ACT-detector: saving tubes is not implemeted";          
        }
        ++count;
      }
    } 
    if (need_save_) {
      CHECK(false) << "ACT-detector: saving tubes is not implemeted";      
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ACTDetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(ACTDetectionOutputLayer);
REGISTER_LAYER_CLASS(ACTDetectionOutput);

}  // namespace caffe
