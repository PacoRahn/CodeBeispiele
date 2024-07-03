/**
 * @file lane_detection_node.cpp
 * @brief implementation of the lane_detection
 * @author PSAF
 * @date 2022-06-01
 */
#include "psaf_lane_detection/lane_detection_node.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

LaneDetectionNode::LaneDetectionNode()
: LaneDetectionInterface(
    LANE_DETECTION_NODE_NAME,
    NBR_OF_CAMS_RGB,
    CAM_TOPIC_RGB,
    STATE_TOPIC,
    LANE_MARKINGS_TOPIC,
    STOP_LINE_TOPIC,
    STATUS_INFO_TOPIC,
    rclcpp::QoS(rclcpp::KeepLast {10}))
{
  // Dynamic reconfigure of the used algorithm
  this->declare_parameter("use_secondary_algorithm", false);
}
// Counter for testing purposes
int counter_test = 0;

// Function for processing images, takes a reference to the input image and the sensor value
void LaneDetectionNode::processImage(cv::Mat & img, int sensor)
{
  // The sensor value is not used in this function, so it can be set to void
  (void)sensor;

  // Get image dimensions
  int width = img.cols;
  int height = img.rows;

  // Declare Mat objects for grayscale, binary, and transformed images
  cv::Mat gray, binary, transformed;

  // Declare a 3x3 matrix for the homography transformation
  cv::Mat homography = cv::Mat::zeros(3, 3, CV_64F);

  // Set homography according to the used car
  if (PSAF1) {
    // homography for our camera calibration
    homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf1)).clone();
  } else {
    homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf2)).clone();
  }

  // Declare Mat objects for the input image, grayscale image, and blurred image
  cv::Mat frame;
  cv::cvtColor(img, frame, cv::COLOR_BGR2RGB);
  cv::Mat img_gray;
  cv::cvtColor(frame, img_gray, cv::COLOR_RGB2GRAY);
  cv::Mat img_blur;
  cv::GaussianBlur(img_gray, img_blur, cv::Size(3, 3), 0);

  // Declare a Mat object for the edges image and apply Canny edge detection
  cv::Mat edges;
  cv::Canny(img_blur, edges, 100, 200, 3, false);

  // Increment the testing counter
  counter_test++;

  // Declare a Mat object for the cropped image and apply the homography transformation
  cv::Mat cropped_image;
  transformImage(edges, homography, cropped_image);

  // Clone the cropped image and convert it to grayscale
  cv::Mat cropped_image2 = cropped_image.clone();
  cv::cvtColor(cropped_image2, cropped_image2, cv::COLOR_GRAY2BGR);

  // Declare vectors for storing points on the middle, right, and left lines, and a vector for storing lanes
  std::vector<cv::Point> points_on_middle_line;
  std::vector<cv::Point> points_on_right_line;
  std::vector<cv::Point> points_on_left_line;

  std::vector<std::vector<std::tuple<cv::Vec2i, double>>> lanes(2);

  // Find the lane markings in the cropped image
  findLane(cropped_image, cropped_image2, lanes);

  // Extract the right lane from the lanes vector and store the points on the middle line
  std::vector<std::tuple<cv::Vec2i, double>> right_lane = lanes[0];
  for (std::tuple<cv::Vec2i, double> line_segment : right_lane) {
    RCLCPP_ERROR(this->get_logger(), "right lane seems not to be empty ");
    cv::Point point = std::get<0>(line_segment);
    double angle = std::get<1>(line_segment);
    cv::Point point_on_middle = getPointFromAngleAndDistance(point, angle, 135, true);
    points_on_middle_line.push_back(point_on_middle);
    cv::circle(cropped_image2, point_on_middle, 4, cv::Scalar(0, 0, 255), -1);
  }

  std::vector<std::tuple<cv::Vec2i, double>> left_lane = lanes[1];
  for (std::tuple<cv::Vec2i, double> line_segment : left_lane) {
    RCLCPP_ERROR(this->get_logger(), "left lane seems not to be empty ");
    cv::Point point = std::get<0>(line_segment);
    double angle = std::get<1>(line_segment);
    cv::Point point_on_middle = getPointFromAngleAndDistance(point, angle, 135, false);
    points_on_middle_line.push_back(point_on_middle);
    cv::circle(cropped_image2, point_on_middle, 4, cv::Scalar(0, 0, 255), -1);
  }

  last_lane_markings_positions_.push_back(points_on_left_line);
  last_lane_markings_positions_.push_back(points_on_middle_line);
  last_lane_markings_positions_.push_back(points_on_right_line);

  std::ostringstream oss4;
  oss4 << "/home/psaf/Documents/Pictures/box/LaneDetection_" << counter_test << ".jpg";
  imwrite(oss4.str(), cropped_image2);

  // Resize image if not 640x480
  if (width != 640 || height != 480) {
    resize(img, img, cv::Size(640, 480));
  }
}

void LaneDetectionNode::update()
{
  // Publish Lane Markings information
  if (!last_lane_markings_positions_.empty()) {
    publishLaneMarkings(
      last_lane_markings_positions_.at(0), last_lane_markings_positions_.at(1),
      last_lane_markings_positions_.at(2), has_blocked_area_, no_overtaking_, side_);
    last_lane_markings_positions_.clear();
  }

  // Build the status info message and publish it, but not if state = -1
  if (current_state_ != -1) {
    libpsaf_msgs::msg::StatusInfo status_msg;
    status_msg.header.stamp = rclcpp::Time(0);
    status_msg.header.frame_id = "";
    status_msg.type = status_info_;
    publishStatus(status_msg);
  }

  // Publish Stop Line information if the state is appropriate
  if (current_state_ == 10 || current_state_ == 13 || current_state_ == 14) {
    int x = last_stop_line_position_.x;
    int y = last_stop_line_position_.y;
    publishStopLine(stop_line_found_, stop_line_type_, x, y);
  }
}

void LaneDetectionNode::updateState(std_msgs::msg::Int64::SharedPtr state)
{
  current_state_ = state->data;
}

void LaneDetectionNode::grayscaleImage(cv::Mat & img, cv::Mat & result)
{
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Image is empty");
    return;
  }
  // Check if image has only one channel. If true, do not convert to grayscale
  if (img.channels() == 1) {
    result = img.clone();
  } else {
    // If the image has multiple channels, convert it to grayscale
    cv::cvtColor(img, result, cv::COLOR_BGR2GRAY);
  }
}

void LaneDetectionNode::binarizeImage(
  cv::Mat & img, cv::Mat & result, int threshold_low,
  int threshold_high)
{
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Image is empty");
    return;
  }

  if (threshold_low >= threshold_high) {
    RCLCPP_ERROR(this->get_logger(), "Threshold low is higher than or equal threshold high");
    return;
  }
  // Apply binary thresholding to the input image
  cv::threshold(img, result, threshold_low, threshold_high, cv::THRESH_BINARY);
}

void LaneDetectionNode::transformImage(cv::Mat & img, cv::Mat & homography, cv::Mat & result)
{
  if (DEBUG) {
    std::cout << "PSAF1 homography" << std::endl;
    std::cout << homography << std::endl;
  }

  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Image is empty");
    return;
  }
  // check if homography is valid (not empty and 3x3)
  if (homography.empty() || homography.rows != 3 || homography.cols != 3) {
    RCLCPP_ERROR(this->get_logger(), "Homography is empty or not 3x3");
    return;
  }
  cv::Size size(480, 1280);
  // Transform image based on our calibration
  warpPerspective(img, result, homography, size);
}

void LaneDetectionNode::resizeImage(cv::Mat & img, cv::Mat & result)
{
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Input Image is empty");
    return;
  }
  if (img.cols == 640 && img.rows == 480) {
    result = img.clone();
  } else {
    // Resize image to 640x480
    resize(img, result, cv::Size(640, 480), cv::INTER_LINEAR);
  }
}

void LaneDetectionNode::extractLaneMarkings(cv::Mat & img)
{
  std::vector<std::vector<cv::Point>> lane_markings;
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Input Image is empty");
    return;
  }

  // Get image width and height
  int width = img.cols;
  std::vector<int> y_values{0, 26, 53, 80, 106, 133, 160, 186, 213, 240, 253, 266, 280, 293, 306,
    320, 333, 346, 360, 373, 386, 400, 413, 426, 440, 453, 466};
  // Create a vector for the left, center and right lane points
  std::vector<cv::Point> left_lane_points;
  std::vector<cv::Point> center_lane_points;
  std::vector<cv::Point> right_lane_points;
  for (std::size_t y = 0; y < y_values.size(); y++) {
    // Get the row from the image depending on the y value
    cv::Mat row = img.row(y_values[y]);
    // Find the non zero pixels in the row
    cv::Mat non_zero_pixels;
    findNonZero(row, non_zero_pixels);
    int sum = 0;
    int count = 0;
    // Iterate over the non endzero pixels and average, if the next pixel is a direct neighbor
    for (int x = 0; x < non_zero_pixels.rows - 1; x++) {
      if (non_zero_pixels.at<cv::Point>(x + 1, 0).x - non_zero_pixels.at<cv::Point>(x, 0).x == 1) {
        sum += non_zero_pixels.at<cv::Point>(x, 0).x;
        count++;
      } else {
        if (count == 0) {
          sum = non_zero_pixels.at<cv::Point>(x, 0).x;
          count = 1;
        }
        int average = sum / count;
        cv::Mat slidingWindowHoughTransform(cv::Mat & img,
          std::vector<cv::Vec4i> & hlines, int height_prop, int width_prop);

        // Classify lane points based on their position
        if (average < static_cast<int>(width * .18)) {
          left_lane_points.push_back(cv::Point(average, y_values[y]));
        } else if (average > static_cast<int>(width * .18) && average < width / 2) {
          center_lane_points.push_back(cv::Point(average, y_values[y]));
        } else {
          if (count < 80) {
            right_lane_points.push_back(cv::Point(average, y_values[y]));
          }
        }
        sum = 0;
        count = 0;
      }
      if (x == non_zero_pixels.rows - 2) {
        if (count == 0) {
          sum = non_zero_pixels.at<cv::Point>(x, 0).x;
          count = 1;
        }
        int average = sum / count;
        // Classify lane points based on their position
        if (average < static_cast<int>(width * .18)) {
          left_lane_points.push_back(cv::Point(average, y_values[y]));
        } else if (average > static_cast<int>(width * .18) && average < width / 2) {
          center_lane_points.push_back(cv::Point(average, y_values[y]));
        } else {
          if (count < 80) {
            right_lane_points.push_back(cv::Point(average, y_values[y]));
          }
        }
        sum = 0;
        count = 0;
      }
    }
  }
  if (DEBUG) {
    imshow("Lane Markings", img);
    cv::waitKey(32);
    std::cout << "Found " << left_lane_points.size() << " left lane points" << std::endl;
    std::cout << "Found " << center_lane_points.size() << " center lane points" << std::endl;
    std::cout << "Found " << right_lane_points.size() << " right lane points" << std::endl;
  }
  // Add the left, center, and right lane points to the lane_markings vector
  lane_markings.push_back(left_lane_points);
  lane_markings.push_back(center_lane_points);
  lane_markings.push_back(right_lane_points);
  // Set the side_ variable to 0 (unknown) and no_overtaking_ to true
  side_ = 0;
  no_overtaking_ = true;
  // Store the current lane markings positions in last_lane_markings_positions_
  last_lane_markings_positions_ = lane_markings;
}

// This function extracts lane markings using a secondary algorithm and logs the information
void LaneDetectionNode::extractLaneMarkingsSecondary(cv::Mat & img)
{
  RCLCPP_INFO(this->get_logger(), "Using secondary lane markings algorithm");
  (void)img;
}

// This function extracts stop lines from an input image and sets the corresponding flags and values
void LaneDetectionNode::extractStopLine(cv::Mat & img)
{
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Input Image is empty");
    stop_line_found_ = false;
    last_stop_line_position_ = cv::Point(-1, -1);
    stop_line_type_ = 0;
    return;
  }
  // If a stop line is found in the input image, set the corresponding flags and values
  stop_line_found_ = true;
  last_stop_line_position_ = cv::Point(309, 76);
  stop_line_type_ = 0;
}

int LaneDetectionNode::getCurrentState()
{
  return current_state_;
}

cv::Mat LaneDetectionNode::getLastImage()
{
  return last_image_;
}

// Find the left lane in the input image using a sliding window
bool LaneDetectionNode::findLeftLane(
  cv::Mat & img, cv::Mat & img2,
  std::vector<std::tuple<cv::Vec2i, double>> & left_lane,
  int width, int height, int search_step)
{
  // Set the number of not found lanes to a threshold value
  int not_found = 3;
  // Set the start point for the sliding window
  cv::Vec2i start_point = cv::Vec2i(55, 1279 - search_step * height);

  // Find the starting point of the left lane
  bool starting_point_found = findLineInWindow(img, img2, left_lane, width, height, start_point);
  RCLCPP_ERROR(this->get_logger(), "findLineInWindow in findLeftLane finished");

  // If the starting point of the left lane was found, find the rest of the lane
  if (starting_point_found) {
    // Find rest of left lane
    findRestOfLane(img, img2, left_lane, width, height, not_found);
    RCLCPP_ERROR(this->get_logger(), "findRestOfLaneFound in findLeftLane finished");
    return true;
  }

  return false;
}

// Find the right lane in the input image using a sliding window
bool LaneDetectionNode::findRightLane(
  cv::Mat & img, cv::Mat & img2,
  std::vector<std::tuple<cv::Vec2i, double>> & right_lane,
  int width, int height, int search_step)
{
  // Set the number of not found lanes to a threshold value
  int not_found = 2;
  // Set the start point for the sliding window
  cv::Vec2i start_point = cv::Vec2i(279, 1279 - search_step * height);

  // Find the starting point of the left lane
  bool starting_point_found = findLineInWindow(img, img2, right_lane, width, height, start_point);

  // If the starting point of the right lane was found, find the rest of the lane
  if (starting_point_found) {
    // Find rest of right lane
    findRestOfLane(img, img2, right_lane, width, height, not_found);
    return true;
  }

  return false;
}

void LaneDetectionNode::slidingWindowHoughTransform(
  cv::Mat & img, std::vector<cv::Vec4i> & hlines,
  int height_prop, int width_prop)
{
  // Calculate dimensions of window
  int orig_width = img.cols;
  int orig_height = img.rows;
  int width = orig_width / width_prop;
  int height = orig_height / height_prop;

  std::vector<cv::Vec4i> hlinesSW;
  int line_counter = 1;

  // Loop through each sliding window
  for (int x = 0; x < width_prop; x++) {
    // Calculate start and end points of window in y-direction
    int y_start = x * width;
    int y_end = y_start + width;
    (void)y_end;
    // Loop through each sliding window in x-direction
    for (int y = 0; y < height_prop; y++) {
      // Calculate start and end points of window in x-direction
      int x_start = orig_height - (y + 1) * height;
      int x_end = orig_height - y * height;
      (void)x_end;

      // Calculate region of interest
      cv::Mat window_content(img(cv::Rect(y_start, x_start, width, height)));

      // Apply Hough transform to extract lines from the window
      std::vector<cv::Vec4i> hlinesSW;
      HoughLinesP(window_content, hlinesSW, 2, CV_PI / 180, 40, 30, 50);

      // Adjust coordinates of detected lines to match original image
      for (std::size_t i = 0; i < hlinesSW.size(); i++) {
        hlinesSW[i][0] += y_start;
        hlinesSW[i][1] += x_start;
        hlinesSW[i][2] += y_start;
        hlinesSW[i][3] += x_start;
      }
      // Append detected lines to overall list of lines
      hlines.insert(hlines.end(), hlinesSW.begin(), hlinesSW.end());
      line_counter++;
    }
  }
  int num_lines = static_cast<int>(hlines.size());
}

bool LaneDetectionNode::findLineInWindow(
  cv::Mat & img, cv::Mat & img2,
  std::vector<std::tuple<cv::Vec2i, double>> & lane,
  int width, int height, cv::Vec2i start_point,
  double min_theta, double max_theta)
{
  // Initialize variables
  cv::Vec4i average_line;
  average_line[0] = 0;
  average_line[1] = 0;
  average_line[2] = 0;
  average_line[3] = 0;
  std::tuple<cv::Vec2i, double> lane_segment;

  // Set starting point and region of interest
  int x_start = start_point[0];
  int y_start = start_point[1];

  cv::Rect window = cv::Rect(x_start, y_start, width, height);
  cv::Mat window_content(img(window));

  std::vector<cv::Vec4i> hlinesSW;

  // Detect lines using Hough transform
  HoughLinesP(window_content, hlinesSW, 2, CV_PI / 180, 20, 30, 2);
  std::size_t number_of_lines = hlinesSW.size();
  int num_lines_int = static_cast<int>(number_of_lines);

  // If no lines are detected, return false
  if (num_lines_int <= 0) {
    return false;
  }

  // Process the detected lines
  cv::Point pt1, pt2;
  int effective_number_of_lines = 0;
  // Print for Debug
  // RCLCPP_ERROR(this->get_logger(), "number_of_lines = %d", number_of_lines);
  for (std::size_t i = 0; i < number_of_lines; i++) {
    // Get the two endpoints of the line segment
    pt1.x = hlinesSW[i][0];
    pt1.y = hlinesSW[i][1];
    pt2.x = hlinesSW[i][2];
    pt2.y = hlinesSW[i][3];

    // Calculate the angle of the line
    double theta = calculateAngles(pt1, pt2);

    // If the angle is within the specified range, add it to the average line
    if (theta >= min_theta and theta <= max_theta) {
      average_line[0] += hlinesSW[i][0];
      average_line[1] += hlinesSW[i][1];
      average_line[2] += hlinesSW[i][2];
      average_line[3] += hlinesSW[i][3];
      effective_number_of_lines++;
    }
  }
  // Print for Debug
  // RCLCPP_ERROR(this->get_logger(), "effective_number_of_lines = %d", effective_number_of_lines);
  // If no lines were added to the average line, return false
  if (effective_number_of_lines == 0) {
    return false;
  }
  // Calculate the average line
  pt1.x = average_line[0] / effective_number_of_lines + x_start;
  pt1.y = average_line[1] / effective_number_of_lines + y_start;
  pt2.x = average_line[2] / effective_number_of_lines + x_start;
  pt2.y = average_line[3] / effective_number_of_lines + y_start;
  line(img2, pt1, pt2, cv::Scalar(255, 0, 255), 3, 8);

  // Calculate the midpoint of the detected line and its angle
  cv::Point pta;
  pta.x = (pt1.x + pt2.x) / 2;
  pta.y = (pt1.y + pt2.y) / 2;
  // Store the midpoint and the final angle of the detected lane segment in a tuple
  std::get<0>(lane_segment) = cv::Vec2i(static_cast<int>(pta.x), static_cast<int>(pta.y));
  double final_angle = calculateAngles(pt1, pt2);
  std::get<1>(lane_segment) = final_angle;
  lane.insert(lane.end(), lane_segment);
  return true;
}

void LaneDetectionNode::findLane(
  cv::Mat & img, cv::Mat & img2, std::vector<std::vector<std::tuple<cv::Vec2i, double>>> & lanes)
{
  int width = 200;
  int height = 65;

  std::vector<std::tuple<cv::Vec2i, double>> lane_right;
  std::vector<std::tuple<cv::Vec2i, double>> lane_left;
  int search_step = 1;
  while (search_step <= 10) {
    // Find the right lane
    if (findRightLane(img, img2, lane_right, width, height, search_step)) {
      // Check if the number of points in the lane is sufficient
      if (lane_right.size() >= 6) {
        lanes[0] = lane_right;
        search_step = 10;
        break;
      }
    }
    // Find the left lane
    if (findLeftLane(img, img2, lane_left, width, height, search_step)) {
      // Check if the number of points in the lane is sufficient
      if (lane_left.size() >= 4) {
        lanes[1] = lane_left;
        search_step = 10;
        break;
      }
    }
    search_step++;
  }
  // Output the search step to the console for debugging purposes
  // RCLCPP_ERROR(this->get_logger(), "%d", search_step);
}

// This function finds the rest of the lane given a start point and a certain number of search iterations
void LaneDetectionNode::findRestOfLane(
  cv::Mat & img, cv::Mat & img2,
  std::vector<std::tuple<cv::Vec2i, double>> & start_of_lane,
  int width, int height, int num)
{
  bool search_for_lane = true;
  int no_lane_found_counter = 0;
  double min_theta;
  double max_theta;
  // Split angle is used to divide the window into two parts when searching for the next lane segment
  double split_angle = acos(width / sqrt((pow((width), 2) + pow((height), 2))));
  int iteration = 1;
  int max_iteration = 20;
  // Loop until the desired number of search iterations is reached
  while (search_for_lane and iteration < max_iteration) {
    // Get the last segment of the lane and calculate the range of possible angles for the next segment
    // RCLCPP_ERROR(this->get_logger(), "no_lane_found_counter at beginning %d", no_lane_found_counter);
    std::tuple<cv::Vec2i, double> last_segment = start_of_lane.back();
    min_theta = std::get<1>(last_segment) - CV_PI / 2 *
      (0.1 + 0.1 * no_lane_found_counter);

    if (min_theta < 0) {
      min_theta = 0;
    }
    max_theta = std::get<1>(last_segment) + CV_PI / 2 *
      (0.1 + 0.1 * no_lane_found_counter);

    if (max_theta > CV_PI) {
      max_theta = CV_PI;
    }
    // Calculate the next start point for the window search
    cv::Vec2i next_start_point = getNextStartPoint(
      last_segment, no_lane_found_counter, split_angle, width,
      height);
    // Check if the next start point is out of image boundaries
    if (pointOutOfImage(next_start_point, width, height)) {
      RCLCPP_ERROR(this->get_logger(), "next point to search is outofimage");
      search_for_lane = false;
      break;
    }
    // Search for the next segment within the specified range of angles
    bool next_segment_found = findLineInWindow(
      img, img2, start_of_lane, width, height, next_start_point,
      min_theta, max_theta);
    // Update the no_lane_found_counter value
    if (!next_segment_found) {
      RCLCPP_ERROR(this->get_logger(), " counter++");
      no_lane_found_counter++;
    } else {
      RCLCPP_ERROR(this->get_logger(), "reset counter");
      no_lane_found_counter = 0;
    }
    // Break the loop if the no_lane_found_counter value is greater than or equal to num
    if (no_lane_found_counter >= num) {
      search_for_lane = false;
      // Print for Debug
      // RCLCPP_ERROR(this->get_logger(), "loop should break as no_lane_found_counter= %d", no_lane_found_counter);
      break;
    }
    iteration++;
  }
}

// This function takes in the last detected segment and calculates the starting point for the next search window.
cv::Vec2i LaneDetectionNode::getNextStartPoint(
  std::tuple<cv::Vec2i, double> last_segment,
  int no_lane_found_counter, double split_angle, int width,
  int height)
{
  double scale = 1;
  int x_start = std::get<0>(last_segment)[0] - width / 2;
  int y_start = std::get<0>(last_segment)[1] - height / 2;
  double lane_angle = std::get<1>(last_segment);
  // If the angle of the last detected segment is pi, return a point to the left of the image.
  if (lane_angle == CV_PI) {
    return cv::Vec2i(x_start - (1 + no_lane_found_counter) * width, y_start);
  }
  // If the angle of the last detected segment is 0, return a point to the right of the image.
  if (lane_angle == 0) {
    return cv::Vec2i(x_start + (1 + no_lane_found_counter) * width, y_start);
  }
  // If the angle of the last detected segment is less than pi
  if (lane_angle < CV_PI) {
    // If the angle of the last detected segment is equal to the split angle, return a point to the top right of the image.
    if (lane_angle == split_angle) {
      return cv::Vec2i(
        x_start + (1 + no_lane_found_counter) * width,
        y_start - (1 + no_lane_found_counter) * height);
    }
    // If the angle of the last detected segment is less than the split angle, return a point to the right of the image.
    if (lane_angle < split_angle) {
      return cv::Vec2i(
        x_start + (1 + no_lane_found_counter) * width * scale,
        y_start - (1 + no_lane_found_counter) * tan(lane_angle) * width * scale);
    }
    // If the angle of the last detected segment is greater than the split angle, return a point to the top of the image.
    if (lane_angle > split_angle) {
      return cv::Vec2i(
        x_start + (1 + no_lane_found_counter) * height / tan(lane_angle) * scale,
        y_start - (1 + no_lane_found_counter) * height * scale);
    }
  }
  // If the angle of the last detected segment is greater than pi
  else {
    // If the angle of the last detected segment is equal to the split angle plus pi/2, return a point to the top left of the image.
    if (lane_angle == split_angle + CV_PI / 2) {
      return cv::Vec2i(
        x_start - (1 + no_lane_found_counter) * width,
        y_start - (1 + no_lane_found_counter) * height);
    }
    // If the angle of the last detected segment is less than the split angle plus pi/2, return a point to the top of the image.
    if (lane_angle < split_angle + CV_PI / 2) {
      return cv::Vec2i(
        x_start + (1 + no_lane_found_counter) * height / tan(lane_angle),
        y_start - (1 + no_lane_found_counter) * height);
    }
    // If the angle of the last detected segment is greater than the split angle plus pi/2, return a point to the top of the image.
    else {
      return cv::Vec2i(
        x_start - (1 + no_lane_found_counter) * width,
        y_start + (1 + no_lane_found_counter) * tan(lane_angle) * width);
    }
  }
  return cv::Vec2i(
    x_start - (1 + no_lane_found_counter) * width,
    y_start + (1 + no_lane_found_counter) * tan(lane_angle) * width);
}

// This function checks whether a given point is out of the image boundaries
bool LaneDetectionNode::pointOutOfImage(cv::Vec2i & point, int window_width, int window_height)
{
  if (point[0] >= 480 - window_width - 1) {
    point[0] = 480 - 2 - window_width;
  }
  if (point[0] < 0) {
    point[0] = 0;
  }
  if (point[1] < 0 or point[1] > 1280 - window_height) {
    return true;
  }
  return false;
}

// Calculate the angle between two points in radians
double LaneDetectionNode::calculateAngles(cv::Point start_point, cv::Point end_point)
{
  int x_start = start_point.x;
  int x_end = end_point.x;
  int y_start = start_point.y;
  int y_end = end_point.y;
  if (end_point.x > start_point.x and end_point.y > start_point.y) {
    x_end = start_point.x;
    x_start = end_point.x;
  }
  double distance = sqrt(pow((x_end - x_start), 2) + pow((y_end - y_start), 2));
  double angle = acos((x_end - x_start) / distance);
  if (angle > CV_PI) {
    angle -= CV_PI;
  }
  return angle;
}

// Return points on the lane based on angle and distance
cv::Point LaneDetectionNode::getPointFromAngleAndDistance(
  cv::Point point, double angle, double distance,
  bool right_line)
{
  if (right_line) {

    return cv::Point(
      point.x + distance * cos(angle + CV_PI / 2),
      point.y - distance * sin(angle + CV_PI / 2));
  } else {

    return cv::Point(
      point.x + distance * cos(angle - CV_PI / 2),
      point.y - distance * sin(angle - CV_PI / 2));
  }
}
