/**
 * @file lane_detection_node.hpp
 * @brief this class is the lane detection for the car
 * @author PSAF
 * @date 2022-06-01
 */
#ifndef PSAF_LANE_DETECTION__LANE_DETECTION_NODE_HPP_
#define PSAF_LANE_DETECTION__LANE_DETECTION_NODE_HPP_

#include <string>
#include <vector>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/core/cvdef.h"

#include "libpsaf/interface/lane_detection_interface.hpp"
#include "psaf_configuration/configuration.hpp"
#include "libpsaf_msgs/msg/status_info.hpp"
#include <sstream>
#include <cmath>


/**
 * @class LaneDetectionNode
 * @implements LaneDetectionInterface
 * @brief The Lane detection for the car
 * @details This class is the node implementation of the lane detection.
 * It has 3 tasks:
 * 1. Calculate the position of the lane markings (left, center, right) in the image.
 * 2. Detect the start line (this is only necessary in discipline one of the carolo cup)
 * 3. Detect stop lines (this is only necessary in discipline two of the carolo cup)
 */
class LaneDetectionNode : public libpsaf::LaneDetectionInterface
{
public:
  LaneDetectionNode();

  /**
   * Use this flag to swap to a secondary algorithm at runtime. Use the following command
   * @code ros2 param set /lane_detection use_secondary_algorithm true _ @endcode
   */
  bool use_secondary_algorithm_;

  /**
   * This method is used to publish the results. It is called periodically by the main loop.
   * Call the publishers in this method.
   */
  void update();

  /**
   * This method returns the last received state of the car.
   * If no state was received, it returns -1.
   * @return the last received state
   */
  int getCurrentState();

  /**
   * This method returns the last received image. It´s used for debugging purposes.
   * @return the last received image
   */
  cv::Mat getLastImage();

  /**
   * Homography parameters for the psaf 1 car
   * @todo replace with your own homography
   */
  double homography_data_psaf1[9] = {
    -5.945220678747534, -8.221731543083614, 2029.2412091750095, 2.159328469836063,
    -57.07733531018901, 5685.467602003643, 0.0016374458883317959, -0.036883354445061224,
    0.9999999999999999};

  /**
   * The homography parameters for the psaf 2 car
   * @todo replace with your own homography
   */
  double homography_data_psaf2[9] = {

    -5.945220678747534, -8.221731543083614, 2029.2412091750095, 2.159328469836063,
    -57.07733531018901, 5685.467602003643, 0.0016374458883317959, -0.036883354445061224,
    0.9999999999999999};

protected:
  // Variable to store the last received image
  cv::Mat last_image_;
  // Variable to store the last calculated lane markings
  std::vector<std::vector<cv::Point>> last_lane_markings_positions_;
  // Variable for the current driving side
  volatile int side_{-1};
  // Variable to store the information if overtaking is prohibited
  volatile bool no_overtaking_{false};
  // Variable to store the information if there is a blocked off area
  volatile bool has_blocked_area_{false};
  // Variable to store the status info to be sent next
  volatile int status_info_{127};
  // Variable to store the information if a stop line was detected
  bool stop_line_found_{false};
  // Variable to store the position of the stop line
  cv::Point last_stop_line_position_{-1, -1};
  // Type of the stop line
  volatile unsigned int stop_line_type_{0};
  // Variable to store the current state of the state_machine
  volatile int current_state_{-1};


  /**
   * @brief Callback method for the state of the state machine
   * @param[in] state the state of the state machine
   */
  void updateState(std_msgs::msg::Int64::SharedPtr state) override;

  /**
   * @brief Callback method for the image
   * @param[in] img the camera image
   * @param[in] sensor: the position of the topic name in the topic vector
   * see configuration.hpp for more details
   */
  void processImage(cv::Mat & img, int sensor) final;


  /**
  * Grayscale image
  * @param[in] img a color image
  * @param[out] result the grayscale image
  */
  void grayscaleImage(cv::Mat & img, cv::Mat & result);

  /**
   * @brief convert the input grayscale image into a binary image
   * @details this method creates a grayscale image into a binary image. Every pixel inside the
   * lower and upper threshold is set to 255, every pixel outside the threshold is set to 0.
   * @param[in] img the input grayscale image
   * @param[out] result the ouput binary image
   * @param[in] threshold_low the lower bound of the threshold. Default: 127
   * @param[in] threshold_high the upper bound of the threshold. Default: 255
   */
  void binarizeImage(
    cv::Mat & img, cv::Mat & result, int threshold_low = 127,
    int threshold_high = 255);

  /**
  * Transform an image, i.e birdseye view
  * @param[in] img the image to be transformed
  * @param[in] homography the homography matrix
  * @param[out] result the transformed image
  */
  void transformImage(cv::Mat & img, cv::Mat & homography, cv::Mat & result);

  /**
   * Resize a given image to 640x480 Pixels
   * @param[in] image the image to be resized
   * @param[out] result the image where the result will be stored in
   */
  void resizeImage(cv::Mat & image, cv::Mat & result);

  /**
   * Extract the lane markings from the image
   * The results is a vector of vectors. The inner vectors contain the points of the lane markings
   * They should be left, center, right. If a certain lane is not detected, the vector will be empty
   * The inner vectors are allowed to have different sizes.
   * @param[in] img the image to be processed. This should already be the preprocessed image, i.e binarized
   */
  void extractLaneMarkings(cv::Mat & img);

  /**
   * @brief A secondary version of the lane detection. You can swap between the used methods via
   * parameters
   * @param[in] img the image to be processed.
   */
  void extractLaneMarkingsSecondary(cv::Mat & img);

  /**
   * Method to extract the stop line from the image
   * @param[in] img an input image, probably the binary image
   */
  void extractStopLine(cv::Mat & img);

  /**
   * Method to apply sliding window in order to to detect curves with hough transform
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] hlines vector for storing all found lines
   * @param[in] height_prop number of segments the height of the image is divided to
   * @param[in] widthProp number of segments the widht of the image is divided to
   */
  void slidingWindowHoughTransform(
    cv::Mat & img, std::vector<cv::Vec4i> & hlines, int height_prop,
    int width_prop);

  /**
   * Method that find a line in thw
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] img2 clone of img
   * @param[in] lane vector with lane segments
   * @param[in] width width of window in which the method looks for a line, here 200
   * @param[in] height height of window in which the method looks for a line, here 65
   * @param[in] min_theta minimum of theta angle
   * @param[in] max_theta maximum of theta angle
   * @return true if found a line
   */
  bool findLineInWindow(
    cv::Mat & img, cv::Mat & img2, std::vector<std::tuple<cv::Vec2i,
    double>> & lane, int width, int height, cv::Vec2i start_point,
    double min_theta = 0, double max_theta = CV_PI);

  /**
   * Method that search for one coherent lane (left, right or middle)
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] img2 clone of img
   * @param[in] lanes vector containing lanes
   */
  void findLane(
    cv::Mat & img, cv::Mat & img2, std::vector<std::vector<std::tuple<cv::Vec2i, double>>> & lanes);

  /**
   * Method that Find the right lane in the input image using a sliding window
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] img2 clone of img
   * @param[in] right_lane vector with lane segments of the right lane
   * @param[in] width width of window in which the method looks for a line, here 200
   * @param[in] height height of window in which the method looks for a line, here 65
   * @param[in] search_step number of sliding windows
   * @return true if found a line
   */
  bool findRightLane(
    cv::Mat & img, cv::Mat & img2, std::vector<std::tuple<cv::Vec2i,
    double>> & right_lane, int width, int height, int search_step);

  /**
   * Method that Find the light lane in the input image using a sliding window
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] img2 clone of img
   * @param[in] right_lane vector with lane segments of the left lane
   * @param[in] width width of window in which the method looks for a line, here 200
   * @param[in] height height of window in which the method looks for a line, here 65
   * @param[in] search_step number of sliding windows
   * @return true if found a line
   */
  bool findLeftLane(
    cv::Mat & img, cv::Mat & img2, std::vector<std::tuple<cv::Vec2i,
    double>> & left_lane, int width, int height, int search_step);

  /**
   * Method that finds the rest oft the lane in the input image using a sliding window
   * @param[in] img input image that contains the edges that should be recognized as lines
   * @param[in] img2 clone of img
   * @param[in] start_of_lane vector with lane segments of the beginning of the lane
   * @param[in] width width of window in which the method looks for a line, here 200
   * @param[in] height height of window in which the method looks for a line, here 65
   * @param[in] num number which defines how long to search for the rest of the lane
   */
  void findRestOfLane(
    cv::Mat & img, cv::Mat & img2, std::vector<std::tuple<cv::Vec2i,
    double>> & start_of_lane, int width, int height, int num);

  /**
   * Method that calculates the next start point for the window search
   * @param[in] last_segment vector containing the last lane segment
   * @param[in] no_lane_found_counter counts how long no lane is found
   * @param[in] width width of window in which the method looks for a line, here 200
   * @param[in] height height of window in which the method looks for a line, here 65
   * @param[out] point start_point for the sliding window search
   */
  cv::Vec2i getNextStartPoint(
    std::tuple<cv::Vec2i, double> last_segment, int no_lane_found_counter,
    double split_angle, int width, int height);

  /**
   * Method that checks if point is out of image
   * @param point point that has to be checked
   * @param window_width width of window
   * @param window_height height of window
   * @return true if point is out of image
   */
  bool pointOutOfImage(cv::Vec2i & point, int window_width, int window_height);

  /**
   * Method that calculates the angle between two points
   * @param start_point first point
   * @param end_point second point
   * @return calculated angle between start_point and end_point
   */
  double calculateAngles(cv::Point start_point, cv::Point end_point);

  /**
   * Method that calculate a point from angle and distance
   * @param point start point of the calculation
   * @param angle angle at which the point is to be moved
   * @param distance how far the point is to be moved at the specified angle
   * @param right_line if point is on the right line project it to the left otherwise to the right
   * @return calculated point with should be on the middle lane
   */
  cv::Point getPointFromAngleAndDistance(
    cv::Point point, double angle, double distance,
    bool right_line);

private:
};

#endif  // PSAF_LANE_DETECTION__LANE_DETECTION_NODE_HPP_
