#include <ros/ros.h>
#include <ros/console.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <math.h>

ros::Publisher pub;
image_transport::Publisher *pub_bird;
image_transport::Publisher *pub_velo;
pcl::PointCloud<pcl::PointXYZI> cloudXYZ;
int depthRate = 2000;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
  // Container for original & filtered data
  pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
  //pcl::PCLPointCloud2 cloud_filtered;

  // Convert to PCL data type
  pcl_conversions::toPCL(*cloud_msg, *cloud);

  /*
  // Perform the actual filtering
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud(cloudPtr);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-1.9, 1.9);
  pass.filter(cloud_filtered);
  */

  pcl::fromPCLPointCloud2(*cloudPtr, cloudXYZ);
  //ROS_INFO("%d",cloudXYZRI.)
}

void filter_cb(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  int width = cv_ptr->image.cols;
  int height = cv_ptr->image.rows;

  //int point_cnt = *cloudXYZ.width * cloudXYZ.height;
  pcl::PointCloud<pcl::PointXYZI>::Ptr p_ruts(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(cloudXYZ, *p_ruts);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

  sensor_msgs::Image img;
  img.header.frame_id = "velo_2d";
  img.height = 1000;
  img.width = 1000;
  img.encoding = "rgb8";
  img.step = img.width * 3;
  img.data.resize(img.height * img.step);
  img.header.seq = 0;
  img.header.stamp = ros::Time::now();

  sensor_msgs::Image img_3d2d;
  img_3d2d.header.frame_id = "3d_2d";
  img_3d2d.height = height;
  img_3d2d.width = width;
  img_3d2d.encoding = "rgb8";
  img_3d2d.step = img_3d2d.width * 3;
  img_3d2d.data.resize(img_3d2d.height * img_3d2d.step);
  img_3d2d.header.seq = 0;
  img_3d2d.header.stamp = ros::Time::now();

  int birdeyeUpW = 100;
  int birdeyeDownW = width - 150;
  int birdeyeUp = 190;
  int birdeyeHeight = height - birdeyeUp;
  double theta = 3.0 * M_PI / 180; //rad系
  double tan40 = tan(3.141592 * 40 / 180);
  double tan28 = tan(3.141592 * 28 / 180);

  for (int i = 1; i < cloudXYZ.size(); i++)
  {
    float x = cloudXYZ.points[i].x * cos(theta) + cloudXYZ.points[i].y * sin(theta) + 0.05 /*0.09*/;
    float y = -cloudXYZ.points[i].x * sin(theta) + cloudXYZ.points[i].y * cos(theta) /*+ 0.03*/;
    float z = cloudXYZ.points[i].z - 0.21;
    int point_img_x = width / 2 * (1 - y / tan40 / x);
    int point_img_y = height / 2 * (1 - z / tan28 / x);

    if (0 <= point_img_x && point_img_x < width && 0 <= point_img_y && point_img_y < height)
    {
      img_3d2d.data[point_img_x * 3 + point_img_y * img_3d2d.step] = 255;
      img_3d2d.data[point_img_x * 3 + point_img_y * img_3d2d.step + 1] = 255;
      img_3d2d.data[point_img_x * 3 + point_img_y * img_3d2d.step + 2] = 255;
    }

    int xLim = birdeyeUpW + (birdeyeDownW - birdeyeUpW) * (point_img_y - birdeyeUp) / birdeyeHeight;
    if (point_img_x < (width - xLim) / 2 || point_img_x >= (width + xLim) / 2 || point_img_y < birdeyeUp || point_img_y >= height)
    {
      continue;
    }

    inliers->indices.push_back(i);
  }

  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(p_ruts);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*p_ruts);

  //ransac plane detection
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr sacInliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.05);

  seg.setInputCloud(p_ruts);
  seg.segment(*sacInliers, *coefficients);
  extract.setIndices(sacInliers);
  extract.filter(*p_ruts);

  //get 4 points
  cv::Point2f *ptsUV = new cv::Point2f[4];
  cv::Point2f *ptsXY = new cv::Point2f[4];
  cv::Point2f *vanishXY = new cv::Point2f[4]{cv::Point2f(0, 100), cv::Point2f(0, -100),
                                             cv::Point2f(100, 100 * 0.707), cv::Point2f(100, -100 * 0.707)};
  bool onFirst = true;
  for (int i = 0; i < p_ruts->size(); i++)
  {
    float x = p_ruts->points[i].x * cos(theta) + p_ruts->points[i].y * sin(theta) + 0.05 /*0.09*/;
    float y = -p_ruts->points[i].x * sin(theta) + p_ruts->points[i].y * cos(theta) /*+ 0.03*/;
    float z = p_ruts->points[i].z - 0.21;
    int point_img_x = width / 2 * (1 - y / tan40 / x);
    int point_img_y = height / 2 * (1 - z / tan28 / x);

    int xIm = img.width / 2 - y * img.width / 20;
    int yIm = img.height * 9 / 10 - x * img.height / 20;
    //ここで範囲外だと強制終了
    /*
    if (0 <= xIm && xIm < img.width && 0 <= yIm && yIm < img.height)
    {
      img.data[xIm * 3 + yIm * img.step + 2] = 255;
    }
    */

    if (onFirst)
    {
      for (int j = 0; j < 4; j++)
      {
        ptsUV[j] = cv::Point2f(point_img_x, point_img_y);
        ptsXY[j] = cv::Point2f(x, y);
      }
      onFirst = false;
    }
    else
    {
      for (int j = 0; j < 4; j++)
      {
        double distTemp = pow(vanishXY[j].x - x, 2) + pow(vanishXY[j].y - y, 2);
        double dist = pow(vanishXY[j].x - ptsXY[j].x, 2) + pow(vanishXY[j].y - ptsXY[j].y, 2);
        if (distTemp < dist)
        {
          ptsUV[j] = cv::Point2f(point_img_x, point_img_y);
          ptsXY[j] = cv::Point2f(x, y);
        }
      }
    }
  }
  for (int i = 0; i < 4; i++)
  {
    /*
    img.data[ptsUV[i].x * 3 + ptsUV[i].y * width * 3] = 255;
    img.data[ptsUV[i].x * 3 + ptsUV[i].y * width * 3 + 1] = 255;
    img.data[ptsUV[i].x * 3 + ptsUV[i].y * width * 3 + 2] = 255;
    */
    float xx = ptsXY[i].x * img.height / 20;
    float yy = ptsXY[i].y * img.width / 20;
    ptsXY[i].x = img.width / 2 - yy;
    ptsXY[i].y = img.height * 9 / 10 - xx;
    ptsUV[i].y -= birdeyeUp;
  }

  cv::Mat mat = cv::getPerspectiveTransform(ptsUV, ptsXY);
  cv::Mat srcImage(cv_ptr->image, cv::Rect(0, birdeyeUp, width, birdeyeHeight));
  cv::Mat outImage(cv::Size(img.width, img.height), CV_8UC3);
  cv::warpPerspective(srcImage, outImage, mat, cv::Size(img.width, img.height));

  pcl::PCLPointCloud2 cloud_out;
  pcl::toPCLPointCloud2(*p_ruts, cloud_out);
  // Convert to ROS data type
  sensor_msgs::PointCloud2 output;
  pcl_conversions::fromPCL(cloud_out, output);

  // Publish the data
  pub.publish(output);
  pub_bird->publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", outImage).toImageMsg());
  pub_velo->publish(img_3d2d);
}

int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "rut_calibration");
  ros::NodeHandle n("~");
  n.getParam("depthRate", depthRate);

  ros::NodeHandle nh;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, cloud_cb);
  ros::Subscriber sub2 = nh.subscribe("/thermal_mono", 1, filter_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_rut", 1);
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub_birdeye = it.advertise("birdeye", 1);
  pub_bird = &pub_birdeye;
  image_transport::Publisher pub_velo2d = it.advertise("velo_2d", 1);
  pub_velo = &pub_velo2d;

  // Spin
  ros::spin();
}