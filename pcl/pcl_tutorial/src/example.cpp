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
#include <velodyne_pointcloud/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>

ros::Publisher pub;
pcl::PointCloud<pcl::PointXYZ> cloudXYZ;

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

void filter_cb(const sensor_msgs::ImageConstPtr& msg)
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

  //int point_cnt = *cloudXYZ.width * cloudXYZ.height;
  pcl::PointCloud<pcl::PointXYZ>::Ptr p_ruts(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloudXYZ, *p_ruts);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  ROS_INFO("%d", cloudXYZ.height);
  for (int i = 0; i < cloudXYZ.size(); i++)
  {
    //calibration
    int rate = 85;//大きいほど範囲小さく,これを用いて画像のほうで幅を判定する？
    int point_img_x = (int)(cloudXYZ.points[i].y * rate);
    int point_img_y = (int)((cloudXYZ.points[i].x - 3.3) * rate);
    point_img_x += cv_ptr->image.cols / 2;
    point_img_x = cv_ptr->image.cols - point_img_x;//反転
    point_img_y = cv_ptr->image.rows - point_img_y;

    if (point_img_x < 0 || point_img_x >= cv_ptr->image.cols
    || point_img_y < 0 || point_img_y >= cv_ptr->image.rows)
      continue;

    /*if(cloudXYZ.points[i].z<-2.1)//z座標で単純にフィルター
    {
      inliers->indices.push_back(i);
    }*/

    if(cv_ptr->image.data[point_img_y*cv_ptr->image.step+point_img_x*cv_ptr->image.elemSize()]==255)
    {
      inliers->indices.push_back(i);
    }
  }
  extract.setInputCloud(p_ruts);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*p_ruts);

  pcl::PCLPointCloud2 cloud_out;
  pcl::toPCLPointCloud2(*p_ruts, cloud_out);
  // Convert to ROS data type
  sensor_msgs::PointCloud2 output;
  pcl_conversions::fromPCL(cloud_out, output);

  // Publish the data
  pub.publish(output);
}

int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "rut_calibration");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, cloud_cb);
  ros::Subscriber sub2 = nh.subscribe("/bird_eye_image", 1, filter_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_rut", 1);

  // Spin
  ros::spin();
}