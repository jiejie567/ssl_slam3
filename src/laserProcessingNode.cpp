// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

//c++ lib
#include <cmath>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>

//ros lib
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

//pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//local lib
#include "utils.h"
#include "param.h"
#include "laserProcessingClass.h"

LaserProcessingClass laserProcessing;
std::mutex mutex_lock;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudBuf;

ros::Publisher pubEdgePoints;
ros::Publisher pubSurfPoints;
ros::Publisher pubLaserCloudFiltered;

void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    mutex_lock.lock();
    pointCloudBuf.push(laserCloudMsg);
    mutex_lock.unlock();

}
void RGBDHandler(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{

    //read data
    laserProcessing.frame_count++;
//    if(laserProcessing.frame_count%4!=0)
//        return;
    cv_bridge::CvImagePtr color_ptr, depth_ptr;
    cv::Mat color_pic, depth_pic;
    color_ptr = cv_bridge::toCvCopy(msgRGB, sensor_msgs::image_encodings::BGR8);
    color_pic = color_ptr->image;
    depth_ptr = cv_bridge::toCvCopy(msgD, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_pic = depth_ptr->image;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
    ros::Time pointcloud_time = msgRGB->header.stamp;
    double ThMaxDepth = laserProcessing.lidar_param.max_distance * laserProcessing.lidar_param.camera_factor;
    double ThMinDepth = laserProcessing.lidar_param.min_distance * laserProcessing.lidar_param.camera_factor;
    for (int m = 0; m < depth_pic.rows; m++){
        for (int n = 0; n < depth_pic.cols; n++){
            if(depth_pic.ptr<float>(m)[n] <  ThMinDepth ||
            depth_pic.ptr<float>(m)[n] > ThMaxDepth)
                depth_pic.ptr<float>(m)[n] = 0.;//depth filter
            float d = depth_pic.ptr<float>(m)[n];//ushort d = depth_pic.ptr<ushort>(m)[n];
            if (d == 0.)
                continue;
            pcl::PointXYZRGB p;
            p.z = double(d) /  laserProcessing.lidar_param.camera_factor;
            p.x = (n - laserProcessing.lidar_param.camera_cx) * p.z / laserProcessing.lidar_param.camera_fx;
            p.y = (m - laserProcessing.lidar_param.camera_cy) * p.z / laserProcessing.lidar_param.camera_fy;

            p.b = color_pic.ptr<uchar>(m)[n*3];
            p.g = color_pic.ptr<uchar>(m)[n*3+1];
            p.r = color_pic.ptr<uchar>(m)[n*3+2];
            cloud->points.push_back( p );
        }
    }
//    cout<<1<<endl;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointcloud_edge(new pcl::PointCloud<pcl::PointXYZRGBL>());
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointcloud_surf(new pcl::PointCloud<pcl::PointXYZRGBL>());

    static TicToc timer("laser processing");
    timer.tic();
    laserProcessing.featureExtraction(color_pic,depth_pic, pointcloud_edge,pointcloud_surf);
    timer.toc(60);


    sensor_msgs::PointCloud2 laserCloudFilteredMsg;
    pcl::toROSMsg(*cloud, laserCloudFilteredMsg);
    laserCloudFilteredMsg.header.stamp = pointcloud_time;
    laserCloudFilteredMsg.header.frame_id = "camera_depth_optical_frame";
    pubLaserCloudFiltered.publish(laserCloudFilteredMsg);

    sensor_msgs::PointCloud2 edgePointsMsg;
    pcl::toROSMsg(*pointcloud_edge, edgePointsMsg);
    edgePointsMsg.header.stamp = pointcloud_time;
    edgePointsMsg.header.frame_id = "camera_depth_optical_frame";
    pubEdgePoints.publish(edgePointsMsg);

    sensor_msgs::PointCloud2 surfPointsMsg;
    pcl::toROSMsg(*pointcloud_surf, surfPointsMsg);
    surfPointsMsg.header.stamp = pointcloud_time;
    surfPointsMsg.header.frame_id = "camera_depth_optical_frame";
    pubSurfPoints.publish(surfPointsMsg);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    std::string file_path;
    nh.getParam("/file_path", file_path); 
    laserProcessing.init(file_path);

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&RGBDHandler, _1, _2));

    pubLaserCloudFiltered = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_filtered", 100);
    pubEdgePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_edge", 100);
    pubSurfPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100);

    ros::Rate loop_rate(40);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
