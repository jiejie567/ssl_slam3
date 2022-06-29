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
    const double camera_factor = 1000;
    const double camera_cx = 333.891;
    const double camera_cy = 254.687;
    const double camera_fx = 597.53;
    const double camera_fy = 597.795;
    const float max_use_range = 9.0;
    const float min_use_range = 0.6;

    //read data
    laserProcessing.frame_count++;
    if(laserProcessing.frame_count%3!=0)
        return;
    cv_bridge::CvImagePtr color_ptr, depth_ptr;
    cv::Mat color_pic, depth_pic;
     color_ptr = cv_bridge::toCvCopy(msgRGB, sensor_msgs::image_encodings::BGR8);
    color_pic = color_ptr->image;
    depth_ptr = cv_bridge::toCvCopy(msgD, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_pic = depth_ptr->image;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
    ros::Time pointcloud_time = msgRGB->header.stamp;
    for (int m = 0; m < depth_pic.rows; m++){
        for (int n = 0; n < depth_pic.cols; n++){
            if(depth_pic.ptr<float>(m)[n] > max_use_range * camera_factor ||
            depth_pic.ptr<float>(m)[n] < min_use_range * camera_factor)
                depth_pic.ptr<float>(m)[n] = 0.;//depth filter
            float d = depth_pic.ptr<float>(m)[n];//ushort d = depth_pic.ptr<ushort>(m)[n];
            if (d == 0.)
                continue;
            pcl::PointXYZRGB p;

            p.z = double(d) / camera_factor;
    //            if(m==(int)(depth_pic.rows/2)&&n==(int)(depth_pic.cols/2))
    //                cout<<double(d)<<endl;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;


            p.b = color_pic.ptr<uchar>(m)[n*3];
            p.g = color_pic.ptr<uchar>(m)[n*3+1];
            p.r = color_pic.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back( p );
        }
    }
//    cout<<1<<endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_edge(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_surf(new pcl::PointCloud<pcl::PointXYZRGB>());

    static TicToc timer("laser processing");
    timer.tic();
    laserProcessing.featureExtraction(cloud, pointcloud_edge,pointcloud_surf);
    timer.toc(300);

    sensor_msgs::PointCloud2 laserCloudFilteredMsg;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    *pointcloud_filtered+=*pointcloud_edge;
    *pointcloud_filtered+=*pointcloud_surf;
    pcl::toROSMsg(*pointcloud_filtered, laserCloudFilteredMsg);
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
int frame_count =0;
void laser_processing(){
    while(1){
        if(!pointCloudBuf.empty()){
            //read data
            mutex_lock.lock();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::fromROSMsg(*pointCloudBuf.front(), *pointcloud_in);
            ros::Time pointcloud_time = (pointCloudBuf.front())->header.stamp;
            pointCloudBuf.pop();
            mutex_lock.unlock();
            frame_count++;
            if(frame_count%3!=0)
                continue;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_edge(new pcl::PointCloud<pcl::PointXYZRGB>());          
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_surf(new pcl::PointCloud<pcl::PointXYZRGB>());

            static TicToc timer("laser processing");
            timer.tic();
            laserProcessing.featureExtraction(pointcloud_in, pointcloud_edge,pointcloud_surf);
            timer.toc(300);

            sensor_msgs::PointCloud2 laserCloudFilteredMsg;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());  
            *pointcloud_filtered+=*pointcloud_edge;
            *pointcloud_filtered+=*pointcloud_surf;
            pcl::toROSMsg(*pointcloud_filtered, laserCloudFilteredMsg);
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
        //sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    std::string file_path;
    nh.getParam("/file_path", file_path); 
    laserProcessing.init(file_path);

//    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 100, velodyneHandler);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 100);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 100);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&RGBDHandler, _1, _2));

    pubLaserCloudFiltered = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_filtered", 100);
    pubEdgePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_edge", 100);
    pubSurfPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100);


//    std::thread laser_processing_process{laser_processing};

    ros::spin();

    return 0;
}
