// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#ifndef _LASER_PROCESSING_CLASS_H_
#define _LASER_PROCESSING_CLASS_H_
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h> // 拟合平面
#include <pcl/sample_consensus/sac_model_line.h> // 拟合直线

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "param.h"
#include <ros/ros.h>
#include "AHCPlaneFitter.hpp"
#include "yolox_openvino.h"

//#include "YoloDetect.h"

//points covariance class
class Double2d{
	public:
		int id;
		double value;
		Double2d(int id_in, double value_in);
};

//points info class
class PointsInfo{
	public:
		int layer;
		double time;
		PointsInfo(int layer_in, double time_in);
};

class LaserProcessingClass {
    public:
    	LaserProcessingClass(){};
		void init(std::string& file_path,std::string& yolo_path);
        void featureExtraction(cv::Mat& color_im,cv::Mat& depth_im, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pc_out_line,
                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pc_out_surf,pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_surf, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filter);
        void yoloXDetect(const cv::Mat& bgr,std::vector<Object>& objects);
        int frame_count;
        LidarParam lidar_param;
    private:
     	pcl::VoxelGrid<pcl::PointXYZRGBL> edge_downsize_filter;
     	pcl::VoxelGrid<pcl::PointXYZRGBL> surf_downsize_filter;
     	pcl::RadiusOutlierRemoval<pcl::PointXYZRGBL> edge_noise_filter;
     	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> surf_noise_filter;
        uint32_t num_of_plane;
        uint32_t num_of_line;
        int gap_plane;
        int gap_line;
        int gap_surf;
        //yolox
        Core ie;
        CNNNetwork network;
        ExecutableNetwork executable_network;
        InferRequest infer_request;
        InputInfo::Ptr input_info;
        std::string input_name;
        std::string output_name;

};

#endif // _LASER_PROCESSING_CLASS_H_

