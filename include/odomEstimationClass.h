// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#ifndef _ODOM_ESTIMATION_CLASS_H_
#define _ODOM_ESTIMATION_CLASS_H_
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

#include <ros/ros.h>

// local lib
#include "utils.h"
#include "param.h"
#include "imuOptimizationFactor.h"
#include "lidarOptimizationFactor.h"
#include "poseOptimizationFactor.h"

#define POSE_BUFFER 50
class OdomEstimationClass{
	// pose_k*******************************************************pose_k+1
	// pose_q_arr[k]****imu_integrator_arr[k]*lidar_odom_arr[k]*****pose_q_arr[k+1]
	public:
		bool is_initialized;
		std::vector<ImuPreintegrationClass> imu_preintegrator_arr;
		std::vector<Eigen::Isometry3d> lidar_odom_arr;
		std::vector<Eigen::Vector3d> pose_r_arr; //world coordinate
		std::vector<Eigen::Vector3d> pose_t_arr; //world coordinate
		std::vector<Eigen::Vector3d> pose_v_arr; //world coordinate
		std::vector<Eigen::Vector3d> pose_b_a_arr; //imu coordinate
		std::vector<Eigen::Vector3d> pose_b_g_arr; //imu coordinate
		OdomEstimationClass();
		void init(std::string& file_path);
		bool initialize(void);
		void initMapWithPoints(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr edge_in, const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr surf_in);
		void addImuPreintegration(std::vector<double> dt_arr, std::vector<Eigen::Vector3d> acc_arr, std::vector<Eigen::Vector3d> gyr_arr);
		void addLidarFeature(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr edge_in, const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr surf_in);
		void optimize(void);

	private:
		CommonParam common_param;
		LidarParam lidar_param;
		ImuParam imu_param;

		// map points
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr edge_map;
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr surf_map;
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr current_edge_points;
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr current_surf_points;


        //plane
        int current_plane_num;
		std::vector<Eigen::Vector4d> *pv_plane_info;
        std::vector<double> *pv_plane_believe_rate;
        std::vector<Eigen::Vector4d> v_current_plane_info;
        std::vector<int> v_current_plane_points_num;
        std::vector<double> v_current_plane_believe_rate;
        //line
        int current_line_num;
        std::vector<Eigen::Vector4d> *pv_line_point_info;
        std::vector<Eigen::Vector4d> *pv_line_direction_info;
        std::vector<double> *pv_line_believe_rate;
        std::vector<Eigen::Vector4d> v_current_line_point_info;
        std::vector<Eigen::Vector4d> v_current_line_direction_info;
        std::vector<Eigen::Vector4d> v_current_line_endpoint1;
        std::vector<Eigen::Vector4d> v_current_line_endpoint2;
        std::vector<int> v_current_line_points_num;
        std::vector<double> v_current_line_believe_rate;


    // kdtree for fast indexing
		pcl::KdTreeFLANN<pcl::PointXYZRGBL> edge_kd_tree;
		pcl::KdTreeFLANN<pcl::PointXYZRGBL> surf_kd_tree;

		//pose
        Eigen::Isometry3d last_pose;
        Eigen::Isometry3d current_pose;



    // points downsampling before add to map
		pcl::VoxelGrid<pcl::PointXYZRGBL> edge_downsize_filter;
		pcl::VoxelGrid<pcl::PointXYZRGBL> surf_downsize_filter;
		
		void addEdgeCost(ceres::Problem& problem, ceres::LossFunction *loss_function, double* pose);
		void addSurfCost(ceres::Problem& problem, ceres::LossFunction *loss_function, double* pose);
		void addOdometryCost(const Eigen::Isometry3d& odom, ceres::Problem& problem, ceres::LossFunction *loss_function, double* pose1, double* pose2);
		void addImuCost(ImuPreintegrationClass& imu_integrator, ceres::Problem& problem, ceres::LossFunction *loss_function, double* pose1, double* pose2);
		void updateLocalMap(Eigen::Isometry3d& transform);
};
#endif // _ODOM_ESTIMATION_CLASS_H_

