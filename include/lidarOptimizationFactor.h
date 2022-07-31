// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _LIDAR_OPTIMIZATION_FACTOR_H_
#define _LIDAR_OPTIMIZATION_FACTOR_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "utils.h"
#include <ros/ros.h>
class LidarOdometryFactor : public ceres::SizedCostFunction<6, 15, 15>{
public:
    LidarOdometryFactor(Eigen::Isometry3d odom_in, Eigen::Matrix<double, 6, 1> covariance_in);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    Eigen::Isometry3d odom;
    Eigen::Matrix<double, 6, 6> sqrt_info;
};

class LidarEdgeFactor : public ceres::SizedCostFunction<1, 15> {
public:
	LidarEdgeFactor(Eigen::Vector3d curr_point_in, Eigen::Vector3d last_point_a_in, Eigen::Vector3d last_point_b_in, double covariance_in);
	virtual ~LidarEdgeFactor() {}
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

	Eigen::Vector3d curr_point;
	Eigen::Vector3d last_point_a;
	Eigen::Vector3d last_point_b;
	double sqrt_info;
};

class LidarPlaneFactor : public ceres::SizedCostFunction<3, 15> {
public:
    LidarPlaneFactor(Eigen::Vector4d current_plane_hessian, Eigen::Vector4d traget_plane_hessian, int quantity_plane_matched, double covariance_in);
    virtual ~LidarPlaneFactor() {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector4d current_plane_hessian_;
    Eigen::Vector4d target_plane_hessian_;
    double sqrt_info;
    int quantity_plane_matched_;
};


#endif // _LIDAR_OPTIMIZATION_FACTOR_H_

