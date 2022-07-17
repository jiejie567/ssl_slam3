// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#include "laserProcessingClass.h"

void LaserProcessingClass::init(std::string& file_path){
    lidar_param.loadParam(file_path);
    double map_resolution = lidar_param.getLocalMapResolution();
//    edge_downsize_filter.setLeafSize(map_resolution/4.0, map_resolution/4.0, map_resolution/4.0);
//    surf_downsize_filter.setLeafSize(map_resolution/2.0, map_resolution/2.0, map_resolution/2.0);
    edge_downsize_filter.setLeafSize(0.01, 0.01, 0.01);
    surf_downsize_filter.setLeafSize(0.15, 0.15, 0.15);
    
    edge_noise_filter.setRadiusSearch(map_resolution);
    edge_noise_filter.setMinNeighborsInRadius(3);
    surf_noise_filter.setRadiusSearch(map_resolution);
    surf_noise_filter.setMinNeighborsInRadius(14);

}


void LaserProcessingClass::featureExtraction(cv::Mat& color_im, cv::Mat& depth_im, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_out_edge, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_out_surf){

// plane filter
    struct OrganizedImage3D {
        const cv::Mat_<cv::Vec3f>& cloud_peac;
        //note: ahc::PlaneFitter assumes mm as unit!!!
        OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c): cloud_peac(c) {}
        inline int width() const { return cloud_peac.cols; }
        inline int height() const { return cloud_peac.rows; }
        inline bool get(const int row, const int col, double& x, double& y, double& z) const {
            const cv::Vec3f& p = cloud_peac.at<cv::Vec3f>(row,col);
            x = p[0];
            y = p[1];
            z = p[2];
            return z > 0 && isnan(z)==0; //return false if current depth is NaN
        }
    };
    typedef ahc::PlaneFitter< OrganizedImage3D > PlaneFitter;

    cv::Mat_<cv::Vec3f> cloud_peac(depth_im.rows, depth_im.cols);
    for(int r=0; r<depth_im.rows; r++)
    {
        const float* depth_ptr = depth_im.ptr<float>(r);
        cv::Vec3f* pt_ptr = cloud_peac.ptr<cv::Vec3f>(r);
        for(int c=0; c<depth_im.cols; c++)
        {
            float z = (float)depth_ptr[c]/lidar_param.camera_factor;
            if(z>lidar_param.max_distance||z<lidar_param.min_distance||isnan(z)){z=0.0;}
            pt_ptr[c][0] = (c-lidar_param.camera_cx)/lidar_param.camera_fx*z*1000.0;//m->mm
            pt_ptr[c][1] = (r-lidar_param.camera_cy)/lidar_param.camera_fy*z*1000.0;//m->mm
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }
    PlaneFitter pf;
    pf.minSupport = 600;
    pf.windowWidth = 12;
    pf.windowHeight = 12;
    pf.doRefine = true;

    cv::Mat seg(depth_im.rows, depth_im.cols, CV_8UC3);
    std::vector<std::vector<int>> vSeg;
    OrganizedImage3D Ixyz(cloud_peac);
    pf.run(&Ixyz, &vSeg, &seg);

    // pcl 拟合平面
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(auto idx_plane = 0; idx_plane<vSeg.size();idx_plane++) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//        viewer.reset();
//        viewer->removeAllPointClouds();
        for (auto idx_idx = 0; idx_idx < vSeg[idx_plane].size(); idx_idx++) {
            pcl::PointXYZRGB pt;
            int pt_idx = vSeg[idx_plane].at(idx_idx);
            int row = pt_idx / depth_im.cols;
            int col = pt_idx % depth_im.cols;
            pt.x = cloud_peac[row][col][0] / 1000.; //mm -> m
            pt.y = cloud_peac[row][col][1] / 1000.;
            pt.z = cloud_peac[row][col][2] / 1000.;
            if(isnan(pt.z))
                continue;
            pt.b = seg.ptr<uchar>(row)[col * 3];
            pt.g = seg.ptr<uchar>(row)[col * 3 + 1];
            pt.r = seg.ptr<uchar>(row)[col * 3 + 2];
            cloud->push_back(pt);
        }

        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);
        sor.filter(*cloud);
        //--------------------------RANSAC拟合平面--------------------------
        pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_plane(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>(cloud));
        pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_plane);
        ransac.setDistanceThreshold(0.01);    //设置距离阈值，与平面距离小于0.1的点作为内点
        ransac.computeModel();                //执行模型估计
        //-------------------------根据索引提取内点--------------------------
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<int> inliers;                //存储内点索引的容器
        ransac.getInliers(inliers);            //提取内点索引
        pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *cloud_plane);
        *cloud_all_plane += *cloud_plane;
        //----------------------------输出模型参数---------------------------
        Eigen::VectorXf coefficient;
        ransac.getModelCoefficients(coefficient);
//        cout << "平面方程为：\n" << coefficient[0] << "x + " << coefficient[1] << "y + " << coefficient[2] << "z + "
//             << coefficient[3] << " = 0" << endl;
    }
// line filter
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all_line(new pcl::PointCloud<pcl::PointXYZRGB>);
    int length_threshold = 40;
    int distance_threshold = 4;
    int canny_th1 = 50;
    int canny_th2 = 50;
    int canny_aperture_size = 3;
    bool do_merge = true;

    cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(length_threshold,
                                                       distance_threshold, canny_th1, canny_th2, canny_aperture_size,
                                                       do_merge);
    cv::Mat image_gray(seg.rows, seg.cols, CV_8U);
    cvtColor(seg, image_gray, cv::COLOR_BGR2GRAY);
    cv::Mat depth_pic_8u;
    depth_im.convertTo(depth_pic_8u,CV_8U,255./9000.);
    std::vector<cv::Vec4f> lines_fld;
    fld->detect(depth_pic_8u, lines_fld);

    int line_cnt = 0;
    for( size_t i = 0; i < lines_fld.size(); i++ )
    {
        cv::Vec4i l = lines_fld[i];
        cv::LineIterator lit(depth_im, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]));//todo Note
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(int i = 0; i < lit.count; ++i, ++lit)
        {
            int col = lit.pos().x;
            int row = lit.pos().y;
            pcl::PointXYZRGB pt;
            pt.x = cloud_peac[row][col][0]/1000.; //mm -> m
            pt.y = cloud_peac[row][col][1]/1000.;
            pt.z = cloud_peac[row][col][2]/1000.;
            if(pt.z == 0.0)
                continue;
            pt.b = color_im.ptr<uchar>(row)[col*3];
            pt.g = color_im.ptr<uchar>(row)[col*3+1];
            pt.r = color_im.ptr<uchar>(row)[col*3+2];
            cloud->push_back(pt);
        }
        if(cloud->size()<4)
            continue;

        //-----------------------------拟合直线-----------------------------
        pcl::SampleConsensusModelLine<pcl::PointXYZRGB>::Ptr model_line(new pcl::SampleConsensusModelLine<pcl::PointXYZRGB>(cloud));
        pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_line);
        ransac.setDistanceThreshold(0.005);	//内点到模型的最大距离
//        ransac.setDistanceThreshold(0.1);	//内点到模型的最大距离

        ransac.setMaxIterations(1000);		//最大迭代次数
        ransac.computeModel();				//直线拟合

        //--------------------------根据索引提取内点------------------------
        vector<int> inliers;
        ransac.getInliers(inliers);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *cloud_line);
//        if(inliers.size()<5)
//            continue;
        line_cnt++;
        //----------------------------输出模型参数--------------------------
//        Eigen::VectorXf coef;
//        ransac.getModelCoefficients(coef);
//        cout << "直线方程为：\n"
//             << "   (x - " << coef[0] << ") / " << coef[3]
//             << " = (y - " << coef[1] << ") / " << coef[4]
//             << " = (z - " << coef[2] << ") / " << coef[5] << endl;
        *cloud_all_line += *cloud_line;
    }
//    cout << "The number of lines is "<<line_cnt<<endl;
//    cout << "The number of lines points are "<<cloud_all_line->size()<<endl;




    // reduce cloud size
    edge_downsize_filter.setInputCloud(cloud_all_line);
    edge_downsize_filter.filter(*pc_out_edge);
    surf_downsize_filter.setInputCloud(cloud_all_plane);
    surf_downsize_filter.filter(*pc_out_surf);

}


Double2d::Double2d(int id_in, double value_in){
    id = id_in;
    value = value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in){
    layer = layer_in;
    time = time_in;
};
