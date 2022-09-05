// Author of SSL_SLAM3: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#include "laserProcessingClass.h"

void LaserProcessingClass::init(std::string& file_path, std::string& yolo_path){
    //yolox
    network = ie.ReadNetwork(yolo_path);
    if (network.getOutputsInfo().size() != 1)
        throw std::logic_error("Sample supports topologies with 1 output only");
    if (network.getInputsInfo().size() != 1)
        throw std::logic_error("Sample supports topologies with 1 input only");
    input_info = network.getInputsInfo().begin()->second;
    input_name = network.getInputsInfo().begin()->first;
    if (network.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
    }
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);
    executable_network = ie.LoadNetwork(network, "CPU");
    infer_request = executable_network.CreateInferRequest();

    lidar_param.loadParam(file_path);
    double map_resolution = lidar_param.getLocalMapResolution();
//    edge_downsize_filter.setLeafSize(map_resolution/4.0, map_resolution/4.0, map_resolution/4.0);
//    plane_downsize_filter.setLeafSize(map_resolution/2.0, map_resolution/2.0, map_resolution/2.0);
    edge_downsize_filter.setLeafSize(0.01, 0.01, 0.01);
    surf_downsize_filter.setLeafSize(0.1, 0.1, 0.1);

    edge_noise_filter.setRadiusSearch(map_resolution);
    edge_noise_filter.setMinNeighborsInRadius(3);
    surf_noise_filter.setRadiusSearch(map_resolution);
    surf_noise_filter.setMinNeighborsInRadius(14);
    num_of_plane=0;
    num_of_line=0;
    gap_line = lidar_param.gap_line;
    gap_plane = lidar_param.gap_plane;
    gap_surf = lidar_param.gap_surf;
}

void LaserProcessingClass::yoloXDetect(const cv::Mat& bgr, std::vector<Object>& objects){
    cv::Mat pr_img = static_resize(bgr);
    Blob::Ptr imgBlob = infer_request.GetBlob(input_name);
    blobFromImage(pr_img, imgBlob);
    infer_request.Infer();
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
    if (!moutput) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                               "but by fact we were not able to cast output to MemoryBlob");
    }
    auto moutputHolder = moutput->rmap();
    const float* net_pred = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    float scale = std::min(INPUT_W / (bgr.cols*1.0), INPUT_H / (bgr.rows*1.0));
    decode_outputs(net_pred, objects, scale, img_w, img_h);
}

void LaserProcessingClass::featureExtraction(cv::Mat& color_im, cv::Mat& depth_im, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pc_out_line,
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& pc_out_plane, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_surf, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filter){
    std::vector<Object> objects;
    yoloXDetect(color_im,objects);
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
            if(z>lidar_param.max_distance||z<lidar_param.min_distance||isnan(z)){
                z=0.0;
                depth_im.at<float>(r,c)==0;}
            pcl::PointXYZRGB p;
            p.z = z;
            p.x = (c - lidar_param.camera_cx) * p.z / lidar_param.camera_fx;
            p.y = (r - lidar_param.camera_cy) * p.z / lidar_param.camera_fy;

            p.b = color_im.ptr<uchar>(r)[c*3];
            p.g = color_im.ptr<uchar>(r)[c*3+1];
            p.r = color_im.ptr<uchar>(r)[c*3+2];
            cloud_filter->push_back( p );

            pt_ptr[c][0] = p.x*1000.0;//m->mm
            pt_ptr[c][1] = p.y*1000.0;//m->mm
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }
    PlaneFitter pf;
    pf.minSupport = 300;
    pf.windowWidth = 12;
    pf.windowHeight = 12;
    pf.doRefine = true;

    cv::Mat seg(depth_im.rows, depth_im.cols, CV_8UC3);
    std::vector<std::vector<int>> vSeg;
    OrganizedImage3D Ixyz(cloud_peac);
    pf.run(&Ixyz, &vSeg, &seg);
    int gap_tmp = gap_plane;
    if(vSeg.size()<3){
        cout<<12<<endl;
//        pf.windowWidth = 8;
//        pf.windowHeight = 8;
//        pf.run(&Ixyz, &vSeg, &seg);
        gap_tmp = max(gap_tmp/1.2,1.0);
    }

    // pcl 拟合平面
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_all_plane(new pcl::PointCloud<pcl::PointXYZRGBL>);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr plane_info_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    int plane_cnt = 0;
    for(auto idx_plane = 0; idx_plane<vSeg.size();idx_plane++) {
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
        for (auto idx_idx = 0; idx_idx < vSeg[idx_plane].size(); idx_idx++) {
            pcl::PointXYZRGBL pt;
            int pt_idx = vSeg[idx_plane].at(idx_idx);
            int row = pt_idx / depth_im.cols;
            int col = pt_idx % depth_im.cols;
            if(row%gap_tmp==0 && col%gap_tmp==0)
            {
                pt.x = cloud_peac[row][col][0] / 1000.; //mm -> m
                pt.y = cloud_peac[row][col][1] / 1000.;
                pt.z = cloud_peac[row][col][2] / 1000.;

                pt.b = seg.ptr<uchar>(row)[col * 3];
                pt.g = seg.ptr<uchar>(row)[col * 3 + 1];
                pt.r = seg.ptr<uchar>(row)[col * 3 + 2];
                pt.label = num_of_plane;

                cloud->push_back(pt);
            }
        }
//        plane_downsize_filter.setInputCloud(cloud);
//        plane_downsize_filter.filter(*cloud);
        if(cloud->size()<4) {
            continue;
        }

        //--------------------------RANSAC拟合平面--------------------------
        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBL>::Ptr model_plane(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZRGBL>(cloud));
        pcl::RandomSampleConsensus<pcl::PointXYZRGBL> ransac(model_plane);
        ransac.setDistanceThreshold(0.02);    //设置距离阈值，与平面距离小于0.1的点作为内点
        ransac.computeModel();                //执行模型估计
        //-------------------------根据索引提取内点--------------------------
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBL>);
        std::vector<int> inliers;                //存储内点索引的容器
        ransac.getInliers(inliers);            //提取内点索引
        pcl::copyPointCloud<pcl::PointXYZRGBL>(*cloud, inliers, *cloud_plane);
//        if(inliers.size()/cloud->size()<0.3)
//            continue;
        *cloud_all_plane += *cloud_plane;

        //----------------------------输出模型参数---------------------------
        Eigen::VectorXf coefficient;
        ransac.getModelCoefficients(coefficient);
        if(coefficient[3]<0)
        {
            coefficient = -coefficient;
        }
        pcl::PointXYZRGBL plane_info;
        plane_info.x = coefficient[0];
        plane_info.y = coefficient[1];
        plane_info.z = coefficient[2];
        plane_info.data[3] = coefficient[3];
        plane_info.rgb = cloud_plane->size();
        plane_info.label = num_of_plane;
        plane_info_cloud->push_back(plane_info);

        num_of_plane++;
        plane_cnt++;
    }
    pcl::PointXYZRGBL plane_num;
    plane_num.x = static_cast<float>(plane_cnt);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr plane_num_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    plane_num_cloud->push_back(plane_num);
    *pc_out_plane = *plane_num_cloud + *plane_info_cloud + *cloud_all_plane;//num of plane + plane info + plane points

// line filter
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
    cvtColor(color_im, image_gray, cv::COLOR_BGR2GRAY);
    cv::Mat depth_pic_8u;
    depth_im.convertTo(depth_pic_8u,CV_8U,255./9000.);
    std::vector<cv::Vec4f> lines_fld;
    fld->detect(image_gray, lines_fld);
    int gap_line_tmp = gap_line;
    if(lines_fld.size()<10)
    {
        cout<<11<<endl;
//        length_threshold = 20;
//        distance_threshold = 8;
//        fld = cv::ximgproc::createFastLineDetector(length_threshold,
//                                                   distance_threshold, canny_th1, canny_th2, canny_aperture_size,
//                                                   do_merge);
        gap_line_tmp = max(gap_line_tmp/1.2,1.0);
        lines_fld.clear();
        fld->detect(depth_pic_8u, lines_fld);
    }

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_all_line(new pcl::PointCloud<pcl::PointXYZRGBL>);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr line_info_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    int line_cnt = 0;
    for( size_t i = 0; i < lines_fld.size(); i++ )
    {
        cv::Vec4i l = lines_fld[i];
        cv::LineIterator lit(depth_im, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]));//todo Note
        line(seg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,255,255), 3);
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
        int gap_tmp_tmp = gap_line_tmp;
        if(lit.count< gap_tmp_tmp*5)
            gap_tmp_tmp = gap_tmp_tmp/4;
        for(int i = 0; i < lit.count; ++i, ++lit)
        {
            if(i%gap_tmp_tmp==0) {
                int col = lit.pos().x;
                int row = lit.pos().y;
                pcl::PointXYZRGBL pt;
                pt.x = cloud_peac[row][col][0] / 1000.; //mm -> m
                pt.y = cloud_peac[row][col][1] / 1000.;
                pt.z = cloud_peac[row][col][2] / 1000.;
                if(pt.z==0.)
                    continue;
                pt.b = color_im.ptr<uchar>(row)[col * 3];
                pt.g = color_im.ptr<uchar>(row)[col * 3 + 1];
                pt.r = color_im.ptr<uchar>(row)[col * 3 + 2];
                pt.label = num_of_line;
//                pt.label = pt.z;
                cloud->push_back(pt);
            }
        }
//        edge_downsize_filter.setInputCloud(cloud);
//        edge_downsize_filter.filter(*cloud);
        if(cloud->size()<3)
            continue;
        num_of_line++;
        line_cnt++;
        //-----------------------------拟合直线-----------------------------
        pcl::SampleConsensusModelLine<pcl::PointXYZRGBL>::Ptr model_line(new pcl::SampleConsensusModelLine<pcl::PointXYZRGBL>(cloud));
        pcl::RandomSampleConsensus<pcl::PointXYZRGBL> ransac(model_line);
        ransac.setDistanceThreshold(11);	//内点到模型的最大距离
        ransac.setMaxIterations(1000);		//最大迭代次数
        ransac.computeModel();				//直线拟合
        //--------------------------根据索引提取内点------------------------
        vector<int> inliers;
        ransac.getInliers(inliers);
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZRGBL>);
        pcl::copyPointCloud<pcl::PointXYZRGBL>(*cloud, inliers, *cloud_line);
        *cloud_all_line += *cloud_line;
//        if(inliers.size()<5)
//            continue;

        //----------------------------输出模型参数--------------------------
        Eigen::VectorXf coef;
        ransac.getModelCoefficients(coef);
        pcl::PointXYZRGBL line_point_info;
        line_point_info.x = coef[0];
        line_point_info.y = coef[1];
        line_point_info.z = coef[2];
        line_point_info.label = cloud->size();
        line_point_info.rgb = cloud_line->size();
        line_info_cloud->push_back(line_point_info);

        pcl::PointXYZRGBL line_direction_info;
        line_direction_info.x = coef[3]; //nx
        line_direction_info.y = coef[4]; //ny
        line_direction_info.z = coef[5]; //nz
        line_info_cloud->push_back(line_direction_info);

        //project to line
        pcl::PointXYZRGBL endpoint1 = cloud->at(inliers[0]);
        pcl::PointXYZRGBL endpoint2 = cloud->at(inliers.back());
        Eigen::Vector3d point(coef[0],coef[1],coef[2]);
        Eigen::Vector3d direction(coef[3],coef[4],coef[5]);
        Eigen::Vector3d vect_end1_to_point(coef[0]-endpoint1.x,coef[1]-endpoint1.y,coef[2]-endpoint1.z);
        Eigen::Vector3d vect_end2_to_point(coef[0]-endpoint2.x,coef[1]-endpoint2.y,coef[2]-endpoint2.z);
        auto endpt1 = point-vect_end1_to_point.dot(direction)*direction;
        auto endpt2 = point-vect_end2_to_point.dot(direction)*direction;
        pcl::PointXYZRGBL line_endpoint1_info;
        line_endpoint1_info.x = endpt1(0);
        line_endpoint1_info.y = endpt1(1);
        line_endpoint1_info.z = endpt1(2);
        line_info_cloud->push_back(line_endpoint1_info);

        pcl::PointXYZRGBL line_endpoint2_info;
        line_endpoint2_info.x = endpt2(0);
        line_endpoint2_info.y = endpt2(1);
        line_endpoint2_info.z = endpt2(2);
        line_info_cloud->push_back(line_endpoint2_info);

    }
    pcl::PointXYZRGBL line_num;
    line_num.x = static_cast<float>(line_cnt);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr line_num_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    line_num_cloud->push_back(line_num);
    *pc_out_line = *line_num_cloud + *line_info_cloud + *cloud_all_line;
//    cout << "The number of lines is "<<line_cnt<<endl;
//    cout << "The number of lines points are "<<cloud_all_line->size()<<endl;

    for (int m = 0; m < depth_im.rows; m+=gap_surf){
        for (int n = 0; n < depth_im.cols; n+=gap_surf){
            if(seg.at<cv::Vec3b>(m,n)==cv::Vec3b(0,0,0)){
                pcl::PointXYZRGB pt;
                pt.x = cloud_peac[m][n][0] / 1000.; //mm -> m
                pt.y = cloud_peac[m][n][1] / 1000.;
                pt.z = cloud_peac[m][n][2] / 1000.;
                pt.b = color_im.ptr<uchar>(m)[n * 3];
                pt.g = color_im.ptr<uchar>(m)[n * 3 + 1];
                pt.r = color_im.ptr<uchar>(m)[n * 3 + 2];
                cloud_surf->points.push_back(pt);
            }
        }
    }
//    surf_noise_filter.setInputCloud(cloud_surf);
//    surf_noise_filter.filter(*cloud_surf);


}


Double2d::Double2d(int id_in, double value_in){
    id = id_in;
    value = value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in){
    layer = layer_in;
    time = time_in;
};
