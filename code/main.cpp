/*****************************************************************************
*   3D Reconstruction
******************************************************************************
*   by Yongfu Hao
*   https://www.linkedin.com/in/yongfu-hao-90252b35/
*****************************************************************************/
#pragma once
#pragma warning(disable: 4244 18 4996 4800)
#define _SCL_SECURE_NO_WARNINGS

#define DEFINE_GLOBALS
#include <iostream>
#include <string.h>
#include "Visualization.h"
#include "Common.h"
using namespace std;

//visualization
class VisualizerListener : public SfMUpdateListener {
public:
	void update(std::vector<cv::Point3d> pcld,
		std::vector<cv::Vec3b> pcldrgb,
		std::vector<cv::Point3d> pcld_alternate,
		std::vector<cv::Vec3b> pcldrgb_alternate,
		std::vector<cv::Matx34d> cameras) {
		ShowClouds(pcld, pcldrgb, pcld_alternate, pcldrgb_alternate);

		vector<cv::Matx34d> v = cameras;
		for (unsigned int i = 0; i < v.size(); i++) {
			stringstream ss; ss << "camera" << i;
			cv::Matx33f R;
			R(0, 0) = v[i](0, 0); R(0, 1) = v[i](0, 1); R(0, 2) = v[i](0, 2);
			R(1, 0) = v[i](1, 0); R(1, 1) = v[i](1, 1); R(1, 2) = v[i](1, 2);
			R(2, 0) = v[i](2, 0); R(2, 1) = v[i](2, 1); R(2, 2) = v[i](2, 2);
			visualizerShowCamera(R, cv::Vec3f(v[i](0, 3), v[i](1, 3), v[i](2, 3)), 255, 0, 0, 0.2, ss.str());
		}
	}
};


int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " <image directory> [downscale_factor=0.5]" << endl;
		return 0;
	}
	string image_dir = argv[1];
	float downscale_factor = 0.5;
	if (argc >= 3)
	{
		downscale_factor = atof(argv[2]);
	}
	read_images_and_calibration_matrix(image_dir, downscale_factor);
	if (images.size() == 0) {
		cerr << "can't get image files" << endl;
		return 1;
	}

	cv::Ptr<VisualizerListener> visualizerListener = new VisualizerListener; //with ref-count
	attach(visualizerListener);
	RunVisualizationThread();

	optical_flow_feature_match();
	PruneMatchesBasedOnF();
	GetBaseLineTriangulation();
	AdjustCurrentBundle();
	update();
	RecoverCamerasIncremental();

	//get the scale of the result cloud using PCA
	double scale_cameras_down = 1.0;
	{
		vector<cv::Point3d> cld = getPointCloud();
		if (cld.size() == 0) cld = getPointCloudBeforeBA();
		cv::Mat_<double> cldm(cld.size(), 3);
		for (unsigned int i = 0; i < cld.size(); i++) {
			cldm.row(i)(0) = cld[i].x;
			cldm.row(i)(1) = cld[i].y;
			cldm.row(i)(2) = cld[i].z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm, mean, 0);//CV_PCA_DATA_AS_ROW
		scale_cameras_down = pca.eigenvalues.at<double>(0) / 5.0;
	}

	visualizerListener->update(getPointCloud(),
		getPointCloudRGB(),
		getPointCloudBeforeBA(),
		getPointCloudRGBBeforeBA(),
		getCameras());
	WaitForVisualizationThread();

}

