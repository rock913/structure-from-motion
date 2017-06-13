#ifndef _COMMON_H
#define _COMMON_H
#pragma once
#pragma warning(disable: 4244 18 4996 4800)

#define  __SFM__DEBUG__
#define _SCL_SECURE_NO_WARNINGS

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include <list>
#include <set>
#include <string>

#ifdef DEFINE_GLOBALS
#define EXTERN
#else
#define EXTERN extern
#endif
//////////////////////////////////////////////////data structure
struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};
//images
EXTERN std::vector<cv::Mat> images;
EXTERN std::vector<std::string> images_names;
EXTERN std::vector<cv::Mat_<cv::Vec3b> > imgs_orig;
EXTERN std::vector<cv::Mat> grey_imgs;

//image keypoints
EXTERN std::vector<std::vector<cv::KeyPoint> > imgpts;
EXTERN std::vector<std::vector<cv::KeyPoint> > fullpts;
EXTERN std::vector<std::vector<cv::KeyPoint> > imgpts_good;

//match
EXTERN std::map<std::pair<int, int>, std::vector<cv::DMatch> > matches_matrix;

//calibration matrix
EXTERN cv::Mat K;
EXTERN cv::Mat_<double> Kinv;
EXTERN cv::Mat cam_matrix, distortion_coeff;
EXTERN cv::Mat distcoeff_32f;
EXTERN cv::Mat K_32f;

//cloud
EXTERN std::vector<CloudPoint> pcloud;
EXTERN std::vector<cv::Vec3b> pointCloudRGB;
EXTERN std::vector<cv::KeyPoint> correspImg1Pt; //TODO: remove
EXTERN std::vector<CloudPoint> pointcloud_beforeBA;
EXTERN std::vector<cv::Vec3b> pointCloudRGB_beforeBA;

EXTERN int m_first_view;
EXTERN int m_second_view; //baseline's second view other to 0
EXTERN std::set<int> done_views;
EXTERN std::set<int> good_views;

EXTERN std::map<int, cv::Matx34d> Pmats;


class SfMUpdateListener
{
public:
	virtual void update(std::vector<cv::Point3d> pcld,
		std::vector<cv::Vec3b> pcldrgb,
		std::vector<cv::Point3d> pcld_alternate,
		std::vector<cv::Vec3b> pcldrgb_alternate,
		std::vector<cv::Matx34d> cameras) = 0;
};
EXTERN std::vector < SfMUpdateListener * > listeners;

//////////////////////////////////////////////procedural programming
void read_images_and_calibration_matrix(std::string dir_name, double downscale_factor);
void optical_flow_feature_match();
void PruneMatchesBasedOnF();
void GetBaseLineTriangulation();
void AdjustCurrentBundle();
void attach(SfMUpdateListener *sul);
void update();
void RecoverCamerasIncremental();
//////////////////////////////assist functions
using namespace std;
using namespace cv;
void MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches);
std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);
void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, const vector<float>& verror, const Scalar& _line_color);
void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);
void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps);
Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
	const vector<KeyPoint>& imgpts2,
	vector<KeyPoint>& imgpts1_good,
	vector<KeyPoint>& imgpts2_good,
	vector<DMatch>& matches
#ifdef __SFM__DEBUG__
	, int older_view,
	int working_view
#endif
	);
Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
	const vector<KeyPoint>& imgpts2,
	vector<KeyPoint>& imgpts1_good,
	vector<KeyPoint>& imgpts2_good,
	vector<DMatch>& matches
#ifdef __SFM__DEBUG__
	, int older_view,
	int working_view
#endif
	);

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
	const std::vector<cv::KeyPoint>& imgpts2,
	const std::vector<cv::DMatch>& matches,
	std::vector<cv::KeyPoint>& pt_set1,
	std::vector<cv::KeyPoint>& pt_set2);
int FindHomographyInliers2Views(int vi, int vj);

bool FindCameraMatrices(const Mat& K,
	const Mat& Kinv,
	const Mat& distcoeff,
	const vector<KeyPoint>& imgpts1,
	const vector<KeyPoint>& imgpts2,
	vector<KeyPoint>& imgpts1_good,
	vector<KeyPoint>& imgpts2_good,
	Matx34d& P,
	Matx34d& P1,
	vector<DMatch>& matches,
	vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
	, int older_view,
	int working_view
#endif
	);

bool DecomposeEtoRandT(Mat_<double>& E, Mat_<double>& R1, Mat_<double>& R2, Mat_<double>& t1, Mat_<double>& t2);
void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w);
bool CheckCoherentRotation(cv::Mat_<double>& R);

double TriangulatePoints(const vector<KeyPoint>& pt_set1,
	const vector<KeyPoint>& pt_set2,
	const Mat& K,
	const Mat& Kinv,
	const Mat& distcoeff,
	const Matx34d& P,
	const Matx34d& P1,
	vector<CloudPoint>& pointcloud,
	vector<KeyPoint>& correspImg1Pt);

Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
	Matx34d P,		//camera 1 matrix
	Point3d u1,		//homogenous image point in 2nd camera
	Matx34d P1		//camera 2 matrix
	);

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
	Matx34d P,			//camera 1 matrix
	Point3d u1,			//homogenous image point in 2nd camera
	Matx34d P1			//camera 2 matrix
	);
bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status);
std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

bool TriangulatePointsBetweenViews(
	int working_view,
	int older_view,
	vector<struct CloudPoint>& new_triangulated,
	vector<int>& add_to_cloud
	);

void GetRGBForPointCloud(
	const std::vector<struct CloudPoint>& _pcloud,
	std::vector<cv::Vec3b>& RGBforCloud
	);

void adjustBundle(vector<CloudPoint>& pointcloud,
	Mat& cam_matrix,
	const std::vector<std::vector<cv::KeyPoint> >& imgpts,
	std::map<int, cv::Matx34d>& Pmats
	);
int Count2DMeasurements(const vector<CloudPoint>& pointcloud);

std::vector<cv::Point3d> getPointCloud();
const cv::Mat& get_im_orig(int frame_num);
const std::vector<cv::KeyPoint>& getcorrespImg1Pt();
const std::vector<cv::Vec3b>& getPointCloudRGB();
std::vector<cv::Matx34d> getCameras();
std::vector<cv::Point3d> getPointCloudBeforeBA();
const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA();

void RecoverCamerasIncremental();


void Find2D3DCorrespondences(int working_view,
	std::vector<cv::Point3f>& ppcloud,
	std::vector<cv::Point2f>& imgPoints);

bool FindPoseEstimation(
	int working_view,
	cv::Mat_<double>& rvec,
	cv::Mat_<double>& t,
	cv::Mat_<double>& R,
	std::vector<cv::Point3f> ppcloud,
	std::vector<cv::Point2f> imgPoints
	);

#endif