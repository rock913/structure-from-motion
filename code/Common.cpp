#include "Common.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define V3DLIB_ENABLE_SUITESPARSE
#include "Math/v3d_linear.h"
#include "Base/v3d_vrmlio.h"
#include "Geometry/v3d_metricbundle.h"


#include <iostream>
#include <windows.h>
#ifndef WIN32
#include <dirent.h>
#endif

using namespace V3D;
using namespace std;
using namespace cv;

#define DECOMPOSE_SVD
#ifndef CV_PCA_DATA_AS_ROW
#define CV_PCA_DATA_AS_ROW 0
#endif

#define EPSILON 0.0001

bool hasEnding(std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}

bool hasEndingLower(string const &fullString_, string const &_ending)
{
	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(), fullString_.end(), fullstring.begin(), ::tolower); // to lower
	return hasEnding(fullstring, ending);
}


void read_images_and_calibration_matrix(std::string dir_name,double downscale_factor)
{
	string dir_name_ = dir_name;
	vector<string> files_;

#ifndef WIN32
	//open a directory the POSIX way

	DIR *dp;
	struct dirent *ep;
	dp = opendir(dir_name);

	if (dp != NULL)
	{
		while (ep = readdir(dp)) {
			if (ep->d_name[0] != '.')
				files_.push_back(ep->d_name);
		}

		(void)closedir(dp);
	}
	else {
		cerr << ("Couldn't open the directory");
		return;
	}

#else
	//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;

	if (dir_name_[dir_name_.size() - 1] == '\\' || dir_name_[dir_name_.size() - 1] == '/') {
		dir_name_ = dir_name_.substr(0, dir_name_.size() - 1);
	}

	hFind = FindFirstFile(string(dir_name_).append("\\*").c_str(), &fdata);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(fdata.cFileName, ".") != 0 &&
				strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					continue; // a diretory
				}
				else
				{
					files_.push_back(fdata.cFileName);
				}
			}
		} while (FindNextFile(hFind, &fdata) != 0);
	}
	else {
		cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		cerr << "some other error with opening directory: " << GetLastError() << endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;
#endif

	for (unsigned int i = 0; i < files_.size(); i++) {
		if (files_[i][0] == '.' || !(hasEndingLower(files_[i], "jpg") || hasEndingLower(files_[i], "png"))) {
			continue;
		}
		cv::Mat m_ = cv::imread(string(dir_name_).append("/").append(files_[i]));
		if (downscale_factor != 1.0)
			cv::resize(m_, m_, Size(), downscale_factor, downscale_factor);
		images_names.push_back(files_[i]);
		images.push_back(m_);
	}

	std::cout << "=========================== Load Images ===========================\n";
	//ensure images are CV_8UC3
	for (unsigned int i = 0; i < images.size(); i++) {
		imgs_orig.push_back(cv::Mat_<cv::Vec3b>());
		if (!images[i].empty()) {
			if (images[i].type() == CV_8UC1) {
				cvtColor(images[i], imgs_orig[i], CV_GRAY2BGR);
			}
			else if (images[i].type() == CV_32FC3 || images[i].type() == CV_64FC3) {
				images[i].convertTo(imgs_orig[i], CV_8UC3, 255.0);
			}
			else {
				images[i].copyTo(imgs_orig[i]);
			}
		}

		grey_imgs.push_back(cv::Mat());
		cvtColor(imgs_orig[i], grey_imgs[i], CV_BGR2GRAY);

		imgpts.push_back(std::vector<cv::KeyPoint>());
		imgpts_good.push_back(std::vector<cv::KeyPoint>());
		std::cout << ".";
	}
	std::cout << std::endl;

	//load calibration matrix
	cv::FileStorage fs;
	if (fs.open(dir_name_ + "\\out_camera_data.yml", cv::FileStorage::READ)) {
		fs["camera_matrix"] >> cam_matrix;
		fs["distortion_coefficients"] >> distortion_coeff;
	}
	else {
		//no calibration matrix file - mockup calibration
		cv::Size imgs_size = images[0].size();
		double max_w_h = MAX(imgs_size.height, imgs_size.width);
		cam_matrix = (cv::Mat_<double>(3, 3) << max_w_h, 0, imgs_size.width / 2.0,
			0, max_w_h, imgs_size.height / 2.0,
			0, 0, 1);
		distortion_coeff = cv::Mat_<double>::zeros(1, 4);
	}

	K = cam_matrix;
	invert(K, Kinv); //get inverse of camera matrix

	distortion_coeff.convertTo(distcoeff_32f, CV_32FC1);
	K.convertTo(K_32f, CV_32FC1);
	}


	void optical_flow_feature_match()
	{
		//detect keypoints for all images
		FastFeatureDetector ffd;
		//	DenseFeatureDetector ffd;
		ffd.detect(grey_imgs, imgpts);

		int loop1_top = grey_imgs.size() - 1, loop2_top = grey_imgs.size();
		int frame_num_i = 0;
		#pragma omp parallel for
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
			{
				std::cout << "------------ Match " << images_names[frame_num_i] << "," << images_names[frame_num_j] << " ------------\n";
				std::vector<cv::DMatch> matches_tmp;
				MatchFeatures(frame_num_i, frame_num_j, &matches_tmp);
				matches_matrix[std::make_pair(frame_num_i, frame_num_j)] = matches_tmp;

				std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
				matches_matrix[std::make_pair(frame_num_j, frame_num_i)] = matches_tmp_flip;
			}
		}
	}

	void MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {
		vector<Point2f> i_pts;
		KeyPointsToPoints(imgpts[idx_i], i_pts);

		vector<Point2f> j_pts(i_pts.size());

		// making sure images are grayscale
		Mat prevgray, gray;
		if (grey_imgs[idx_i].channels() == 3) {
			cvtColor(grey_imgs[idx_i], prevgray, CV_RGB2GRAY);
			cvtColor(grey_imgs[idx_j], gray, CV_RGB2GRAY);
		}
		else {
			prevgray = grey_imgs[idx_i];
			gray = grey_imgs[idx_j];
		}

		vector<uchar> vstatus(i_pts.size()); vector<float> verror(i_pts.size());
		calcOpticalFlowPyrLK(prevgray, gray, i_pts, j_pts, vstatus, verror);

		double thresh = 1.0;
		vector<Point2f> to_find;
		vector<int> to_find_back_idx;
		for (unsigned int i = 0; i < vstatus.size(); i++) {
			if (vstatus[i] && verror[i] < 12.0) {
				to_find_back_idx.push_back(i);
				to_find.push_back(j_pts[i]);
			}
			else {
				vstatus[i] = 0;
			}
		}

		std::set<int> found_in_imgpts_j;
		Mat to_find_flat = Mat(to_find).reshape(1, to_find.size());

		vector<Point2f> j_pts_to_find;
		KeyPointsToPoints(imgpts[idx_j], j_pts_to_find);
		Mat j_pts_flat = Mat(j_pts_to_find).reshape(1, j_pts_to_find.size());

		vector<vector<DMatch> > knn_matches;
		//FlannBasedMatcher matcher;
		BFMatcher matcher(CV_L2);
		matcher.radiusMatch(to_find_flat, j_pts_flat, knn_matches, 2.0f);
		//Prune
		for (int i = 0; i < knn_matches.size(); i++) {
			DMatch _m;
			if (knn_matches[i].size() == 1) {
				_m = knn_matches[i][0];
			}
			else if (knn_matches[i].size()>1) {
				if (knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
					_m = knn_matches[i][0];
				}
				else {
					continue; // did not pass ratio test
				}
			}
			else {
				continue; // no match
			}
			if (found_in_imgpts_j.find(_m.trainIdx) == found_in_imgpts_j.end()) { // prevent duplicates
				_m.queryIdx = to_find_back_idx[_m.queryIdx]; //back to original indexing of points for <i_idx>
				matches->push_back(_m);
				found_in_imgpts_j.insert(_m.trainIdx);
			}
		}

		cout << "pruned " << matches->size() << " / " << knn_matches.size() << " matches" << endl;
#if 0
		{
			// draw flow field
			Mat img_matches; cvtColor(grey_imgs[idx_i], img_matches, CV_GRAY2BGR);
			i_pts.clear(); j_pts.clear();
			for (int i = 0; i < matches->size(); i++) {
				//if (i%2 != 0) {
				//				continue;
				//			}
				Point i_pt = imgpts[idx_i][(*matches)[i].queryIdx].pt;
				Point j_pt = imgpts[idx_j][(*matches)[i].trainIdx].pt;
				i_pts.push_back(i_pt);
				j_pts.push_back(j_pt);
				vstatus[i] = 1;
			}
			drawArrows(img_matches, i_pts, j_pts, vstatus, verror, Scalar(0, 255));
			stringstream ss;
			ss << matches->size() << " matches";
			ss.clear(); ss << "flow_field_"<<idx_i<<"and"<<idx_j << ".png";//<< omp_get_thread_num() 
			imshow(ss.str(), img_matches);

			//direct wirte
			imwrite(ss.str(), img_matches);
			//int c = waitKey(0);
			//if (c == 's') {
			//	imwrite(ss.str(), img_matches);
			//}
			destroyWindow(ss.str());
		}
#endif
	}


	void PruneMatchesBasedOnF() {
		//prune the match between <_i> and all views using the Fundamental matrix to prune
		//#pragma omp parallel for
		for (int _i = 0; _i < grey_imgs.size() - 1; _i++)
		{
			for (unsigned int _j = _i + 1; _j < grey_imgs.size(); _j++) {
				int older_view = _i, working_view = _j;

				GetFundamentalMat(imgpts[older_view],
					imgpts[working_view],
					imgpts_good[older_view],
					imgpts_good[working_view],
					matches_matrix[std::make_pair(older_view, working_view)]
#ifdef __SFM__DEBUG__
					,older_view, working_view 
#endif
					);
				//update flip matches as well
#pragma omp critical
				matches_matrix[std::make_pair(working_view, older_view)] = FlipMatches(matches_matrix[std::make_pair(older_view, working_view)]);
			}
		}
	}

	Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
		const vector<KeyPoint>& imgpts2,
		vector<KeyPoint>& imgpts1_good,
		vector<KeyPoint>& imgpts2_good,
		vector<DMatch>& matches
#ifdef __SFM__DEBUG__
		, int older_view,
		int working_view
#endif
		)
	{

		Mat img_1 = imgs_orig[older_view];
		Mat img_2 = imgs_orig[working_view];
		//Try to eliminate keypoints based on the fundamental matrix
		//(although this is not the proper way to do this)
		vector<uchar> status(imgpts1.size());

#ifdef __SFM__DEBUG__
		std::vector< DMatch > good_matches_;
		std::vector<KeyPoint> keypoints_1, keypoints_2;
#endif		
		imgpts1_good.clear(); imgpts2_good.clear();

		vector<KeyPoint> imgpts1_tmp;
		vector<KeyPoint> imgpts2_tmp;
		if (matches.size() <= 0) {
			//points already aligned...
			imgpts1_tmp = imgpts1;
			imgpts2_tmp = imgpts2;
		}
		else {
			GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
		}

		Mat F;
		{
			vector<Point2f> pts1, pts2;
			KeyPointsToPoints(imgpts1_tmp, pts1);
			KeyPointsToPoints(imgpts2_tmp, pts2);
#ifdef __SFM__DEBUG__
			cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_tmp.size() << ")" << endl;
			cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_tmp.size() << ")" << endl;
#endif
			double minVal, maxVal;
			cv::minMaxIdx(pts1, &minVal, &maxVal);
			F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
		}

		vector<DMatch> new_matches;
		cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
		for (unsigned int i = 0; i < status.size(); i++) {
			if (status[i])
			{
				imgpts1_good.push_back(imgpts1_tmp[i]);
				imgpts2_good.push_back(imgpts2_tmp[i]);

				if (matches.size() <= 0) { //points already aligned...
					new_matches.push_back(DMatch(matches[i].queryIdx, matches[i].trainIdx, matches[i].distance));
				}
				else {
					new_matches.push_back(matches[i]);
				}

#ifdef __SFM__DEBUG__
				good_matches_.push_back(DMatch(imgpts1_good.size() - 1, imgpts1_good.size() - 1, 1.0));
				keypoints_1.push_back(imgpts1_tmp[i]);
				keypoints_2.push_back(imgpts2_tmp[i]);
#endif
			}
		}

		cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
		matches = new_matches; //keep only those points who survived the fundamental matrix

#if 0
		//-- Draw only "good" matches
#ifdef __SFM__DEBUG__
		if (!img_1.empty() && !img_2.empty()) {
			vector<Point2f> i_pts, j_pts;
			Mat img_orig_matches;
			{ //draw original features in red
				vector<uchar> vstatus(imgpts1_tmp.size(), 1);
				vector<float> verror(imgpts1_tmp.size(), 1.0);
				img_1.copyTo(img_orig_matches);
				KeyPointsToPoints(imgpts1_tmp, i_pts);
				KeyPointsToPoints(imgpts2_tmp, j_pts);
				drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0, 0, 255));
			}
		{ //superimpose filtered features in green
			vector<uchar> vstatus(imgpts1_good.size(), 1);
			vector<float> verror(imgpts1_good.size(), 1.0);
			i_pts.resize(imgpts1_good.size());
			j_pts.resize(imgpts2_good.size());
			KeyPointsToPoints(imgpts1_good, i_pts);
			KeyPointsToPoints(imgpts2_good, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0, 255, 0));
			imshow("Filtered Matches", img_orig_matches);
		}


		stringstream ss;
		ss.clear(); ss << "pruned_flow_field_" << older_view << "and" << working_view << ".png";//<< omp_get_thread_num() 

		//direct wirte
		imwrite(ss.str(), img_orig_matches);
/*
		int c = waitKey(0);
		if (c == 's') {
			imwrite("fundamental_mat_matches.png", img_orig_matches);
		}*/
		destroyWindow("Filtered Matches");
		}
#endif		
#endif

		return F;
	}

	//Following Snavely07 4.2 - find how many inliers are in the Homography between 2 views
	int FindHomographyInliers2Views(int vi, int vj)
	{
		vector<cv::KeyPoint> ikpts, jkpts; vector<cv::Point2f> ipts, jpts;
		GetAlignedPointsFromMatch(imgpts[vi], imgpts[vj], matches_matrix[make_pair(vi, vj)], ikpts, jkpts);
		KeyPointsToPoints(ikpts, ipts); KeyPointsToPoints(jkpts, jpts);

		double minVal, maxVal; cv::minMaxIdx(ipts, &minVal, &maxVal); //TODO flatten point2d?? or it takes max of width and height

		vector<uchar> status;
		cv::Mat H = cv::findHomography(ipts, jpts, status, CV_RANSAC, 0.004 * maxVal); //threshold from Snavely07
		return cv::countNonZero(status); //number of inliers
	}


	//count number of 2D measurements
	int Count2DMeasurements(const vector<CloudPoint>& pointcloud) {
		int K = 0;
		for (unsigned int i = 0; i < pointcloud.size(); i++) {
			for (unsigned int ii = 0; ii < pointcloud[i].imgpt_for_img.size(); ii++) {
				if (pointcloud[i].imgpt_for_img[ii] >= 0) {
					K++;
				}
			}
		}
		return K;
	}

	inline void showErrorStatistics(double const f0,
		StdDistortionFunction const& distortion,
		vector<CameraMatrix> const& cams,
		vector<Vector3d> const& Xs,
		vector<Vector2d> const& measurements,
		vector<int> const& correspondingView,
		vector<int> const& correspondingPoint)
	{
		int const K = measurements.size();

		double meanReprojectionError = 0.0;
		for (int k = 0; k < K; ++k)
		{
			int const i = correspondingView[k];
			int const j = correspondingPoint[k];
			Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

			double reprojectionError = norm_L2(f0 * (p - measurements[k]));
			meanReprojectionError += reprojectionError;
		}
		cout << "mean reprojection error (in pixels): " << meanReprojectionError / K << endl;
	}


	void adjustBundle(vector<CloudPoint>& pointcloud,
		Mat& cam_matrix,
		const std::vector<std::vector<cv::KeyPoint> >& imgpts,
		std::map<int, cv::Matx34d>& Pmats
		)
	{
		int N = Pmats.size(), M = pointcloud.size(), K = Count2DMeasurements(pointcloud);

		cout << "N (cams) = " << N << " M (points) = " << M << " K (measurements) = " << K << endl;

		StdDistortionFunction distortion;

		//conver camera intrinsics to BA datastructs
		Matrix3x3d KMat;
		makeIdentityMatrix(KMat);
		KMat[0][0] = cam_matrix.at<double>(0, 0); //fx
		KMat[1][1] = cam_matrix.at<double>(1, 1); //fy
		KMat[0][1] = cam_matrix.at<double>(0, 1); //skew
		KMat[0][2] = cam_matrix.at<double>(0, 2); //ppx
		KMat[1][2] = cam_matrix.at<double>(1, 2); //ppy

		double const f0 = KMat[0][0];
		cout << "intrinsic before bundle = "; displayMatrix(KMat);
		Matrix3x3d Knorm = KMat;
		// Normalize the intrinsic to have unit focal length.
		scaleMatrixIP(1.0 / f0, Knorm);
		Knorm[2][2] = 1.0;

		vector<int> pointIdFwdMap(M);
		map<int, int> pointIdBwdMap;

		//conver 3D point cloud to BA datastructs
		vector<Vector3d > Xs(M);
		for (int j = 0; j < M; ++j)
		{
			int pointId = j;
			Xs[j][0] = pointcloud[j].pt.x;
			Xs[j][1] = pointcloud[j].pt.y;
			Xs[j][2] = pointcloud[j].pt.z;
			pointIdFwdMap[j] = pointId;
			pointIdBwdMap.insert(make_pair(pointId, j));
		}
		cout << "Read the 3D points." << endl;

		vector<int> camIdFwdMap(N, -1);
		map<int, int> camIdBwdMap;

		//convert cameras to BA datastructs
		vector<CameraMatrix> cams(N);
		for (int i = 0; i < N; ++i)
		{
			int camId = i;
			Matrix3x3d R;
			Vector3d T;

			Matx34d& P = Pmats[i];

			R[0][0] = P(0, 0); R[0][1] = P(0, 1); R[0][2] = P(0, 2); T[0] = P(0, 3);
			R[1][0] = P(1, 0); R[1][1] = P(1, 1); R[1][2] = P(1, 2); T[1] = P(1, 3);
			R[2][0] = P(2, 0); R[2][1] = P(2, 1); R[2][2] = P(2, 2); T[2] = P(2, 3);

			camIdFwdMap[i] = camId;
			camIdBwdMap.insert(make_pair(camId, i));

			cams[i].setIntrinsic(Knorm);
			cams[i].setRotation(R);
			cams[i].setTranslation(T);
		}
		cout << "Read the cameras." << endl;

		vector<Vector2d > measurements;
		vector<int> correspondingView;
		vector<int> correspondingPoint;

		measurements.reserve(K);
		correspondingView.reserve(K);
		correspondingPoint.reserve(K);

		//convert 2D measurements to BA datastructs
		for (unsigned int k = 0; k < pointcloud.size(); ++k)
		{
			for (unsigned int i = 0; i < pointcloud[k].imgpt_for_img.size(); i++) {
				if (pointcloud[k].imgpt_for_img[i] >= 0) {
					int view = i, point = k;
					Vector3d p, np;

					Point cvp = imgpts[i][pointcloud[k].imgpt_for_img[i]].pt;
					p[0] = cvp.x;
					p[1] = cvp.y;
					p[2] = 1.0;

					if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
						pointIdBwdMap.find(point) != pointIdBwdMap.end())
					{
						// Normalize the measurements to match the unit focal length.
						scaleVectorIP(1.0 / f0, p);
						measurements.push_back(Vector2d(p[0], p[1]));
						correspondingView.push_back(camIdBwdMap[view]);
						correspondingPoint.push_back(pointIdBwdMap[point]);
					}
				}
			}
		} // end for (k)

		K = measurements.size();

		cout << "Read " << K << " valid 2D measurements." << endl;

		showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

		//	V3D::optimizerVerbosenessLevel = 1;
		double const inlierThreshold = 2.0 / fabs(f0);

		Matrix3x3d K0 = cams[0].getIntrinsic();
		cout << "K0 = "; displayMatrix(K0);

		bool good_adjustment = false;
		{
			ScopedBundleExtrinsicNormalizer extNorm(cams, Xs);
			ScopedBundleIntrinsicNormalizer intNorm(cams, measurements, correspondingView);
			CommonInternalsMetricBundleOptimizer opt(V3D::FULL_BUNDLE_FOCAL_LENGTH_PP, inlierThreshold, K0, distortion, cams, Xs,
				measurements, correspondingView, correspondingPoint);
			//		StdMetricBundleOptimizer opt(inlierThreshold,cams,Xs,measurements,correspondingView,correspondingPoint);

			opt.tau = 1e-3;
			opt.maxIterations = 50;
			opt.minimize();

			cout << "optimizer status = " << opt.status << endl;

			good_adjustment = (opt.status != 2);
		}

		cout << "refined K = "; displayMatrix(K0);

		for (int i = 0; i < N; ++i) cams[i].setIntrinsic(K0);

		Matrix3x3d Knew = K0;
		scaleMatrixIP(f0, Knew);
		Knew[2][2] = 1.0;
		cout << "Knew = "; displayMatrix(Knew);

		showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

		if (good_adjustment) { //good adjustment?

			//Vector3d mean(0.0, 0.0, 0.0);
			//for (unsigned int j = 0; j < Xs.size(); ++j) addVectorsIP(Xs[j], mean);
			//scaleVectorIP(1.0/Xs.size(), mean);
			//
			//vector<float> norms(Xs.size());
			//for (unsigned int j = 0; j < Xs.size(); ++j)
			//	norms[j] = distance_L2(Xs[j], mean);
			//
			//std::sort(norms.begin(), norms.end());
			//float distThr = norms[int(norms.size() * 0.9f)];
			//cout << "90% quantile distance: " << distThr << endl;

			//extract 3D points
			for (unsigned int j = 0; j < Xs.size(); ++j)
			{
				//if (distance_L2(Xs[j], mean) > 3*distThr) makeZeroVector(Xs[j]);

				pointcloud[j].pt.x = Xs[j][0];
				pointcloud[j].pt.y = Xs[j][1];
				pointcloud[j].pt.z = Xs[j][2];
			}

			//extract adjusted cameras
			for (int i = 0; i < N; ++i)
			{
				Matrix3x3d R = cams[i].getRotation();
				Vector3d T = cams[i].getTranslation();

				Matx34d P;
				P(0, 0) = R[0][0]; P(0, 1) = R[0][1]; P(0, 2) = R[0][2]; P(0, 3) = T[0];
				P(1, 0) = R[1][0]; P(1, 1) = R[1][1]; P(1, 2) = R[1][2]; P(1, 3) = T[1];
				P(2, 0) = R[2][0]; P(2, 1) = R[2][1]; P(2, 2) = R[2][2]; P(2, 3) = T[2];

				Pmats[i] = P;
			}


			//TODO: extract camera intrinsics
			cam_matrix.at<double>(0, 0) = Knew[0][0];
			cam_matrix.at<double>(0, 1) = Knew[0][1];
			cam_matrix.at<double>(0, 2) = Knew[0][2];
			cam_matrix.at<double>(1, 1) = Knew[1][1];
			cam_matrix.at<double>(1, 2) = Knew[1][2];
		}
	}

	void attach(SfMUpdateListener *sul)
	{
		listeners.push_back(sul);
	}
	void update()
	{
		for (int i = 0; i < listeners.size(); i++)
			listeners[i]->update(getPointCloud(),
			getPointCloudRGB(),
			getPointCloudBeforeBA(),
			getPointCloudRGBBeforeBA(),
			getCameras());
	}


	void Find2D3DCorrespondences(int working_view,
		std::vector<cv::Point3f>& ppcloud,
		std::vector<cv::Point2f>& imgPoints)
	{
		ppcloud.clear(); imgPoints.clear();

		vector<int> pcloud_status(pcloud.size(), 0);
		for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view)
		{
			int old_view = *done_view;
			//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
			std::vector<cv::DMatch> matches_from_old_to_working = matches_matrix[std::make_pair(old_view, working_view)];

			for (unsigned int match_from_old_view = 0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
				// the index of the matching point in <old_view>
				int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;

				//scan the existing cloud (pcloud) to see if this point from <old_view> exists
				for (unsigned int pcldp = 0; pcldp < pcloud.size(); pcldp++) {
					// see if corresponding point was found in this point
					if (idx_in_old_view == pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
					{
						//3d point in cloud
						ppcloud.push_back(pcloud[pcldp].pt);
						//2d point in image i
						imgPoints.push_back(imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);

						pcloud_status[pcldp] = 1;
						break;
					}
				}
			}
		}
		cout << "found " << ppcloud.size() << " 3d-2d point correspondences" << endl;
	}

	bool FindPoseEstimation(
		int working_view,
		cv::Mat_<double>& rvec,
		cv::Mat_<double>& t,
		cv::Mat_<double>& R,
		std::vector<cv::Point3f> ppcloud,
		std::vector<cv::Point2f> imgPoints
		)
	{
		if (ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
			//something went wrong aligning 3D to 2D points..
			cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << endl;
			return false;
		}

		vector<int> inliers;
		double minVal, maxVal; cv::minMaxIdx(imgPoints, &minVal, &maxVal);
		//"solvePnPRansac", 
		cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);
		
		vector<cv::Point2f> projected3D;
		cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);

		if (inliers.size() == 0) { //get inliers
			for (int i = 0; i < projected3D.size(); i++) {
				if (norm(projected3D[i] - imgPoints[i]) < 10.0)
					inliers.push_back(i);
			}
		}
		//cv::Rodrigues(rvec, R);
		//visualizerShowCamera(R,t,0,255,0,0.1);

		if (inliers.size() < (double)(imgPoints.size()) / 5.0) {
			cerr << "not enough inliers to consider a good pose (" << inliers.size() << "/" << imgPoints.size() << ")" << endl;
			return false;
		}

		if (cv::norm(t) > 200.0) {
			// this is bad...
			cerr << "estimated camera movement is too big, skip this camera\r\n";
			return false;
		}

		cv::Rodrigues(rvec, R);
		if (!CheckCoherentRotation(R)) {
			cerr << "rotation is incoherent. we should try a different base view..." << endl;
			return false;
		}

		std::cout << "found t = " << t << "\nR = \n" << R << std::endl;
		return true;
	}
	void RecoverCamerasIncremental()
	{
		cv::Matx34d P1 = Pmats[m_second_view];
		cv::Mat_<double> t = (cv::Mat_<double>(1, 3) << P1(0, 3), P1(1, 3), P1(2, 3));
		cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << P1(0, 0), P1(0, 1), P1(0, 2),
			P1(1, 0), P1(1, 1), P1(1, 2),
			P1(2, 0), P1(2, 1), P1(2, 2));
		cv::Mat_<double> rvec(1, 3); Rodrigues(R, rvec);

		done_views.insert(m_first_view);
		done_views.insert(m_second_view);
		good_views.insert(m_first_view);
		good_views.insert(m_second_view);

		//loop images to incrementally recover more cameras 
		//for (unsigned int i=0; i < imgs.size(); i++) 
		while (done_views.size() != grey_imgs.size())
		{
			//find image with highest 2d-3d correspondance [Snavely07 4.2]
			unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
			vector<cv::Point3f> max_3d; vector<cv::Point2f> max_2d;
			for (unsigned int _i = 0; _i < grey_imgs.size(); _i++) {
				if (done_views.find(_i) != done_views.end()) continue; //already done with this view

				vector<cv::Point3f> tmp3d; vector<cv::Point2f> tmp2d;
				cout << images_names[_i] << ": ";
				Find2D3DCorrespondences(_i, tmp3d, tmp2d);
				if (tmp3d.size() > max_2d3d_count) {
					max_2d3d_count = tmp3d.size();
					max_2d3d_view = _i;
					max_3d = tmp3d; max_2d = tmp2d;
				}
			}
			int i = max_2d3d_view; //highest 2d3d matching view

			std::cout << "-------------------------- " << images_names[i] << " --------------------------\n";
			done_views.insert(i); // don't repeat it for now

			bool pose_estimated = FindPoseEstimation(i, rvec, t, R, max_3d, max_2d);
			if (!pose_estimated)
				continue;

			//store estimated pose	
			Pmats[i] = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
				R(1, 0), R(1, 1), R(1, 2), t(1),
				R(2, 0), R(2, 1), R(2, 2), t(2));

			// start triangulating with previous GOOD views
			for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view)
			{
				int view = *done_view;
				if (view == i) continue; //skip current...

				cout << " -> " << images_names[view] << endl;

				vector<CloudPoint> new_triangulated;
				vector<int> add_to_cloud;
				bool good_triangulation = TriangulatePointsBetweenViews(i, view, new_triangulated, add_to_cloud);
				if (!good_triangulation) continue;

				std::cout << "before triangulation: " << pcloud.size();
				for (int j = 0; j < add_to_cloud.size(); j++) {
					if (add_to_cloud[j] == 1)
						pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << pcloud.size() << std::endl;
				//break;
			}
			good_views.insert(i);

			AdjustCurrentBundle();
			update();
		}

		cout << "======================================================================\n";
		cout << "========================= Depth Recovery DONE ========================\n";
		cout << "======================================================================\n";
	}

	void AdjustCurrentBundle() {
		cout << "======================== Bundle Adjustment ==========================\n";

		pointcloud_beforeBA = pcloud;
		GetRGBForPointCloud(pointcloud_beforeBA, pointCloudRGB_beforeBA);

		cv::Mat _cam_matrix = K;
		adjustBundle(pcloud, _cam_matrix, imgpts, Pmats);
		K = cam_matrix;
		Kinv = K.inv();

		cout << "use new K " << endl << K << endl;

		GetRGBForPointCloud(pcloud, pointCloudRGB);
	}

	void GetRGBForPointCloud(
		const std::vector<struct CloudPoint>& _pcloud,
		std::vector<cv::Vec3b>& RGBforCloud
		)
	{
		RGBforCloud.resize(_pcloud.size());
		for (unsigned int i = 0; i < _pcloud.size(); i++) {
			unsigned int good_view = 0;
			std::vector<cv::Vec3b> point_colors;
			for (; good_view < imgs_orig.size(); good_view++) {
				if (_pcloud[i].imgpt_for_img[good_view] != -1) {
					int pt_idx = _pcloud[i].imgpt_for_img[good_view];
					if (pt_idx >= imgpts[good_view].size()) {
						std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << std::endl;
						continue;
					}
					cv::Point _pt = imgpts[good_view][pt_idx].pt;
					assert(good_view < imgs_orig.size() && _pt.x < imgs_orig[good_view].cols && _pt.y < imgs_orig[good_view].rows);

					point_colors.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));

					//				std::stringstream ss; ss << "patch " << good_view;
					//				imshow_250x250(ss.str(), imgs_orig[good_view](cv::Range(_pt.y-10,_pt.y+10),cv::Range(_pt.x-10,_pt.x+10)));
				}
			}
			//		cv::waitKey(0);
			cv::Scalar res_color = cv::mean(point_colors);
			RGBforCloud[i] = (cv::Vec3b(res_color[0], res_color[1], res_color[2])); //bgr2rgb
			if (good_view == grey_imgs.size()) //nothing found.. put red dot
				RGBforCloud.push_back(cv::Vec3b(255, 0, 0));
		}
	}


	bool sort_by_first(pair<int, pair<int, int> > a, pair<int, pair<int, int> > b) { return a.first < b.first; }
	/**
	* Get an initial 3D point cloud from 2 views only
	*/
	void GetBaseLineTriangulation() {
		std::cout << "=========================== Baseline triangulation ===========================\n";

		cv::Matx34d P(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0),
			P1(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);

		std::vector<CloudPoint> tmp_pcloud;

		//sort pairwise matches to find the lowest Homography inliers [Snavely07 4.2]
		cout << "Find highest match...";
		list<pair<int, pair<int, int> > > matches_sizes;
		//TODO: parallelize!
		for (std::map<std::pair<int, int>, std::vector<cv::DMatch> >::iterator i = matches_matrix.begin(); i != matches_matrix.end(); ++i) {
			if ((*i).second.size() < 100)
				matches_sizes.push_back(make_pair(100, (*i).first));
			else {
				int Hinliers = FindHomographyInliers2Views((*i).first.first, (*i).first.second);
				int percent = (int)(((double)Hinliers) / ((double)(*i).second.size()) * 100.0);
				cout << "[" << (*i).first.first << "," << (*i).first.second << " = " << percent << "] ";
				matches_sizes.push_back(make_pair((int)percent, (*i).first));
			}
		}
		cout << endl;
		matches_sizes.sort(sort_by_first);

		//Reconstruct from two views
		bool goodF = false;
		int highest_pair = 0;
		m_first_view = m_second_view = 0;
		//reverse iterate by number of matches
		for (list<pair<int, pair<int, int> > >::iterator highest_pair = matches_sizes.begin();
			highest_pair != matches_sizes.end() && !goodF;
			++highest_pair)
		{
			m_second_view = (*highest_pair).second.second;
			m_first_view = (*highest_pair).second.first;

			std::cout << " -------- " << images_names[m_first_view] << " and " << images_names[m_second_view] << " -------- " << std::endl;
			//what if reconstrcution of first two views is bad? fallback to another pair
			//See if the Fundamental Matrix between these two views is good
			goodF = FindCameraMatrices(K, Kinv, distortion_coeff,
				imgpts[m_first_view],
				imgpts[m_second_view],
				imgpts_good[m_first_view],
				imgpts_good[m_second_view],
				P,
				P1,
				matches_matrix[std::make_pair(m_first_view, m_second_view)],
				tmp_pcloud
#ifdef __SFM__DEBUG__
				, m_first_view, m_second_view
#endif
				);
			if (goodF) {
				vector<CloudPoint> new_triangulated;
				vector<int> add_to_cloud;

				Pmats[m_first_view] = P;
				Pmats[m_second_view] = P1;

				bool good_triangulation = TriangulatePointsBetweenViews(m_second_view, m_first_view, new_triangulated, add_to_cloud);
				if (!good_triangulation || cv::countNonZero(add_to_cloud) < 10) {
					std::cout << "triangulation failed" << std::endl;
					goodF = false;
					Pmats[m_first_view] = 0;
					Pmats[m_second_view] = 0;
					m_second_view++;
				}
				else {
					std::cout << "before triangulation: " << pcloud.size();
					for (unsigned int j = 0; j<add_to_cloud.size(); j++) {
						if (add_to_cloud[j] == 1)
							pcloud.push_back(new_triangulated[j]);
					}
					std::cout << " after " << pcloud.size() << std::endl;
				}
			}
		}

		if (!goodF) {
			cerr << "Cannot find a good pair of images to obtain a baseline triangulation" << endl;
			exit(0);
		}

		cout << "Taking baseline from " << images_names[m_first_view] << " and " << images_names[m_second_view] << endl;

	}


	bool TriangulatePointsBetweenViews(
		int working_view,
		int older_view,
		vector<struct CloudPoint>& new_triangulated,
		vector<int>& add_to_cloud
		)
	{
		cout << " Triangulate " << images_names[working_view] << " and " << images_names[older_view] << endl;
		//get the left camera matrix
		//TODO: potential bug - the P mat for <view> may not exist? or does it...
		cv::Matx34d P = Pmats[older_view];
		cv::Matx34d P1 = Pmats[working_view];

		std::vector<cv::KeyPoint> pt_set1, pt_set2;
		std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(older_view, working_view)];
		GetAlignedPointsFromMatch(imgpts[older_view], imgpts[working_view], matches, pt_set1, pt_set2);


		//adding more triangulated points to general cloud
		double reproj_error = TriangulatePoints(pt_set1, pt_set2, K, Kinv, distortion_coeff, P, P1, new_triangulated, correspImg1Pt);
		std::cout << "triangulation reproj error " << reproj_error << std::endl;

		vector<uchar> trig_status;
		if (!TestTriangulation(new_triangulated, P, trig_status) || !TestTriangulation(new_triangulated, P1, trig_status)) {
			cerr << "Triangulation did not succeed" << endl;
			return false;
		}

		//filter out outlier points with high reprojection
		vector<double> reprj_errors;
		for (int i = 0; i < new_triangulated.size(); i++) { reprj_errors.push_back(new_triangulated[i].reprojection_error); }
		std::sort(reprj_errors.begin(), reprj_errors.end());
		//get the 80% precentile
		double reprj_err_cutoff = reprj_errors[4 * reprj_errors.size() / 5] * 2.4; //threshold from Snavely07 4.2

		vector<CloudPoint> new_triangulated_filtered;
		std::vector<cv::DMatch> new_matches;
		for (int i = 0; i < new_triangulated.size(); i++) {
			if (trig_status[i] == 0)
				continue; //point was not in front of camera
			if (new_triangulated[i].reprojection_error > 16.0) {
				continue; //reject point
			}
			if (new_triangulated[i].reprojection_error < 4.0 ||
				new_triangulated[i].reprojection_error < reprj_err_cutoff)
			{
				new_triangulated_filtered.push_back(new_triangulated[i]);
				new_matches.push_back(matches[i]);
			}
			else
			{
				continue;
			}
		}

		cout << "filtered out " << (new_triangulated.size() - new_triangulated_filtered.size()) << " high-error points" << endl;

		//all points filtered?
		if (new_triangulated_filtered.size() <= 0) return false;

		new_triangulated = new_triangulated_filtered;

		matches = new_matches;
		matches_matrix[std::make_pair(older_view, working_view)] = new_matches; //just to make sure, remove if unneccesary
		matches_matrix[std::make_pair(working_view, older_view)] = FlipMatches(new_matches);
		add_to_cloud.clear();
		add_to_cloud.resize(new_triangulated.size(), 1);
		int found_other_views_count = 0;
		int num_views = grey_imgs.size();

		//scan new triangulated points, if they were already triangulated before - strengthen cloud
		//#pragma omp parallel for num_threads(1)
		for (int j = 0; j < new_triangulated.size(); j++) {
			new_triangulated[j].imgpt_for_img = std::vector<int>(grey_imgs.size(), -1);

			//matches[j] corresponds to new_triangulated[j]
			//matches[j].queryIdx = point in <older_view>
			//matches[j].trainIdx = point in <working_view>
			new_triangulated[j].imgpt_for_img[older_view] = matches[j].queryIdx;	//2D reference to <older_view>
			new_triangulated[j].imgpt_for_img[working_view] = matches[j].trainIdx;		//2D reference to <working_view>
			bool found_in_other_view = false;
			for (unsigned int view_ = 0; view_ < num_views; view_++) {
				if (view_ != older_view) {
					//Look for points in <view_> that match to points in <working_view>
					std::vector<cv::DMatch> submatches = matches_matrix[std::make_pair(view_, working_view)];
					for (unsigned int ii = 0; ii < submatches.size(); ii++) {
						if (submatches[ii].trainIdx == matches[j].trainIdx &&
							!found_in_other_view)
						{
							//Point was already found in <view_> - strengthen it in the known cloud, if it exists there

							//cout << "2d pt " << submatches[ii].queryIdx << " in img " << view_ << " matched 2d pt " << submatches[ii].trainIdx << " in img " << i << endl;
							for (unsigned int pt3d = 0; pt3d < pcloud.size(); pt3d++) {
								if (pcloud[pt3d].imgpt_for_img[view_] == submatches[ii].queryIdx)
								{
									//pcloud[pt3d] - a point that has 2d reference in <view_>

									//cout << "3d point "<<pt3d<<" in cloud, referenced 2d pt " << submatches[ii].queryIdx << " in view " << view_ << endl;
#pragma omp critical 
								{
									pcloud[pt3d].imgpt_for_img[working_view] = matches[j].trainIdx;
									pcloud[pt3d].imgpt_for_img[older_view] = matches[j].queryIdx;
									found_in_other_view = true;
									add_to_cloud[j] = 0;
								}
								}
							}
						}
					}
				}
			}
#pragma omp critical
		{
			if (found_in_other_view) {
				found_other_views_count++;
			}
			else {
				add_to_cloud[j] = 1;
			}
		}
		}
		std::cout << found_other_views_count << "/" << new_triangulated.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
		return true;
	}


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
						) 
{
	//Find camera matrices
	{
		cout << "Find camera matrices...";
		double t = getTickCount();
		
		Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches
#ifdef __SFM__DEBUG__
			, older_view, working_view
#endif
								  );
		if(matches.size() < 100) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
			cerr << "not enough inliers after F matrix" << endl;
			return false;
		}
		
		//Essential matrix: compute then extract cameras [R|t]
		Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

		//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
		if(fabsf(determinant(E)) > 1e-07) {
			cout << "det(E) != 0 : " << determinant(E) << "\n";
			P1 = 0;
			return false;
		}
		
		Mat_<double> R1(3,3);
		Mat_<double> R2(3,3);
		Mat_<double> t1(1,3);
		Mat_<double> t2(1,3);

		//decompose E to P' , HZ (9.19)
		{			
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

			if(determinant(R1)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
				E = -E;
				DecomposeEtoRandT(E,R1,R2,t1,t2);
			}
			if (!CheckCoherentRotation(R1)) {
				cout << "resulting rotation is not coherent\n";
				P1 = 0;
				return false;
			}
			
			P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
			cout << "Testing P1 " << endl << Mat(P1) << endl;
			
			vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
			double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
			double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
			vector<uchar> tmp_status;
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
				reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
				
				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
					if (!CheckCoherentRotation(R2)) {
						cout << "resulting rotation is not coherent\n";
						P1 = 0;
						return false;
					}
					
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
					reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
					
					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
						reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
						
						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
							cout << "Shit." << endl; 
							return false;
						}
					}				
				}			
			}
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
			}
		}		
		
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Done. (" << t <<"s)"<< endl;
	}
	return true;
}


std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i = 0; i < cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) {
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

	Matx44d P4x4 = Matx44d::eye();
	for (int i = 0; i < 12; i++) P4x4.val[i] = P.val[i];

	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

	status.resize(pcloud.size(), 0);
	for (int i = 0; i < pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if (percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if (false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(), 3);
		for (unsigned int i = 0; i < pcloud.size(); i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm, mean, CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i = 0; i < pcloud.size(); i++) {
			Vec3d w = Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if (D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if ((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}

//Triagulate points
double TriangulatePoints(const vector<KeyPoint>& pt_set1,
	const vector<KeyPoint>& pt_set2,
	const Mat& K,
	const Mat& Kinv,
	const Mat& distcoeff,
	const Matx34d& P,
	const Matx34d& P1,
	vector<CloudPoint>& pointcloud,
	vector<KeyPoint>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
	vector<double> depths;
#endif

	//	pointcloud.clear();
	correspImg1Pt.clear();

	Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
		P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
		P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
		0, 0, 0, 1);
	Matx44d P1inv(P1_.inv());

	cout << "Triangulating...";
	double t = getTickCount();
	vector<double> reproj_error;
	unsigned int pts_size = pt_set1.size();

	Mat_<double> KP1 = K * Mat(P1);
#pragma omp parallel for num_threads(1)
	for (int i = 0; i < pts_size; i++) {
		Point2f kp = pt_set1[i].pt;
		Point3d u(kp.x, kp.y, 1.0);
		Mat_<double> um = Kinv * Mat_<double>(u);
		u.x = um(0); u.y = um(1); u.z = um(2);

		Point2f kp1 = pt_set2[i].pt;
		Point3d u1(kp1.x, kp1.y, 1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);
		//		cout << "3D Point: " << X << endl;
		//		Mat_<double> x = Mat(P1) * X;
		//		cout <<	"P1 * Point: " << x << endl;
		//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
		//		cout <<	"Point: " << xPt << endl;
		Mat_<double> xPt_img = KP1 * X;				//reproject
		//		cout <<	"Point * K: " << xPt_img << endl;
		Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

#pragma omp critical
		{
			double reprj_err = norm(xPt_img_ - kp1);
			reproj_error.push_back(reprj_err);

			CloudPoint cp;
			cp.pt = Point3d(X(0), X(1), X(2));
			cp.reprojection_error = reprj_err;

			pointcloud.push_back(cp);
			correspImg1Pt.push_back(pt_set1[i]);
		}
	}

	Scalar mse = mean(reproj_error);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Done. (" << pointcloud.size() << "points, " << t << "s, mean reproj err = " << mse[0] << ")" << endl;

	return mse[0];
}

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
	Matx34d P,		//camera 1 matrix
	Point3d u1,		//homogenous image point in 2nd camera
	Matx34d P1		//camera 2 matrix
	)
{

	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
	Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));

	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}


/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
	Matx34d P,			//camera 1 matrix
	Point3d u1,			//homogenous image point in 2nd camera
	Matx34d P1			//camera 2 matrix
	) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);

	Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
	X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

	for (int i = 0; i<10; i++) { //Hartley suggests 10 iterations at most		

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
			(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
			(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
			(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
			);
		Mat_<double> B = (Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
			-(u.y*P(2, 3) - P(1, 3)) / wi,
			-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
			-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
			);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}


bool CheckCoherentRotation(cv::Mat_<double>& R) {

	if (fabsf(determinant(R)) - 1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}

bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2)
{
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E, svd_u, svd_vt, svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if (singular_values_ratio > 1.0) singular_values_ratio = 1.0 / singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0, -1, 0,	//HZ 9.13
		1, 0, 0,
		0, 0, 1);
	Matx33d Wt(0, 1, 0,
		-1, 0, 0,
		0, 0, 1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
	return true;
}


void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
	//Using OpenCV's SVD
	cv::SVD svd(E, cv::SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;

	cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n" << svd_u << "\nW:\n" << svd_w << "\nVt:\n" << svd_vt << endl;
	cout << "----------------------------------------------------\n";
}

	void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
		ps.clear();
		for (unsigned int i = 0; i < kps.size(); i++) ps.push_back(kps[i].pt);
	}

	void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
		kps.clear();
		for (unsigned int i = 0; i < ps.size(); i++) kps.push_back(KeyPoint(ps[i], 1.0f));
	}


#define intrpmnmx(val,min,max) (max==min ? 0.0 : ((val)-min)/(max-min))

	void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, const vector<float>& verror, const Scalar& _line_color)
	{
		double minVal, maxVal; minMaxIdx(verror, &minVal, &maxVal, 0, 0, status);
		int line_thickness = 1;

		for (size_t i = 0; i < prevPts.size(); ++i)
		{
			if (status[i])
			{
				double alpha = intrpmnmx(verror[i], minVal, maxVal); alpha = 1.0 - alpha;
				Scalar line_color(alpha*_line_color[0],
					alpha*_line_color[1],
					alpha*_line_color[2]);

				Point p = prevPts[i];
				Point q = nextPts[i];

				double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

				double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

				if (hypotenuse < 1.0)
					continue;

				// Here we lengthen the arrow by a factor of three.
				q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
				q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

				// Now we draw the main line of the arrow.
				line(frame, p, q, line_color, line_thickness);

				// Now draw the tips of the arrow. I do some scaling so that the
				// tips look proportional to the main line of the arrow.

				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
				line(frame, p, q, line_color, line_thickness);

				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
				line(frame, p, q, line_color, line_thickness);
			}
		}
	}


	std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches) {
		std::vector<cv::DMatch> flip;
		for (int i = 0; i < matches.size(); i++) {
			flip.push_back(matches[i]);
			swap(flip.back().queryIdx, flip.back().trainIdx);
		}
		return flip;
	}


	void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
		const std::vector<cv::KeyPoint>& imgpts2,
		const std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& pt_set1,
		std::vector<cv::KeyPoint>& pt_set2)
	{
		for (unsigned int i = 0; i < matches.size(); i++) {
			//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
			assert(matches[i].queryIdx < imgpts1.size());
			pt_set1.push_back(imgpts1[matches[i].queryIdx]);
			assert(matches[i].trainIdx < imgpts2.size());
			pt_set2.push_back(imgpts2[matches[i].trainIdx]);
		}
	}


	std::vector<cv::Point3d> getPointCloud() { return CloudPointsToPoints(pcloud); }
	const cv::Mat& get_im_orig(int frame_num) { return imgs_orig[frame_num]; }
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { return correspImg1Pt; }
	const std::vector<cv::Vec3b>& getPointCloudRGB() { if (pointCloudRGB.size() == 0) { GetRGBForPointCloud(pcloud, pointCloudRGB); } return pointCloudRGB; }
	std::vector<cv::Matx34d> getCameras() {
		std::vector<cv::Matx34d> v;
		for (std::map<int, cv::Matx34d>::const_iterator it = Pmats.begin(); it != Pmats.end(); ++it) {
			v.push_back(it->second);
		}
		return v;
	}
	std::vector<cv::Point3d> getPointCloudBeforeBA() { return CloudPointsToPoints(pointcloud_beforeBA); }
	const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA() { return pointCloudRGB_beforeBA; }
