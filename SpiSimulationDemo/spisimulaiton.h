#ifndef __SPISIMULATION__
#define __SPISIMULATION__

#include <iostream>
#include <cmath>
#include <ctime>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <gsl/gsl_fft.h>

#define PI acos(-1.0)

struct Mat4Step
{
	cv::Mat Mat1;
	cv::Mat Mat2;
	cv::Mat Mat3;
	cv::Mat Mat4;
	Mat4Step(int rows, int cols, int type) {
		Mat1 = cv::Mat(rows, cols, type);
		Mat2 = cv::Mat(rows, cols, type);
		Mat3 = cv::Mat(rows, cols, type);
		Mat4 = cv::Mat(rows, cols, type);
	}
};

#endif // !__SPISIMULATION__