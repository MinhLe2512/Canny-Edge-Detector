#pragma once
#ifndef _EDGE_DETECTORS_H_
#define _EDGE_DETECTORS_H_
#include <opencv2/core/mat.hpp>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include "Convolution.h"

const float epsilon = 1e-6;
const float sobel_threshold = 0.1875;
const float highThreshold = 0.75;
const float lowThreshold = 0.6;
const float prewitt_threshold = 0.2;
const float laplace_threshold = 0.5;

class EdgeDetector {
private:
	int width;
	int height;
	uint8_t* edge_direction;
public:
	int detectBySobel(const cv::Mat& srcImg, cv::Mat& dstImg);
	int detectByPrewitt(const cv::Mat& srcImg, cv::Mat& dstImg);
	int detectByLaplace(const cv::Mat& srcImg, cv::Mat& dstImg);
	int detectByCanny(const cv::Mat& srcImg, cv::Mat& dstImg);
	EdgeDetector() {};
	~EdgeDetector() {
		delete[] edge_direction;
	};
};

#endif // ! _EDGE_DETECTORS_H_
