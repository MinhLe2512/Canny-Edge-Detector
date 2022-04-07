#include <iostream>
#include "Convolution.h"
#include "EdgeDetectors.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, char* argv[]) {
	char* str = new char[20];
	char* inFile = new char[20];
	char* flag = new char[20];
	int parameter;

	for (int i = 1; i < argc; i++) {
		if (i == 1) inFile = argv[i];
		else if (i == 2) flag = argv[i];
		else if (i == 3) parameter = atoi(argv[i]);
	}

	cv::Mat srcImg = cv::imread(inFile, cv::IMREAD_COLOR);
	cv::Mat grayImg;
	cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	cv::Mat dstImg;
	
	EdgeDetector edg = EdgeDetector();
	if (strcmp(flag, "-sobel") == 0) {
		if (edg.detectBySobel(grayImg, dstImg) == 0) {
			cv::imshow("Output image", dstImg);
			cv::waitKey(0);
		}
		else {
			std::cout << "Can't write to file" << std::endl;
		}
	}
	else if (strcmp(flag, "-prewitt") == 0) {
		if (edg.detectByPrewitt(grayImg, dstImg) == 0) {
			cv::imshow("Output image", dstImg);
			cv::waitKey(0);
		}
		else {
			std::cout << "Can't write to file" << std::endl;
		}
	}
	else if (strcmp(flag, "-laplace") == 0) {
		if (edg.detectByLaplace(grayImg, dstImg) == 0) {
			cv::imshow("Output image", dstImg);
			cv::waitKey(0);
		}
		else {
			std::cout << "Can't write to file" << std::endl;
		}
	}
	else if (strcmp(flag, "-canny") == 0) {
		if (edg.detectByCanny(grayImg, dstImg) == 0) {
			cv::imshow("Output image", dstImg);
			cv::waitKey(0);
		}
		else {
			std::cout << "Can't write to file" << std::endl;
		}
	}

	return 0;
}
