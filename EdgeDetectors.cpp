#include "EdgeDetectors.h"

int EdgeDetector::detectBySobel(const cv::Mat& srcImg, cv::Mat& dstImg) {
	int rows = srcImg.rows, cols = srcImg.cols;
	dstImg = cv::Mat(rows - 2, cols - 2, CV_32FC1, cv::Scalar(0));
	cv::Mat cloneImg = srcImg.clone();
	//Sobel kernel
	//kernelY -1 | 0 | 1  kernelX -1 | -2 | -1
	//        -2 | 0 | 2           0 |  0 |  0
	//        -1 | 0 | 1           1 |  2 |  1
	std::vector<float> kernelX = { -0.25, 0, 0.25, -0.5, 0, 0.5, -0.25, 0, 0.25 };
	std::vector<float> kernelY = { -0.25, -0.5, -0.25, 0, 0, 0, 0.25, 0.5, 0.25 };

	Convolution convoX, convoY;
	convoX.setKernel(kernelX, 3, 3);
	convoY.setKernel(kernelY, 3, 3);

	cv::Mat Gx, Gy;

	convoX.doConvolution(cloneImg, Gx);
	cv::imshow("Gx", Gx);

	convoY.doConvolution(cloneImg, Gy);
	cv::imshow("Gy", Gy);
	cv::waitKey(0);

	for (int i = 0; i < dstImg.cols; i++) {
		for (int j = 0; j < dstImg.rows; j++) {
			float fx = Gx.ptr<float>(i)[j];
			float fy = Gy.ptr<float>(i)[j];
			float e = sqrt(fx * fx + fy * fy);
			if (e - sobel_threshold >= epsilon) {
				dstImg.ptr<float>(i)[j] = 255;
			}
		}
	}
	return 0;
}

int EdgeDetector::detectByPrewitt(const cv::Mat& srcImg, cv::Mat& dstImg) {
	int rows = srcImg.rows, cols = srcImg.cols;
	dstImg = cv::Mat(rows - 2, cols - 2, CV_32FC1, cv::Scalar(0));
	cv::Mat cloneImg = srcImg.clone();
	//Prewitt kernel
	//kernelY -1 | 0 | 1  kernelX -1 | -1 | -1
	//        -1 | 0 | 1           0 |  0 |  0
	//        -1 | 0 | 1           1 |  1 |  1
	std::vector<float> kernelX = { -0.33, 0, 0.33, -0.33, 0, 0.33, -0.33, 0, 0.33 };
	std::vector<float> kernelY = { -0.33, -0.33, -0.33, 0, 0, 0, 0.33, 0.33, 0.33 };

	Convolution convoX, convoY;
	convoX.setKernel(kernelX, 3, 3);
	convoY.setKernel(kernelY, 3, 3);

	cv::Mat Gx, Gy;

	convoX.doConvolution(cloneImg, Gx);
	cv::imshow("Gx", Gx);

	convoY.doConvolution(cloneImg, Gy);
	cv::imshow("Gy", Gy);
	cv::waitKey(0);

	for (int i = 0; i < dstImg.cols; i++) {
		for (int j = 0; j < dstImg.rows; j++) {
			float fx = Gx.ptr<float>(i)[j];
			float fy = Gy.ptr<float>(i)[j];
			float e = sqrt(fx * fx + fy * fy);
			if (e - prewitt_threshold >= epsilon) {
				dstImg.ptr<float>(i)[j] = 255;
			}
		}
	}
	return 0;
}
int EdgeDetector::detectByLaplace(const cv::Mat& srcImg, cv::Mat& dstImg) {
	int rows = srcImg.rows, cols = srcImg.cols;
	dstImg = cv::Mat(rows - 2, cols - 2, CV_32FC1, cv::Scalar(0));
	cv::Mat cloneImg = srcImg.clone();
		//ლ(◉◞⊖◟◉｀ლ)
		//Change threshold for better edge
		//kernel 1 |  1 | 1 
		//       1 | -8 | 1 
		//       1 |  1 | 1 
		std::vector<float> weightLaplace{ 1, 1, 1, 1, -8, 1, 1, 1, 1 };

		cv::Mat tmpImg = cv::Mat(rows - 2, cols - 2, CV_32FC1, cv::Scalar(0));
		Convolution convo;
		convo.setKernel(weightLaplace, 3, 3);
		convo.doConvolution(cloneImg, tmpImg);

		//move direction (-1, -1) = NW, (1, -1) = NE, (0, 1) = Down, (-1, 0) = Left 
		std::vector<int> dx = { -1, 1, 0, -1 };
		std::vector<int> dy = { -1, -1, 1, 0 };

		for (int i = 1; i < tmpImg.cols - 1; i++) {
			for (int j = 1; j < tmpImg.rows - 1; j++) {
				int count = 0;
				for (int dir = 0; dir < 4; dir++) {
					float value1 = tmpImg.at<float>(j + dx[dir], i + dy[dir]);
					float value2 = tmpImg.at<float>(j - dx[dir], i - dy[dir]);

					int s1 = value1 < 0 ? -1 : 1;
					int s2 = value2 < 0 ? -1 : 1;
					//found zero crossing
					if (s1 != s2 && abs(value1 - value2) - epsilon > laplace_threshold)
						count++;
				}
				if (count >= 2)
					dstImg.at<float>(j, i) = 255;
			}
		}
		return 0;
}
int EdgeDetector::detectByCanny(const cv::Mat& srcImg, cv::Mat& dstImg) {
	int rows = srcImg.rows, cols = srcImg.cols;
	dstImg = cv::Mat(rows - 2, cols - 2, CV_32FC1, cv::Scalar(0));
	cv::Mat cloneImg = srcImg.clone();

	edge_direction = new uint8_t[rows * cols];
	for (int i = 0; i < cols; i++) 
		for (int j = 0; j < rows; j++)
			edge_direction[i * rows + j] = 0;
	width = dstImg.rows;
	height = dstImg.cols;
	//Step 1: Noise reduction
	//Gaussian kernel
	std::vector<float> gaussianKernel = { 2.0f / 159.0f, 4.0f / 159.0f, 5.0f / 159.0f, 4.0f / 159.0f, 2.0f / 159.0f,
										4.0f / 159.0f, 9.0f / 159.0f, 12.0f / 159.0f, 9.0f / 159.0f, 4.0f / 159.0f,
										5.0f / 159.0f, 12.0f / 159.0f, 15.0f / 159.0f, 12.0f / 159.0f, 5.0f / 159.0f,
										4.0f / 159.0f, 9.0f / 159.0f, 12.0f / 159.0f, 9.0f / 159.0f, 4.0f / 159.0f,
										2.0f / 159.0f, 4.0f / 159.0f, 5.0f / 159.0f, 4.0f / 159.0f, 2.0f / 159.0f };
	cv::Mat gaussianBlur;

	Convolution convo;
	convo.setKernel(gaussianKernel, 5, 5);
	convo.doConvolution(cloneImg, gaussianBlur);

	cv::imshow("Gaussian Blur", gaussianBlur);
	cv::waitKey(0);
	//Step 2: Gradient Calculation:
	//Sobel kernel
	std::vector<float> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	std::vector<float> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	Convolution convoX, convoY;
	convoX.setKernel(kernelX, 3, 3);
	convoY.setKernel(kernelY, 3, 3);

	cv::Mat Gx, Gy;

	convoX.doConvolution(cloneImg, Gx);
	cv::imshow("Gx", Gx);

	convoY.doConvolution(cloneImg, Gy);
	cv::imshow("Gy", Gy);
	cv::waitKey(0);

	float angle = 0.0;
	float max = 0.0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float fx = Gx.ptr<float>(i)[j];
			float fy = Gy.ptr<float>(i)[j];
			dstImg.ptr<float>(i)[j] = sqrt(fx * fx + fy * fy);
			max = (dstImg.ptr<float>(i)[j] > max) ? dstImg.ptr<float>(i)[j] : max;

			// Angle calculation.
			if ((fx != 0.0) || (fy != 0.0)) {
				angle = atan2(fx, fy) * 180.0 / 3.14;
			}
			else {
				angle = 0.0;
			}
			if (((angle > -22.5) && (angle <= 22.5)) ||
				((angle > 157.5) && (angle <= -157.5))) {
				edge_direction[i * dstImg.rows + j] = 0;
			}
			else if (((angle > 22.5) && (angle <= 67.5)) ||
				((angle > -157.5) && (angle <= -112.5))) {
				edge_direction[i * dstImg.rows + j] = 45;
			}
			else if (((angle > 67.5) && (angle <= 112.5)) ||
				((angle > -112.5) && (angle <= -67.5))) {
				edge_direction[i * dstImg.rows + j] = 90;
			}
			else if (((angle > 112.5) && (angle <= 157.5)) ||
				((angle > -67.5) && (angle <= -22.5))) {
				edge_direction[i * dstImg.rows + j] = 135;
			}

		}
	}


	cv::imshow("Step 2", dstImg);
	cv::waitKey(0);
		
	//Step 3: Non Max Supression
	float p1 = 0.0f;
	float p2 = 0.0f;
	float pixel;

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (edge_direction[i * width + j] == 0) {
				p1 = dstImg.ptr<float>(i + 1)[j];
				p2 = dstImg.ptr<float>(i - 1)[j];
			}
			else if (edge_direction[i * width + j] == 45) {
				p1 = dstImg.ptr<float>(i + 1)[j - 1];
				p2 = dstImg.ptr<float>(i - 1)[j + 1];
			}
			else if (edge_direction[i * width + j] == 90) {
				p1 = dstImg.ptr<float>(i)[j - 1];
				p2 = dstImg.ptr<float>(i)[j + 1];
			}
			else if (edge_direction[i * width + j] == 135) {
				p1 = dstImg.ptr<float>(i + 1)[j + 1];
				p2 = dstImg.ptr<float>(i - 1)[j - 1];
			}
			pixel = dstImg.ptr<float>(i)[j];
			(pixel >= p1 && pixel >= p2) ? dstImg.ptr<float>(i)[j] = pixel
				: dstImg.ptr<float>(i)[j] = 0;
		}
	}
	cv::imshow("Step 3", dstImg);
	cv::waitKey(0);
	//Step 4: Hysteresis:

	//for (int x = 0; x < height; x++) {
	//	for (int y = 0; y < width; y++) {
	//		if (dstImg.ptr<float>(x)[y] >= highThreshold) {
	//			dstImg.ptr<float>(x)[y] = 1.0;
	//			//std::cout << x << " " << y << std::endl;
	//			for (long x1 = x - 1; x1 <= x + 1; x1++) {
	//				for (long y1 = y - 1; y1 <= y + 1; y1++) {
	//					if ((x1 < height) & (y1 < width) & (x1 >= 0) & (y1 >= 0)
	//						& (x1 != x) & (y1 != y)) {
	//						float value = dstImg.ptr<float>(x1)[y1];
	//						if (value != 1.0) {
	//							if (value >= lowThreshold) {
	//								dstImg.ptr<float>(x1)[y1] = 1.0;
	//								x = x1;
	//								y = y1;
	//							}
	//							else {
	//								dstImg.ptr<float>(x1)[y1] = 0.0;
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}

	//}
	//for (int x = 0; x < height; x++) {
	//	for (int y = 0; y < width; y++) {
	//		if (dstImg.ptr<float>(x)[y] != 1.0) {
	//			dstImg.ptr<float>(x)[y] = 0.0;
	//		}
	//	}
	//}
	return 0;
}

