#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include "UDPConnection.h"

class ShapeRecognizer
{
public:
	ShapeRecognizer(void);
	~ShapeRecognizer(void);

	

	void computeImageGradient(cv::Mat img, cv::Mat &gradImg);

	int recogImageClass(cv::Mat img);

	void detectContours(cv::Mat img,std::vector<cv::Point> &srcContours,std::vector<cv::Point> &shuffleContours );

	void extractShapesFromDatabase(std::vector<std::string> &strImgs);

	void extractIPImage(cv::Mat img,std::vector<cv::Point> &srcContours ,cv::Mat &png);


	std::map<std::string,std::vector<cv::Point>> mShapeDb;
	std::vector<std::string> mShapeNames;
	int mProcessImgWidth;
	CUDPConnection mUDPSender;
};

