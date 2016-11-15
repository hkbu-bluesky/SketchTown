#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>

class ShapeRecognizer
{
public:
	ShapeRecognizer(void);
	~ShapeRecognizer(void);

	

	void computeImageGradient(cv::Mat img, cv::Mat &gradImg);

	int recogImageClass(cv::Mat img);

	void detectContours(cv::Mat img,std::vector<cv::Point> &contours );

	void extractShapesFromDatabase(std::vector<std::string> &strImgs);


	std::map<std::string,std::vector<cv::Point>> mShapeDb;
	std::vector<std::string> mShapeNames;
	int mProcessImgWidth;
};

