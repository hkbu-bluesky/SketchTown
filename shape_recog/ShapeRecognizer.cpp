#include "StdAfx.h"
#include "ShapeRecognizer.h"
#include <opencv2/shape.hpp>
using namespace cv;
using namespace std;
ShapeRecognizer::ShapeRecognizer(void)
{
	mProcessImgWidth = 320;
}


ShapeRecognizer::~ShapeRecognizer(void)
{
}

void ShapeRecognizer::computeImageGradient(cv::Mat img, cv::Mat &gradImg)
{
	Canny(img,gradImg,50,80);
	return;

	Mat src;
	cvtColor(img,src,COLOR_BGR2GRAY);
	Mat dst_x, dst_y, dst;
	Sobel(src, dst_x, src.depth(), 1, 0);
	Sobel(src, dst_y, src.depth(), 0, 1);
	convertScaleAbs(dst_x, dst_x);
	convertScaleAbs(dst_y, dst_y);
	addWeighted( dst_x, 0.5, dst_y, 0.5, 0, dst);	
	gradImg = dst.clone();
}

int ShapeRecognizer::recogImageClass(cv::Mat img)
{
	
	int useWidth = mProcessImgWidth;
	int useHeight = img.rows*useWidth/img.cols;

	resize(img,img,Size(useWidth,useHeight));

	vector<Point> cs;
	detectContours(img,cs);



	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();

	float minMatchingDist = 1e8;
	int minMatchDsitIndex = -1;
	for(int i = 0; i < mShapeNames.size();i++)
	{
		string strImgName = mShapeNames[i];
		vector<Point> &dbShape = mShapeDb[strImgName];
		float dis = mysc->computeDistance( cs, dbShape );
		if(dis<minMatchingDist)
		{
			minMatchingDist = dis;
			minMatchDsitIndex = i;
		}
	}

	//if(minMatchingDist< 1)
	{
		printf("matched image name is :%s,dist=%.2f\n",mShapeNames[minMatchDsitIndex].c_str(),minMatchingDist);
	}
	printf("-----------------------------press any key to continue\n");
	imshow("img",img);
	waitKey(0);
	
	return 1;
}

void ShapeRecognizer::detectContours(cv::Mat img,std::vector<cv::Point> &contours )
{
	

	Mat grad;
	computeImageGradient(img,grad);
	
	threshold(grad,grad,40,255,cv::THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(grad,grad,element);
	erode(grad,grad,element);

	imshow("grad",grad);


	vector<vector<Point> > _contoursQuery;
	
	findContours(grad, _contoursQuery, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	int numMaxCs = 0;
	int maxCsIdx = 0;
	for(int i = 0; i < _contoursQuery.size(); i++)
	{
		if(_contoursQuery[i].size() > numMaxCs)
		{
			numMaxCs = _contoursQuery[i].size();
			maxCsIdx = i;
		}
	}


	std::vector<Point> tempCS;
	 approxPolyDP(Mat( _contoursQuery[maxCsIdx]), tempCS, arcLength(Mat(_contoursQuery[maxCsIdx]), true)*0.002, true);

	 vector<vector<Point>>ccs;
	 ccs.push_back(tempCS);
	 drawContours(img,ccs,0,Scalar(0,0,255),2);
	 imshow("img",img);

	 int dummy=0;
	 int n = 60;
	 for (int add=(int)tempCS.size()-1; add<n; add++)
	 {
		 tempCS.push_back(tempCS[dummy++]); //adding dummy values
	 }

	 // Uniformly sampling
	 random_shuffle(tempCS.begin(), tempCS.end());
	 vector<Point> cont;
	 for (int i=0; i<n; i++)
	 {
		 contours.push_back(tempCS[i]);
	 }
	int debug = 0;
}

void ShapeRecognizer::extractShapesFromDatabase(std::vector<std::string> &strImgs)
{
	mShapeDb.clear();
	mShapeNames.clear();


	for(int i = 0; i < strImgs.size(); i++)
	{
		string strImg = strImgs[i];
		Mat img = imread(strImg,0);
		int useWidth = mProcessImgWidth;
		int useHeight = img.rows*useWidth/img.cols;

		resize(img,img,Size(useWidth,useHeight));
		
		int pos = strImg.find_last_of('\\');
		//extract image name from image path
		string strImgName = strImg.substr(pos+1,strImg.length()-pos-1);

		vector<cv::Point> &cs = mShapeDb[strImgName];
		detectContours(img,cs);	
		mShapeNames.push_back(strImgName);

		printf("data base %s setup done\n",strImgName.c_str());
		/*
		Mat tempImg = imread(strImg);
		resize(tempImg,tempImg,Size(useWidth,useHeight));
		vector<vector<Point>>ccs;
		ccs.push_back(cs);
		drawContours(tempImg,ccs,0,Scalar(0,0,255),2);
		imshow("img",tempImg);
		waitKey(0);*/
	}

	
}