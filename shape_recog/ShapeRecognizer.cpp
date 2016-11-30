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

	vector<Point> srcCs,shuffleCs;
	detectContours(img,srcCs,shuffleCs);

	Mat png;
	extractIPImage(img,srcCs,png);


	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();

	float minMatchingDist = 1e8;
	int minMatchDsitIndex = -1;
	for(int i = 0; i < mShapeNames.size();i++)
	{
		string strImgName = mShapeNames[i];
		vector<Point> &dbShape = mShapeDb[strImgName];
		float dis = mysc->computeDistance( shuffleCs, dbShape );
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

	string strPng = mShapeNames[minMatchDsitIndex] + std::string(".png");
	imwrite(strPng,png);
	printf("save cropped png:%s\n",strPng.c_str());
	waitKey(0);
	
	return 1;
}

void ShapeRecognizer::extractIPImage(cv::Mat img,std::vector<cv::Point> &srcContours ,cv::Mat &png)
{
	//get bounding box
	int leftx=10000,topy=10000,rightx=0,bottomy=0;
	for(int i = 0; i < srcContours.size();i++)
	{
		cv::Point pt = srcContours[i];
		if(pt.x < leftx)
			leftx = pt.x;
		if(pt.x > rightx)
			rightx = pt.x;
		
		if(pt.y < topy)
			topy = pt.y;
		if(pt.y > bottomy)
			bottomy = pt.y;
	}

	cv::Rect rct(leftx,topy,rightx-leftx+1,bottomy-topy+1);
	Mat subImg = img(rct).clone();

	std::vector<cv::Point> subContours;
	for(int i = 0; i < srcContours.size(); i++)
	{
		cv::Point pt = srcContours[i];
		pt.x -= rct.x;
		pt.y -= rct.y;
		subContours.push_back(pt);
	}
	Mat mask = Mat::zeros(Size(rct.width,rct.height),CV_8UC1);
	std::vector<std::vector<cv::Point>> subCCs;
	subCCs.push_back(subContours);
	drawContours(mask,subCCs,0,Scalar(25,255,255),-1);

	png = Mat::zeros(subImg.size(),CV_8UC4);
	for(int r = 0; r < subImg.rows;r++)
	{
		uchar *ptrImgRow = subImg.ptr<uchar>(r);
		uchar *ptrAlphaRow = mask.ptr<uchar>(r);
		uchar *ptrPngRow = png.ptr<uchar>(r);
		for(int c = 0; c < subImg.cols;c++)
		{
			ptrPngRow[4*c+0] = ptrImgRow[3*c+0];
			ptrPngRow[4*c+1] = ptrImgRow[3*c+1];
			ptrPngRow[4*c+2] = ptrImgRow[3*c+2];
			ptrPngRow[4*c+3] = ptrAlphaRow[c]==0?0:255;
		}
	}
	//imshow("png",png);
	//imshow("mask",mask);


}
void ShapeRecognizer::detectContours(cv::Mat img,std::vector<cv::Point> &srcContours,std::vector<cv::Point> &shuffleContours )
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
	 srcContours = tempCS;


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
		 shuffleContours.push_back(tempCS[i]);
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

		vector<cv::Point> &shuffleCs = mShapeDb[strImgName];
		vector<Point> srcCs;
		detectContours(img,srcCs,shuffleCs);
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