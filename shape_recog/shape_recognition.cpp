// shape_recognition.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ShapeRecognizer.h"
#include "FileHelper.h"
#include <string>
using namespace cv;
using namespace std;
int _tmain(int argc, _TCHAR* argv[])
{

	string strFold("images\\database");
	vector<string> strImgs;
	vector<string> strImgExts;
	strImgExts.push_back("png");
	strImgExts.push_back("jpg");
	FileHelper::GetFiles(strFold,strImgs,true,strImgExts);

	ShapeRecognizer recog;
	recog.extractShapesFromDatabase(strImgs);

	string strQueryFold("images\\testimages");
	vector<string> strQueryImgs;
	FileHelper::GetFiles(strQueryFold,strQueryImgs,true,strImgExts);

	for(int i = 0; i < strQueryImgs.size();i++)
	{
		printf("process %s \n",strQueryImgs[i].c_str());

		Mat img = imread(strQueryImgs[i]);		
		recog.recogImageClass(img);

		
	}

	
	return 0;
}

