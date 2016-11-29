#pragma once
#include <string>

class CustomStringHelper
{
public:
	CustomStringHelper(void);
	~CustomStringHelper(void);

	static bool IsDateString(std::string str);
	static bool IsDateStringNoSeg(std::string str);//日期间没有分割号-

	static std::string ToLower(std::string strIn);
	static std::wstring s2ws(const std::string& s);
	static std::wstring s2ws(const char *pStr);

	static std::string GetCurrentTimeString();

	static std::string GetFileNameFromString(std::string str,bool bWithExt);

	static void FillString(char *pDst,int len,std::string strSrc);


	static std::string FormatString(const std::string fmt_str,...);

	static std::string GetParentFold(std::string strPath);

	

};
