
#include "CustomStringHelper.h"
#include "stringEx.h"
#include <ctime>

#include <stdarg.h>  // for va_start, etc
#include <memory>    // for std::unique_ptr
#include <atlstr.h>		  // for A2T
#include <Windows.h>		//for MultiByteToWideChar
CustomStringHelper::CustomStringHelper(void)
{
}

CustomStringHelper::~CustomStringHelper(void)
{
}

bool CustomStringHelper::IsDateString(std::string str)
{
	//解析类似这样的字符串2014-04-03
	stringEx exStr(str);
	exStr.Trim();
	std::vector<stringEx> items = exStr.SplitToStrings('-');
	if(items.size() == 3 && items[0].GetString().length() == 4 &&
		items[1].GetString().length() == 2 &&
		items[2].GetString().length() == 2)
	{
		int y = atoi(items[0].GetString().c_str());
		int m = atoi(items[1].GetString().c_str());
		int d = atoi(items[2].GetString().c_str());
		if(y > 1000 && y < 9999 && m >= 1 && m <=12 && d>=1 && d <=31)
			return true;
	}
	return false;
}
bool CustomStringHelper::IsDateStringNoSeg(std::string str)
{
	//解析类似这样的字符串：20140403
	if(str.length() != 8)
		return false;
	int y = atoi(str.substr(0,4).c_str());
	int m = atoi(str.substr(4,2).c_str());
	int d = atoi(str.substr(6,2).c_str());
	if(y > 1000 && y < 9999 && m >= 1 && m <=12 && d>=1 && d <=31)
		return true;
	return false;
}

std::string CustomStringHelper::ToLower(std::string strIn)
{
	char *buffer = new char[strIn.length()+1];

	 
	for(int i = 0; i < strIn.length(); i++)
	{
		char c = strIn.at(i);
		if(c >= 'A' && c <= 'Z')
			c = c - ('Z' - 'z');
		buffer[i] = c;
	}
	buffer[strIn.length()] = '\0';
	std::string strLower(buffer);

	delete []buffer;
	return strLower;
}

std::wstring CustomStringHelper::s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0); 
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}
std::wstring CustomStringHelper::s2ws(const char *pStr)
{
	std::string str(pStr);
	return s2ws(str);
}

std::string CustomStringHelper::GetCurrentTimeString()
{
	time_t t = time(0);
	//struct tm * now = localtime(&t1);
	//time_t t = 0;
	struct tm  now;
	localtime_s(&now,&t);
	int y = now.tm_year + 1900;
	int m = now.tm_mon + 1;
	int d = now.tm_mday;
	int h = now.tm_hour;
	int mi = now.tm_min;
	int s = now.tm_sec;

	char szBuffer[255];
	sprintf_s(szBuffer,255,"%d%02d%02d%02d%02d%02d\0",y,m,d,h,mi,s);

	return std::string(szBuffer);
}

std::string CustomStringHelper::FormatString(const std::string fmt_str,...)
{
	int size = 100;
	std::string str;
	va_list ap;
	while (1) {
		str.resize(size);
		va_start(ap, fmt_str);
	//	int n = vsnprintf((char *)str.c_str(), size, fmt_str.c_str(), ap);
		int n = vsnprintf_s((char *)str.c_str(), size,size, fmt_str.c_str(), ap);
		va_end(ap);
		if (n > -1 && n < size) {
			str.resize(n);
			return str;
		}
		if (n > -1)
			size = n + 1;
		else
			size *= 2;
	}
	return str;
}

std::string CustomStringHelper::GetFileNameFromString(std::string str,bool bWithExt)
{
	size_t pos = str.find_last_of('\\');
	std::string strFullName;
	if(pos != std::string::npos)
	{
		strFullName = str.substr(pos+1,str.length()-pos-1);
	}
	else
		strFullName = str;

	if(bWithExt)
		return strFullName;

	size_t pos1 = strFullName.find_last_of('.');
	if(pos1 != std::string::npos)
		strFullName = strFullName.substr(0,pos1);

	return strFullName;


}

void CustomStringHelper::FillString(char *pDst,int len,std::string strSrc)
{
	int num = strSrc.length();
	if(num > len-1)
		num = len-1;
	
	if(num	!=	0)
		memcpy_s(pDst,len,strSrc.c_str(),num);
	pDst[num] = '\0';
}

std::string CustomStringHelper::GetParentFold(std::string strPath)
{
	std::string strFold;
	int pos0 = strPath.find_last_of('\\');
	int pos1 = strPath.find_last_of('/');
	int pos = pos0 > pos1 ? pos0 : pos1;	
	if (pos != -1)
	{
		strFold = strPath.substr(0, pos);
	}
	
	return strFold;
		
}