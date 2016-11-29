#pragma once
#include <map>
#include <vector>
#include <string>
#include <io.h>
class FileHelper
{
public:
	FileHelper(void);
	~FileHelper(void);

public:

	static void GetSubDirectories(std::string strFold,std::vector<std::string> &subFolds,bool absPath = false);	
	static void GetFiles(std::string strFold,std::vector<std::string> &files,bool absPath = false ,std::vector<std::string> exts =std::vector<std::string>(),__time64_t t = 0);
	
	static void GetFiles(std::string strFold,std::vector<_finddata_t> &fileInfos, __time64_t t = 0);
	
	
	static __time64_t GetFileCreateTime(std::string strFile);
	static bool IsFileExist(std::string strFile);
	static bool IsPathExist(std::string strPath);

	static std::string ReadString(std::string strFile);
	static void WriteString(std::string strFile,const std::string &strContent,bool bAppend = false);
private:
	static void GetFileOrSubDirectories(std::string strFold,std::vector<std::string> &subFolds,
																	bool bDirectory,bool absPath = false,__time64_t t = 0);




public:
	static bool DeleteDirectoryFiles(std::string strFold);
	static void DeleteDirectory(std::string strFold);
	//static void DeleteFile(std::string strFile);
	static std::string GetWorkingDirectory();
	static std::string GetRunningDirectory();

	static void CreateFold(std::string strFold);

	static void CopyFileX(std::string strSrc,std::string strDst,bool bForceCopy);
	static void CopyFold(std::string strSrc, std::string strDst);
	static void DeleteFileX(std::string strFile);

	static std::string GetSelectedFold();
};
