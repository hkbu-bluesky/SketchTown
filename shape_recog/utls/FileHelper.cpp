
#include "FileHelper.h"
#include "CustomStringHelper.h"

#include <fstream>
#include <iostream>

#include <windows.h>
#include <Shlwapi.h> //for shloperation
#include <Shlobj.h>	//for SHBrowseForFolder
	
#include <direct.h>	//for _getcwd
#include <algorithm>		//for std::find
#pragma comment(lib,"Shlwapi.lib")

FileHelper::FileHelper(void)
{
}

FileHelper::~FileHelper(void)
{
}





void FileHelper::GetSubDirectories(std::string strFold,std::vector<std::string> &subFolds,bool absPath /* = false */)
{
		GetFileOrSubDirectories(strFold,subFolds,true,absPath);
}

void FileHelper::GetFiles(std::string strFold,std::vector<std::string> &files,bool absPath /* = false */ ,std::vector<std::string> exts
											,__time64_t t /* = 0*/)
{
	if(exts.size() == 0)
		GetFileOrSubDirectories(strFold,files,false,absPath,t);
	else
	{
		std::vector<std::string> allFiles;
		GetFileOrSubDirectories(strFold,allFiles,false,absPath,t);
		for(int i = 0; i < allFiles.size(); i++)
		{
			int pos = allFiles[i].find_last_of('.');
			std::string str = allFiles[i].substr(pos+1,allFiles[i].length() - pos -1);
			std::vector<std::string>::iterator iter = std::find(exts.begin(),exts.end(),str);
			if( iter!= exts.end())
				files.push_back(allFiles[i]);
		}
	}
}


void FileHelper::GetFiles(std::string strFold,std::vector<_finddata_t> &fileInfos, __time64_t t /* = 0 */)
{
	struct _finddata_t fileinfo;

	//检查strFold字符是否有在末尾添加"\\"或"/"符号		
	if(!(strFold.find_last_of('\\') == strFold.length() - 1 || strFold.find_last_of('/') ==   strFold.length() - 1 ))
		strFold += "\\";
	std::string searchPath = strFold+"*";	//后面加*表示搜索所有文件、文件夹
	long handle = _findfirst(searchPath.c_str(), &fileinfo);
	if ( -1 == handle )
	{
		_findclose(handle);
		return ;
	}
	do
	{
		if (strcmp(fileinfo.name ,".")==0||strcmp(fileinfo.name,"..")==0)
		{
			continue;
		}

		if(fileinfo.time_create > t && fileinfo.size != 0)
		{
			fileInfos.push_back(fileinfo);
		}		

	}while ( !_findnext(handle, &fileinfo) );	

	_findclose(handle);
}
/**************************************
*	@para results		搜索结果存放处
*	@para bDirectory 为真取子文件夹，为假取文件
*	@para absPath		为真返回绝对路径，为假返回相对路径
**************************************/
void FileHelper::GetFileOrSubDirectories(std::string strFold,std::vector<std::string> &results,bool bDirectory,
																		bool absPath /* = false */,
																		__time64_t t /* = 0*/)
{

	struct _finddata_t fileinfo;

	//检查strFold字符是否有在末尾添加"\\"或"/"符号		
	if(!(strFold.find_last_of('\\') == strFold.length() - 1 || strFold.find_last_of('/') ==   strFold.length() - 1 ))
		strFold += "\\";
	

	std::string searchPath = strFold+"*";	//后面加*表示搜索所有文件、文件夹
	long handle = _findfirst(searchPath.c_str(), &fileinfo);
	if ( -1 == handle )
	{
		_findclose(handle);
		return ;
	}
	do
	{
		if (strcmp(fileinfo.name ,".")==0||strcmp(fileinfo.name,"..")==0)
		{
			continue;
		}

		std::string filePath = strFold+fileinfo.name;
		if(fileinfo.time_create > t && 
			(
			(fileinfo.attrib &_A_SUBDIR && bDirectory) ||		//文件夹
			(!(fileinfo.attrib &_A_SUBDIR) && !bDirectory))	//文件
			)
		{
			//取文件夹或文件			
			if(absPath)
				results.push_back(filePath);
			else
				results.push_back(fileinfo.name);
		}		

	}while ( !_findnext(handle, &fileinfo) );	

	_findclose(handle);
}
bool FileHelper::IsPathExist(std::string strPath)
{
	/*DWORD dwAttrib = GetFileAttributes(strPath.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
	(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));*/

	return false;
}
bool FileHelper::IsFileExist(std::string strFile)
{
	struct _finddata_t fileinfo;
	long handle = _findfirst(strFile.c_str(), &fileinfo);
	if(handle != -1)
	{
		_findclose(handle);
		return true;
	}
	else
	{
		_findclose(handle);
		return false;
	}
}

__time64_t FileHelper::GetFileCreateTime(std::string strFile)
{
	struct _finddata_t fileinfo;
	long handle = _findfirst(strFile.c_str(), &fileinfo);
	__time64_t t= 0;
	if(handle != -1)
	{
		t = fileinfo.time_create;
		
	}
	_findclose(handle);
	return t;
}

std::string FileHelper::ReadString(std::string strFile)
{
	std::ifstream file;
	std::string strResult = "";
	file.open(strFile.c_str());
	if(file.is_open())
	{
		file.seekg(0,std::ios::end);
		std::streampos  size = file.tellg();
		char *memblock = new char [(int)size+1];
		file.seekg (0, std::ios::beg);
		file.read (memblock, size);
		file.close();
		memblock[(int)size] = '\0';
		strResult = std::string(memblock);
		delete[] memblock;
	}
	return strResult;
}

void FileHelper::WriteString(std::string strFile,const std::string &strContent,bool bAppend/* = false*/)
{
	FILE *fp;
	fopen_s(&fp, strFile.c_str(), "w");
	if (fp !=nullptr)	
		fwrite(strContent.c_str(), strContent.length(), 1, fp);
	else
	{
		printf("can not open file %s to write\n", strFile.c_str());
	}
	fclose(fp);
	
}

bool FileHelper::DeleteDirectoryFiles(std::string strFold)
{

	std::vector<std::string> strFiles;
	GetFiles(strFold,strFiles,true);
	for(int i = 0; i < strFiles.size(); i++)
	{
#ifdef _UNICODE
		std::wstring ws = CustomStringHelper::s2ws(strFiles[i]);		
		DeleteFile(ws.c_str());		
#else
		DeleteFile(strFiles[i].c_str());
#endif
	}
	
	return true;
}

void FileHelper::DeleteDirectory(std::string strFold)
{
	DeleteDirectoryFiles(strFold);
//	RemoveDirectory(strFold.c_str()); //win32 api
}

std::string FileHelper::GetWorkingDirectory()
{
	char cwd[MAX_PATH]={0};
	_getcwd(cwd,sizeof(cwd));
	return std::string(cwd);

}

std::string FileHelper::GetRunningDirectory()
{
	char szDir[255];
	GetModuleFileName(NULL,szDir,255);
	std::string strExe(szDir);
	int pos = strExe.find_last_of('\\');
	std::string strFold = strExe.substr(0,pos);

	return strFold;
}

void FileHelper::CreateFold(std::string strFold)
{
	if(IsPathExist(strFold.c_str()))
		return;
	
	std::vector<std::string> parentFolds;
	

	std::string strParent = strFold;
	parentFolds.push_back(strParent);

	
	while (strParent.size()>0)
	{
		strParent = CustomStringHelper::GetParentFold(strParent);
		if(!IsPathExist(strParent.c_str()))
			parentFolds.push_back(strParent);
		else
			break;
	
	}

	for(int i = parentFolds.size()-1; i >= 0; i--)
	{
#ifdef _UNICODE
		std::wstring ws = CustomStringHelper::s2ws(parentFolds[i]);
		CreateDirectory(ws.c_str(),NULL);
#else
		CreateDirectory(parentFolds[i].c_str(),NULL);
#endif
		
	}
	
}

void FileHelper::CopyFileX(std::string strSrc,std::string strDst,bool bForceCopy)
{

	CopyFile(strSrc.c_str(),strDst.c_str(),!bForceCopy);
}

void FileHelper::CopyFold(std::string strSrc, std::string strDst)
{
	std::vector<std::string> files;
	GetFiles(strSrc, files);
	if (files.size() > 0)
	{
		FileHelper::CreateFold(strDst);
		for (int i = 0; i < files.size(); i++)
		{
			std::string strFileSrc = strSrc + std::string("/") + files[i];
			std::string strFileDst = strDst + std::string("/") + files[i];
			CopyFileX(strFileSrc, strFileDst, true);
		}
	}
	
}
void FileHelper::DeleteFileX(std::string strFile)
{
//	DeleteFile(strFile.c_str());
}
std::string FileHelper::GetSelectedFold()
{
	BROWSEINFO  bi;
	bi.hwndOwner=NULL;
	bi.pidlRoot=NULL;
	bi.pszDisplayName=NULL;
	bi.lpszTitle=NULL;
	bi.ulFlags=0;
	bi.lpfn =NULL;
	bi.iImage =0;
	LPCITEMIDLIST pidl=SHBrowseForFolder(&bi);
	if(!pidl)
		return "";
	TCHAR  szDisplayName[255];
	SHGetPathFromIDList(pidl,szDisplayName);

	return std::string(szDisplayName);
}