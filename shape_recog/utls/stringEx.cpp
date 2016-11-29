/****************************************************************
Author:	Justin Ko
Date :		2013-7-23
Description: 基于std::string的扩展操作
*****************************************************************/

#include "stringEx.h"
using namespace std;
stringEx::stringEx(void)
{
	m_str="";
}
stringEx::stringEx(char *pStr)
{
	string strIn(pStr);
	m_str=strIn;
}
stringEx::stringEx(string str)
{
	m_str=str;
}
void stringEx::TrimLeft()
{
	if(m_str.length()==0)return;
	char cBlank=' ';
	int pos=m_str.find_first_not_of(cBlank);
	m_str.erase(0,pos);
}
void stringEx::TrimRight()
{
	if(m_str.length()==0)return;
	char cBlank=' ';
	int pos=m_str.find_last_not_of(cBlank);
	m_str.erase(pos+1,m_str.length()-pos-1);
}
void stringEx::Trim()
{
	this->TrimLeft();
	this->TrimRight();
}
void stringEx::MakeLower()
{
	string strOut="";
	for(size_t i=0;i<m_str.length();i++)
	{
		strOut+=tolower(m_str[i]);
	}
	m_str= strOut;
}
void stringEx::MakeUpper()
{
	string strOut="";
	for(size_t i=0;i<m_str.length();i++)
	{
		strOut+=toupper(m_str[i]);
	}
	m_str=strOut;
}


stringEx stringEx::ReadLine(string strMark)
{
	stringEx temp=m_str;
	
	size_t nStart=temp.FindStrings(strMark);
	size_t nEnd=temp.GetString().find_first_of('\n');
	stringEx result="";
	if(nStart!=string::npos)
	{
		if(nEnd<nStart)
		{
			//换行符在标记符的左边
			temp.Erase(0,nStart);
			nEnd=temp.GetString().find_first_of('\n');
			if(nEnd==string::npos)//只剩下一行，且没有换行符了
			{
				result=temp;
			}
			else
			{
				result.Copy(temp.GetString().substr(0,nEnd));
			}
			
		}
		else
		{
			result.Copy(temp.GetString().substr(nStart,nEnd-nStart));
		}
	}
	return result;
}
size_t stringEx::FindStrings(string strMark)
{
	string temp=m_str;
	size_t lastPos,pos;
	lastPos=0;
	if(strMark.empty())return -1;
	bool bFound=false;
	while(temp.length()>=strMark.length())
	{
		pos=temp.find_first_of(strMark);
		if(pos!=string::npos)
		{
			string sub=temp.substr(pos,strMark.length());
			lastPos+=pos;
			if(sub==strMark)
			{				
				bFound=true;
				break;
			}
			else
			{
				lastPos+=1;
				temp.erase(0,pos+1);
			}
		}
		else
		{
			return string::npos;
		}
	}
	if(bFound)return lastPos;
	else return string::npos;

}
stringEx stringEx::PopLine(string strMark)
{
	stringEx line=this->ReadLine(strMark);
	if(line.GetString().empty())return line;

	size_t pos=this->FindStrings(strMark);
	if(pos == string::npos)
		return "";
	else m_str.erase(0,pos);
	pos=m_str.find_first_of('\n');
	m_str.erase(0,pos+1);//加1是跳过换行符
	return line;
}
stringEx stringEx::PopLine()
{
	size_t pos = m_str.find_first_of('\n');
	string line = m_str.substr(0,pos);
	m_str.erase(0,pos+1);
	stringEx lineEx(line);
	return lineEx;
}
vector<float> stringEx::SplitToFloats(char seperator)
{
	vector<float> vFloats;
	vector<stringEx> items = this->SplitToStrings(seperator);
	for(size_t i = 0 ; i < items.size(); ++i)
	{
		stringEx item = items[i];
		float data = atof(item.GetString().c_str());
		vFloats.push_back(data);
	}
	return vFloats;
}
vector<int> stringEx::SplitToIntegers(char seperator)
{
	vector<int> vInts;
	vector<stringEx> items = this->SplitToStrings(seperator);
	for(size_t i = 0 ; i < items.size(); ++i)
	{
		stringEx item = items[i];
		int data = atoi(item.GetString().c_str());
		vInts.push_back(data);
	}
	return vInts;
}

void stringEx::TrimNoCharacter()
{

	int nLeft=0,nRight=0;
	if(m_str.length()<1)return;
	for(size_t i=0;i<m_str.length();i++)
	{
		char c=tolower(m_str.at(i));
		if((c<='9'&&c>='0')||(c>='a'&&c<='z'))
		{
			nLeft=i;
			break;
		}
	}
	m_str.erase(0,nLeft);
	for(size_t i=0;i<m_str.length();i++)
	{
		size_t pos=m_str.length()-i-1;
		char c=tolower(m_str.at(pos));
		
		if((c<='9'&&c>='0')||(c>='a'&&c<='z'))
		{
			nRight=m_str.length()-1-i;
			break;
		}		
	}
	m_str.erase(nRight+1,m_str.length()-nRight-1);
}

vector<stringEx> stringEx::SplitToStrings(char seperator)
{
	vector<stringEx> result;
	
	vector<int> posVector=this->GetSeperatorPos(seperator);	
	int first,second;
	if(posVector.size()<=1)
	{
		if(posVector.empty())
		{	//没有标记符，返回原始的整行
			result.push_back(*this);
			return result;
		}
		result.push_back(m_str.substr(0,posVector[0]));
		result.push_back(m_str.substr(posVector[0]+1,m_str.length()-posVector[0]-1));
	}
	else
	{
		for(size_t i=0;i<posVector.size()-1;i++)
			{
				first=posVector[i];
				second=posVector[i+1];
				if(first!=0&&i==0)
				{
					result.push_back(m_str.substr(0,first));
				}
				if(second-first==1)continue;//如果两个分隔符相邻，就跳过
				
				result.push_back(m_str.substr(first+1,second-first-1));

			}
			int len = m_str.size();
			if(second!=len-1)
				result.push_back(m_str.substr(second+1,len-1-second));
	}
	
	return result;
}
vector<int> stringEx::GetSeperatorPos(char seperator)
{
	vector<int> result;
	string temp=m_str;
	while(temp.length()>0)
	{
		
		size_t pos=temp.find_first_of(seperator);
		if(pos!=string::npos)
		{			
			if(result.size()>0)
			{
				int lastPos=0;
				lastPos=result[result.size()-1];
				int newPos=lastPos+1+pos;
				result.push_back(newPos);
				
			}
			else
			{
				result.push_back(pos);
				
			}
			temp.erase(0,pos+1);//把分割符以左（包括分割符）去掉
		}
		else
		{
			
			break;
		}
	}
	return result;
}
vector<int> stringEx::GetSeperatorPosEx(string seperator)
{
	vector<int> result;
	stringEx temp(m_str);
	while(temp.GetString().length()>0)
	{
		size_t pos=temp.FindStrings(seperator);
		if(pos!=string::npos)
		{

			if(result.size()>0)
			{
				int lastPos=0;
				lastPos=result[result.size()-1];
				int newPos=lastPos+seperator.length()+pos;
				result.push_back(newPos);

			}
			else
			{
				result.push_back(pos);

			}
			temp.Erase(0,pos+seperator.length());//把分割符以左（包括分割符）去掉
		}
		else
		{

			break;
		}
	}
	return result;
}
vector<stringEx> stringEx::SplitToStringsEx(string seperator)
{
	vector<stringEx> result;

	vector<int> posVector=this->GetSeperatorPosEx(seperator);	
	int first,second;
	int len=seperator.length();
	if(posVector.size()<=1)
	{
		if(posVector.empty())
		{	//没有标记符，返回原始的整行
			result.push_back(*this);
			return result;
		}
		result.push_back(m_str.substr(0,posVector[0]));
		result.push_back(m_str.substr(posVector[0]+len,m_str.length()-posVector[0]-len));
	}
	else
	{
		for(size_t i=0;i<posVector.size()-1;i++)
		{
			first=posVector[i];
			second=posVector[i+1];
			if(first!=0&&i==0)
			{
				result.push_back(m_str.substr(0,first));
			}
			if(second-first==1)continue;//如果两个分隔符相邻，就跳过

			result.push_back(m_str.substr(first+len,second-first-len));

		}
		if(second!=posVector.size()-1)
			result.push_back(m_str.substr(second+len,posVector.size()-len-second));
	}

	return result;
}

void stringEx::ReplaceCharacter(char srcChar, char newChar)
{
	int pos = m_str.find_first_of(srcChar);
	while (pos != -1)
	{
		m_str.at(pos) = newChar;
		pos = m_str.find_first_of(srcChar);
	}
}
void stringEx::Erase(size_t nOffset,size_t nLenght)
{
	m_str.erase(nOffset,nLenght);
}
void stringEx::Copy(string str)
{
	m_str=str;
}
string stringEx::GetString()
{
	return m_str;
}
double stringEx::ToDigital()
{
	double data = atof(m_str.c_str());
	return data;
}


stringEx::~stringEx(void)
{
}
