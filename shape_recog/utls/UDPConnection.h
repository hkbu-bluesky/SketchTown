#pragma once

#include <winsock2.h>
#include <string>
#include <vector>

typedef void (*NetThread)(void *para);



class CUDPConnection
{
public:
	CUDPConnection(void);
	~CUDPConnection(void);

	int m_sockfd;
	SOCKADDR_IN sever_addr ;
	
	char m_szSendBuff[2048];

	bool m_bSendInit;
	bool m_bReceivedInit;
	

	void InitConnection(const char *szServerIP,DWORD dwServerPort,DWORD dwClientPort = 0);
	void CloseConnection();
	void RunThread(NetThread thread,void *para);//一般用于监听线程

	void  CUDPConnection::Send(std::string strContent);
	
	void Recv_Test();	//接收端数据解析示例
	void InitSkelNames();
	
};
