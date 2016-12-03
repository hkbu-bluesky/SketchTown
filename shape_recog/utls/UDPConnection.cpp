//#include "../stdafx.h"

#include <stdio.h>
#include <ws2tcpip.h>
#include "process.h"
#include "UDPConnection.h"

#pragma comment(lib,"Ws2_32.lib")
CUDPConnection::CUDPConnection(void)
{
	m_sockfd = 0;
	m_bReceivedInit = false;
	m_bSendInit = false;
}

CUDPConnection::~CUDPConnection(void)
{
}

/*******
//dwClientPort :本地监听端口，当它需要接收外来数据时，设置本端口，默认为0的话此UDP只向外部发送数据
*/
void CUDPConnection::InitConnection(const char *szServerIP, DWORD dwServerPort, DWORD dwClientPort /*= 0*/)
{

	
	if(m_sockfd != 0)
		CloseConnection();
	SOCKADDR_IN my_addr;

	WSADATA sock_data;
	WSAStartup(1,&sock_data);

	if(SOCKET_ERROR ==(m_sockfd = socket(AF_INET,SOCK_DGRAM,0)))
	{
		printf("Socket Error: %d\n",WSAGetLastError());
	}

	printf("Client Socket Created\n");

	memset(&sever_addr, 0, sizeof( SOCKADDR_IN));
	memset(&my_addr, 0, sizeof( SOCKADDR_IN));

	if(dwClientPort != 0)
	{
		
		my_addr.sin_family = AF_INET ; /*HBO */
		my_addr.sin_port = htons(dwClientPort); /* short,NBO */
		my_addr.sin_addr.s_addr = htonl( INADDR_ANY);/*auto-fill with my IP */

		if(SOCKET_ERROR == bind(m_sockfd ,(SOCKADDR *) &my_addr , sizeof(SOCKADDR))) 
		{
			printf("Bind Error : %d\n",WSAGetLastError());
		}
		m_bReceivedInit = true;
		printf("listen Socket Bind\n\n");
	}
	
	if(dwServerPort != 0)
	{
		sever_addr.sin_family = AF_INET ; /*HBO */
		sever_addr.sin_port = htons(dwServerPort); /* short,NBO */
		sever_addr.sin_addr.s_addr = inet_addr((LPSTR)LPCTSTR(szServerIP));
		m_bSendInit = true;
	}
}

void CUDPConnection::RunThread(NetThread thread, void *para)
{
	_beginthread(thread,0,para);
}

void CUDPConnection::CloseConnection()
{
	closesocket(m_sockfd);
	WSACleanup();
}

void CUDPConnection::Send(std::string strContent)
{

	memset(m_szSendBuff,0,sizeof(char)*1024);
	memcpy(m_szSendBuff,strContent.c_str(),strContent.length());		
	
	sendto(m_sockfd, m_szSendBuff,strContent.length() ,0, (SOCKADDR *) &(sever_addr), sizeof(SOCKADDR));

}








void CUDPConnection::Recv_Test()
{
	
	char szBuf[2048];

	while ( 1 )
	{		
		int len = recv(m_sockfd ,szBuf, 2048, 0);		
		if  (len > 0)
		{
			//parse data
		}		
		
	}	
}