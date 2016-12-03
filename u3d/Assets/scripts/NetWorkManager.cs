using UnityEngine;
using System.Collections;

using System.Net;  
using System.Net.Sockets;  
using System.Text;  
using System.Threading;  

public class NetWorkManager : MonoBehaviour {


	Socket socket;	//socket instance
	IPEndPoint ipEnd; //listening ip port
	EndPoint clientEnd; //client sending data to here
	string recvStr; //received string

	byte[] recvData=new byte[1024]; 

	int recvLen; //received data length
	Thread connectThread; //receive thread



	void InitSocket()
	{
		
		ipEnd=new IPEndPoint(IPAddress.Any,8001);
		socket=new Socket(AddressFamily.InterNetwork,SocketType.Dgram,ProtocolType.Udp);
		socket.Bind(ipEnd);

		IPEndPoint sender=new IPEndPoint(IPAddress.Any,0);  
		clientEnd=(EndPoint)sender;  

		connectThread=new Thread(new ThreadStart(SocketReceiveThread));
		connectThread.Start();
	}




	void SocketReceiveThread()
	{
		
		while(true)
		{
			
			recvData=new byte[1024];
			recvLen=socket.ReceiveFrom(recvData,ref clientEnd);
			Debug.Log("message from: "+clientEnd.ToString()); 

			recvStr=Encoding.UTF8.GetString(recvData,0,recvLen);
			Debug.Log(recvStr);
		}
	}

	//close udp listen network
	void SocketQuit()
	{
		//close thread
		if(connectThread!=null)
		{
			connectThread.Interrupt();
			connectThread.Abort();
		}
		//close socket
		if(socket!=null)
			socket.Close();
		
	}

	// Use this for initialization
	void Start()
	{
		InitSocket(); 
	}


	// Update is called once per frame
	void Update()
	{

	}

	void OnApplicationQuit()
	{
		SocketQuit();
	}
}
