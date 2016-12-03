using UnityEngine;
using System.Collections;

//this script used to control cartoon element to move on the screen
//after it moves out the screen, destroy it
public class CartoonElement : MonoBehaviour {

	// Use this for initialization
	float speedAccScale = 1;
	float startAccTime;//time to start to moving faster
	Camera cam;
	void Start () {
		cam = Camera.main;
		float ptx = Random.Range (Screen.width*0.1f, Screen.width*0.2f);
		float pty = Random.Range (Screen.height*0.2f, Screen.height*0.8f);
		Vector3 ptxyz = new Vector3 (ptx, pty, -cam.transform.position.z);


		Vector3 worldXYZ = cam.ScreenToWorldPoint (ptxyz);
		Debug.Log (ptxyz);
		Debug.Log (worldXYZ);
		transform.position = worldXYZ;
		startAccTime =-10;
	}

	// Update is called once per frame
	void Update () {
		float randVelX = Random.Range (0, 0.4f);
		float randVelY = Random.Range (0, 0.2f);
		Vector3 vel = new Vector3 (randVelX, randVelY, 0);

		float timeDiff = Time.time - startAccTime;
		//Debug.Log (Time.time + "starttime" + startAccTime);
		if (timeDiff > 0 && timeDiff < 1f) {

			//Debug.Log ("src vel" + vel+ "acc"+speedAccScale);
			vel = vel * speedAccScale;
			//Debug.Log ("acc vel" + vel);
			//Debug.Log ("acc......... ");
		} 

		transform.Translate (vel * Time.deltaTime, Space.World);

		//check if the cartoon elements move out of the screen,
		Vector3 vpt = cam.WorldToViewportPoint(transform.position);
		if (vpt.x < 0 || vpt.x > 1 || vpt.y < 0 || vpt.y > 1) {
			//out of screen , destroy it
			Destroy(this);
			//Debug.Log("out of screen");
		}
	}
	public void notifyClick()
	{
		speedAccScale = 10;
		startAccTime = Time.time;

	}
}
