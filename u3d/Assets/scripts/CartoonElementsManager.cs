using UnityEngine;
using System.Collections;

public class CartoonElementsManager : MonoBehaviour {

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {

		if (Input.GetMouseButtonUp (0)) {
			//left button clicked,check if it has clicked on cartoon elements
			Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
			RaycastHit hitInfo;
			if (Physics.Raycast (ray, out hitInfo)) {
				Debug.DrawLine (ray.origin, hitInfo.point);
				GameObject targetObj = hitInfo.collider.gameObject;
				CartoonElement cartoon = targetObj.GetComponent<CartoonElement> ();
				if (cartoon != null) {
					cartoon.notifyClick ();
					Debug.Log ("Click");
				}

			}


		}

	}
}
