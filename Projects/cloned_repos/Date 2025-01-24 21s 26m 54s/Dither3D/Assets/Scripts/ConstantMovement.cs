using UnityEngine;

public class ConstantMovement : MonoBehaviour {

	public Vector3 velocity = new Vector3(1, 0, 0);

	void Update() {
		transform.Translate(velocity * Time.deltaTime, Space.World);
	}
}
