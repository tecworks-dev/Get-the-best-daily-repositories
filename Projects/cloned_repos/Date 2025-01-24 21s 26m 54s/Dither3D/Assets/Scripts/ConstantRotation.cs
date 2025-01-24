using UnityEngine;

public class ConstantRotation : MonoBehaviour {

	public float speed = 22.5f;
	public float phase = 0;
	public int axis = 1;

	void Update() {
		Vector3 rot = Vector3.zero;
		rot[axis] = Time.time * speed + phase;
		transform.localRotation = Quaternion.Euler(rot);
	}
}
