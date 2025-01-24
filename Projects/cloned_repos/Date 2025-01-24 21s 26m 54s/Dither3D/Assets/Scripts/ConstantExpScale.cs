using UnityEngine;

public class ConstantExpScale : MonoBehaviour {

	public float scaleExp = 1.2f;

	void Update() {
		transform.localScale = Vector3.one * Mathf.Pow(scaleExp, Time.time);
	}
}
