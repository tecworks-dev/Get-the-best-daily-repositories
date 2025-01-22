package plugins

import (
	"encoding/json"
)

// MergeJSON merges two JSON documents, preferring the second doc if conflicts
func MergeJSON(d1, d2 []byte) ([]byte, error) {
	var m1 map[string]interface{}
	var m2 map[string]interface{}
	if err := json.Unmarshal(d1, &m1); err != nil {
		m1 = map[string]interface{}{}
	}
	if err := json.Unmarshal(d2, &m2); err != nil {
		m2 = map[string]interface{}{}
	}

	merged := deepMerge(m1, m2)
	return json.MarshalIndent(merged, "", "  ")
}

func deepMerge(a, b map[string]interface{}) map[string]interface{} {
	// for each key in b, overwrite or merge in a
	for k, v := range b {
		if am, ok := a[k].(map[string]interface{}); ok {
			if bm, ok2 := v.(map[string]interface{}); ok2 {
				a[k] = deepMerge(am, bm)
				continue
			}
		}
		a[k] = v
	}
	return a
}
