package plugins

import (
	"os"

	"gopkg.in/yaml.v3"
)

// MergeYAML merges two YAML documents. For conflicts, prefer second doc's values.
func MergeYAML(d1, d2 []byte) ([]byte, error) {
	var m1 map[string]interface{}
	var m2 map[string]interface{}

	if err := yaml.Unmarshal(d1, &m1); err != nil {
		m1 = map[string]interface{}{}
	}
	if err := yaml.Unmarshal(d2, &m2); err != nil {
		m2 = map[string]interface{}{}
	}
	merged := deepMergeYAML(m1, m2)
	return yaml.Marshal(merged)
}

func deepMergeYAML(a, b map[string]interface{}) map[string]interface{} {
	for k, v := range b {
		if am, ok := a[k].(map[string]interface{}); ok {
			if bm, ok2 := v.(map[string]interface{}); ok2 {
				a[k] = deepMergeYAML(am, bm)
				continue
			}
		}
		a[k] = v
	}
	return a
}

func readYAMLFile(path string) (map[string]interface{}, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return make(map[string]interface{}), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	err = yaml.Unmarshal(data, &m)
	if err != nil {
		return make(map[string]interface{}), nil
	}
	return m, nil
}
