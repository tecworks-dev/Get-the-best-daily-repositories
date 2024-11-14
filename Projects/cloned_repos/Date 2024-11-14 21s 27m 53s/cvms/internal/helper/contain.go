package helper

import "strings"

// Contains checks if the slice contains the exact string
func Contains(s []string, str string) bool {
	for _, v := range s {
		if strings.Contains(str, v) {
			return true
		}
	}
	return false
}
