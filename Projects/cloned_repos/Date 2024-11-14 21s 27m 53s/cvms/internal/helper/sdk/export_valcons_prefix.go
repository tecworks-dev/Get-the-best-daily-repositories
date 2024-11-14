package sdkhelper

import (
	"errors"
	"strings"
)

// NOTE: this is not cosmos-sdk native. so, I placed this function into uptime package
// parse valcons prefix with valoper address
func ExportBech32ValconsPrefix(bech32ValoperPrefix string) (string, error) {
	split := strings.Split(bech32ValoperPrefix, "1")

	ok := strings.Contains(split[0], "valoper")
	if ok {
		chainPrefix := strings.Split(split[0], "valoper")
		prefix := chainPrefix[0] + "valcons"
		return prefix, nil

	} else {
		switch split[0] {
		case "iva":
			prefix := "ica"
			return prefix, nil
		case "crocncl":
			prefix := "crocnclcons"
			return prefix, nil

		default:
			return "", errors.New("TODO")
		}
	}
}
