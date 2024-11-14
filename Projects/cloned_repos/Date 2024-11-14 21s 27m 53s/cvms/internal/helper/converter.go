package helper

import (
	"fmt"
	"strconv"
	"strings"
)

func HexaNumberToInteger(hexaString string) string {
	// replace 0x or 0X with empty String
	numberStr := strings.Replace(hexaString, "0x", "", -1)
	numberStr = strings.Replace(numberStr, "0X", "", -1)
	return numberStr
}

func ParsingfromHexaNumberBaseHexaDecimal(intString string) (uint64, error) {
	output, err := strconv.ParseUint(intString, 16, 64)
	if err != nil {
		fmt.Println(err)
		return 0, err
	}
	return output, nil
}

func ParsingfromHexaNumberBaseDecimal(intString string) (uint64, error) {
	output, err := strconv.ParseUint(intString, 10, 64)
	if err != nil {
		fmt.Println(err)
		return 0, err
	}
	return output, nil
}
