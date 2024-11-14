package helper

import (
	"regexp"
)

func ParseChainName(jobName string) (string, error) {
	reg, err := regexp.Compile(`-[\w]*net`)
	if err != nil {
		return "", err
	}
	indexes := reg.FindAllStringIndex(jobName, -1)

	if len(indexes) == 0 {
		return jobName, nil
	}

	for idx, element := range indexes {
		if idx == 0 {
			return jobName[:element[0]], nil
		}
	}

	return jobName, nil
}
