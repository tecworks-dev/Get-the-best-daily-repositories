package internal

import (
	"encoding/json"
	"log"
	"os"
	"strings"

	"github.com/haatos/goshipit/internal/model"
)

var ComponentCodeMap model.ComponentCodeMap
var ComponentExampleCodeMap model.ComponentExampleCodeMap

func init() {
	getComponentCodeMap()
	getComponentExampleCodeMap()
}

func codeSliceToMarkdown(s []string) string {
	if s == nil {
		return ""
	}
	n := make([]string, len(s)+2)
	n[0] = "```go"
	n[len(s)+2-1] = "```"
	copy(n[1:len(s)+1], s)
	return strings.Join(n, "\n")
}

func getComponentCodeMap() {
	b, err := os.ReadFile("generated/component_code_map.json")
	if err != nil {
		log.Fatal(err)
	}

	if err := json.Unmarshal(b, &ComponentCodeMap); err != nil {
		log.Fatal(err)
	}

	for k := range ComponentCodeMap {
		for i := range ComponentCodeMap[k] {
			ComponentCodeMap[k][i].Label = SnakeCaseToCapitalized(ComponentCodeMap[k][i].Name)
			ComponentCodeMap[k][i].CodeMarkdown = codeSliceToMarkdown(ComponentCodeMap[k][i].Code)
		}
	}
}

func getComponentExampleCodeMap() {
	b, err := os.ReadFile("generated/component_example_code_map.json")
	if err != nil {
		log.Fatal(err)
	}

	if err := json.Unmarshal(b, &ComponentExampleCodeMap); err != nil {
		log.Fatal(err)
	}

	for k := range ComponentExampleCodeMap {
		for i := range ComponentExampleCodeMap[k] {
			ComponentExampleCodeMap[k][i].Label = SnakeCaseToCapitalized(ComponentExampleCodeMap[k][i].Name)
			ComponentExampleCodeMap[k][i].CodeMarkdown = codeSliceToMarkdown(ComponentExampleCodeMap[k][i].Code)
			ComponentExampleCodeMap[k][i].HandlerMarkdown = codeSliceToMarkdown(ComponentExampleCodeMap[k][i].Handler)
		}
	}
}

func SnakeCaseToCapitalized(s string) string {
	b := []byte(s)
	for i := range b {
		if i == 0 || (i > 0 && b[i-1] == ' ') {
			b[i] = b[i] - ('a' - 'A')
		}
		if b[i] == '_' {
			b[i] = ' '
		}
	}
	return string(b)
}
