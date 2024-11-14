package helper

import "strings"

// parse to pg schema name
func ParseToSchemaName(chainID string) string {
	var schema string
	schema = strings.Replace(chainID, "-", "_", -1)
	schema = strings.Replace(schema, ".", "_", -1)

	return schema
}
