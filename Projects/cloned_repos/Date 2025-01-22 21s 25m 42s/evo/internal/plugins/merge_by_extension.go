package plugins

import (
	"path/filepath"
	"strings"
)

// MergeByExtension decides how to merge two files based on extension
func MergeByExtension(filePath string, b1, b2 []byte) ([]byte, error) {
	ext := strings.ToLower(filepath.Ext(filePath))

	// 1. If any dynamically loaded plugin supports this extension, try them
	for _, pl := range loadedPlugins {
		for _, ex := range pl.SupportedExtensions() {
			if ex == ext {
				return pl.Merge(b1, b2)
			}
		}
	}

	// 2. If no plugin, fallback to built-in logic
	switch ext {
	case ".json":
		return MergeJSON(b1, b2)
	case ".yaml", ".yml":
		return MergeYAML(b1, b2)
	}
	return b2, nil
}
