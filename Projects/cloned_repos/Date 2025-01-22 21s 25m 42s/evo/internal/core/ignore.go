package core

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
)

// LoadIgnorePatterns reads "evo-ignore" from the repo root if it exists and returns each non-empty line as a pattern.
func LoadIgnorePatterns(repoRoot string) ([]string, error) {
	ignoreFile := filepath.Join(repoRoot, "evo-ignore")
	var patterns []string

	f, err := os.Open(ignoreFile)
	if os.IsNotExist(err) {
		return patterns, nil // no file => no patterns
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		patterns = append(patterns, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return patterns, nil
}

// MatchesIgnorePatterns checks if the given file path matches any of the ignore patterns.
func MatchesIgnorePatterns(relPath string, patterns []string) bool {
	for _, pattern := range patterns {
		// Using doublestar for globbing (supprts **, etc.)
		// Example: pattern = "build/**" or "*.log".
		matched, err := doublestar.PathMatch(pattern, relPath)
		if err == nil && matched {
			return true
		}
	}
	return false
}
