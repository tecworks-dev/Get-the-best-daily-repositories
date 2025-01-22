// (Optional) authentication. We'll just stub a basic token approach.
package server

import (
	"errors"
	"os"
)

// getServerRepoPath returns the path to the server's repo
func getServerRepoPath() (string, error) {
	// For example, use an environment variable or a well-known path.
	path := os.Getenv("EVO_SERVER_REPO")
	if path == "" {
		return "", errors.New("EVO_SERVER_REPO not set; cannot locate server repo")
	}
	return path, nil
}
