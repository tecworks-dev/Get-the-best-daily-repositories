// Client-side logic for pushing/pulling from Evo's HTTP server.
package core

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// Pull: fetch list of remote commits, request missing objects
func Pull(repoPath, remoteURL string) error {
	lockRepo(repoPath)
	defer unlockRepo(repoPath)

	// 1. GET /pull to get list of all remote commit hashes
	resp, err := http.Get(remoteURL + "/pull")
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("remote pull error: %s", resp.Status)
	}
	var remoteHashes []string
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	json.Unmarshal(body, &remoteHashes)

	// 2. Compare with local objecgts
	evoPath := filepath.Join(repoPath, EvoDir, "objects")
	toFetch := []string{}
	for _, h := range remoteHashes {
		objPath := filepath.Join(evoPath, h+".json")
		if _, err := os.Stat(objPath); os.IsNotExist(err) {
			toFetch = append(toFetch, h)
		}
	}

	// 3. Fetch each missing object from /objects/<hash>
	for _, h := range toFetch {
		objResp, err := http.Get(remoteURL + "/objects/" + h)
		if err != nil {
			return err
		}
		if objResp.StatusCode != http.StatusOK {
			continue
		}
		objData, _ := io.ReadAll(objResp.Body)
		objResp.Body.Close()

		// Write to local
		os.WriteFile(filepath.Join(evoPath, h+".json"), objData, 0644)
	}
	return nil
}

func Push(repoPath, remoteURL string) error {
	lockRepo(repoPath)
	defer unlockRepo(repoPath)

	// 1. Gather local commits
	commits, err := GetCommitLog(repoPath)
	if err != nil {
		return err
	}
	if len(commits) == 0 {
		// nothing to push
		return nil
	}
	// 2. GET remote commits
	rResp, err := http.Get(remoteURL + "/pull")
	if err != nil {
		return err
	}
	var remoteHashes []string
	b, _ := io.ReadAll(rResp.Body)
	rResp.Body.Close()
	json.Unmarshal(b, &remoteHashes)
	remoteMap := make(map[string]bool)
	for _, h := range remoteHashes {
		remoteMap[h] = true
	}

	evoPath := filepath.Join(repoPath, EvoDir, "objects")
	// 3. For each local commit not in remoteMap, POST it
	for _, c := range commits {
		if !remoteMap[c.Hash] {
			// read the commit object from disk
			path := filepath.Join(evoPath, c.Hash+".json")
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			// base64 encode
			enc := base64.StdEncoding.EncodeToString(data)
			obj := map[string]string{
				"hash": c.Hash,
				"data": enc,
			}
			objBytes, _ := json.Marshal(obj)
			req, _ := http.NewRequest("POST", remoteURL+"/push", bytes.NewReader(objBytes))
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return err
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("push error: %s", resp.Status)
			}
		}
	}

	// You could also push references (HEAD, workspace refs), etc.
	return nil
}

// Optionally store a "remote" ref in .evo/config
func GetRemoteURL(repoPath string) (string, error) {
	configPath := filepath.Join(repoPath, EvoDir, "config")
	b, err := os.ReadFile(configPath)
	if err != nil {
		return "", nil
	}
	// parse line by line for "remote = <url>"
	lines := bytes.Split(b, []byte("\n"))
	for _, line := range lines {
		str := string(line)
		if len(str) > 0 && str[:6] == "remote" {
			parts := strings.SplitN(str, "=", 2)
			if len(parts) == 2 {
				return strings.TrimSpace(parts[1]), nil
			}
		}
	}
	return "", nil
}

func SetRemoteURL(repoPath, url string) error {
	configPath := filepath.Join(repoPath, EvoDir, "config")
	c := fmt.Sprintf("remote = %s\n", url)
	return os.WriteFile(configPath, []byte(c), 0644)
}

// We also define a quick sanity check on object integrity:
func verifyObjectIntegrity(data []byte) bool {
	// We can re-hash the JSON minus the "hash" field, or simply trust the JSON's "hash" field.
	// Omitted for brevity; but you'd parse the object, re-hash, compare to stored "hash".
	return true
}

func computeHash(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}
