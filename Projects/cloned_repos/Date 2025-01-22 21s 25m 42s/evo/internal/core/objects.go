package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Blob struct {
	Hash string `json:"hash"`
	// Possibly store compression info or data offset. For now, store the actual file data in .evo/objects/blobs
}

type Tree struct {
	Hash  string            `json:"hash"`
	Files map[string]string `json:"files"` // filepath -> blobHash
}

// Commit reprensents a single commit in Evo
type Commit struct {
	Hash        string    `json:"hash"`
	Message     string    `json:"message"`
	Author      string    `json:"author"`
	Timestamp   time.Time `json:"timestamp"`
	Parents     []string  `json:"parents,omitempty"`
	TreeHash    string    `json:"tree_hash,omitempty"`
	Signature   string    `json:"signature,omitempty"`
	MergeDetail string    `json:"merge_detail,omitempty"` // optional, store conflict resolutions, etc.
}

func CreateCommit(repoPath, message string, sign, partial bool, author UserConfig) (string, error) {
	lockRepo(repoPath)
	defer unlockRepo(repoPath)

	evoPath := filepath.Join(repoPath, EvoDir)
	activeRef, _ := ReadRef(evoPath, ActiveRef)

	// Build the tree from working dir or staging if partial
	treeHash, err := buildTreeFromWorking(repoPath, partial)
	if err != nil {
		return "", err
	}

	commit := &Commit{
		Message:   message,
		Author:    fmt.Sprintf("%s <%s>", author.Name, author.Email),
		Timestamp: time.Now().UTC(),
		Parents:   []string{},
		TreeHash:  treeHash,
	}

	if activeRef != "" {
		commit.Parents = append(commit.Parents, activeRef)
	}

	raw := fmt.Sprintf("%s|%s|%v|%s", commit.Message, commit.Author, commit.Timestamp.UnixNano(), commit.TreeHash)
	for _, p := range commit.Parents {
		raw += "|" + p
	}
	sum := sha256.Sum256([]byte(raw))
	commit.Hash = hex.EncodeToString(sum[:])

	// sign if requested
	if sign {
		sig, err := signCommit(repoPath, commit.Hash, author)
		if err != nil {
			return "", err
		}
		commit.Signature = sig
	}

	if err := writeCommitObject(evoPath, commit); err != nil {
		return "", err
	}
	// Update ACTIVE
	if err := WriteRef(evoPath, ActiveRef, commit.Hash); err != nil {
		return "", err
	}

	return commit.Hash, nil
}

func GetCommitLog(repoPath string) ([]Commit, error) {
	lockRepo(repoPath)
	defer unlockRepo(repoPath)

	evoPath := filepath.Join(repoPath, EvoDir)
	activeRef, _ := ReadRef(evoPath, ActiveRef)
	if activeRef == "" {
		return nil, nil
	}

	// We'll do a BFS/DFS collecting commits, then sort by time desc.
	visited := make(map[string]bool)
	var commits []Commit
	queue := []string{activeRef}

	for len(queue) > 0 {
		h := queue[0]
		queue = queue[1:]
		if visited[h] {
			continue
		}
		visited[h] = true

		c, err := ReadCommitObject(evoPath, h)
		if err != nil {
			continue
		}
		commits = append(commits, *c)
		for _, p := range c.Parents {
			if p != "" {
				queue = append(queue, p)
			}
		}
	}

	// Sort by commit.Timestamp desc
	sort.Slice(commits, func(i, j int) bool {
		return commits[i].Timestamp.After(commits[j].Timestamp)
	})
	return commits, nil
}

func RevertCommit(repoPath, commitHash string) (string, error) {
	lockRepo(repoPath)
	defer unlockRepo(repoPath)

	evoPath := filepath.Join(repoPath, EvoDir)
	activeRef, _ := ReadRef(evoPath, ActiveRef)
	if activeRef == "" {
		return "", fmt.Errorf("no ACTIVE to revert onto")
	}

	targetCommit, err := ReadCommitObject(evoPath, commitHash)
	if err != nil {
		return "", fmt.Errorf("commit not found: %s", commitHash)
	}

	// To revert, we need to invert the changes introduced by targetCommit
	// Compare targetCommit's tree to its parent (or empty if no parent).
	if len(targetCommit.Parents) == 0 {
		// no parent, means it introduced entire tree from scratch.
		// revert = removing everything
	}
	var parentHash string
	if len(targetCommit.Parents) > 0 {
		parentHash = targetCommit.Parents[0]
	}

	// We'll create a new tree that is basically the parent's tree
	// applied on top of current ACTIVE's tree
	// Actually, we want to revert the changes from targetCommit -> so we apply the "diff" in reverse to ACTIVE's tree

	// 1. Get ACTIVE tree
	activeCommit, err := ReadCommitObject(evoPath, activeRef)
	if err != nil {
		return "", err
	}
	activeTree, err := ReadTreeObject(evoPath, activeCommit.TreeHash)
	if err != nil {
		return "", err
	}

	// 2. Get parent tree of target commit
	var parentTree *Tree
	if parentHash != "" {
		pc, err := ReadCommitObject(evoPath, parentHash)
		if err != nil {
			return "", err
		}
		pt, err := ReadTreeObject(evoPath, pc.TreeHash)
		if err != nil {
			return "", err
		}
		parentTree = pt
	} else {
		// If no parent, revert means removing everything that was introduced.
		parentTree = &Tree{
			Files: make(map[string]string),
		}
	}

	targetTree, err := ReadTreeObject(evoPath, targetCommit.TreeHash)
	if err != nil {
		return "", err
	}

	// 3. Compute diff (parentTree -> targetTree) = introduced changes
	introduced, removed, _ := diffTrees(parentTree, targetTree)

	// 4. We want to reverse it on ACTIVE, so remove introduced files and re-introduce removed files.
	newTree := CloneTree(activeTree)

	// Remove introduced files if they match in ACTIVE
	for f := range introduced {
		// if ACTIVE has the same file in the same blob, remove it
		if newTree.Files[f] == targetTree.Files[f] {
			delete(newTree.Files, f)
		}
	}
	// Re-introduce removed files
	for f, blobHash := range removed {
		// only restore if ACTIVE doesn't have it
		if _, exists := newTree.Files[f]; !exists {
			newTree.Files[f] = blobHash
		}
	}

	// 5. Write new tree
	newTreeHash, err := writeTreeObject(evoPath, newTree)
	if err != nil {
		return "", err
	}

	revertMessage := fmt.Sprintf("Revert commit %s\n\nOriginal message: %s", commitHash, targetCommit.Message)
	revertCommit := &Commit{
		Message:   revertMessage,
		Author:    activeCommit.Author, // or current user.
		Timestamp: time.Now().UTC(),
		Parents:   []string{activeRef},
		TreeHash:  newTreeHash,
	}

	raw := fmt.Sprintf("%s|%s|%v|%s", revertCommit.Message, revertCommit.Author, revertCommit.Timestamp.UnixNano(), revertCommit.TreeHash)
	for _, p := range revertCommit.Parents {
		raw += "|" + p
	}
	sum := sha256.Sum256([]byte(raw))
	revertCommit.Hash = hex.EncodeToString(sum[:])

	if err := writeCommitObject(evoPath, revertCommit); err != nil {
		return "", err
	}
	if err := WriteRef(evoPath, ActiveRef, revertCommit.Hash); err != nil {
		return "", err
	}
	return revertCommit.Hash, nil
}

// For merges, see merges.go.
// For references (readRef/writeRef), see below:

// ReadCommitObject loads a commit from .evo/objects/<hash>.json
func ReadCommitObject(evoPath, hash string) (*Commit, error) {
	if hash == "" {
		return nil, fmt.Errorf("empty commit hash")
	}
	p := filepath.Join(evoPath, "objects", hash+".json")
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var c Commit
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

// writeCommitObject writes a commit to disk
func writeCommitObject(evoPath string, c *Commit) error {
	objPath := filepath.Join(evoPath, "objects", c.Hash+".json")
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(objPath, data, 0644)
}

// ReadTreeObject loads a tree from .evo/objects/<treeHash>.json
func ReadTreeObject(evoPath, treeHash string) (*Tree, error) {
	if treeHash == "" {
		return &Tree{Files: make(map[string]string)}, nil
	}
	p := filepath.Join(evoPath, "objects", treeHash+".json")
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var t Tree
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// writeTreeObject writes the Tree struct to disk, returns the new hash
func writeTreeObject(evoPath string, t *Tree) (string, error) {
	// compute hash from Files map
	var sb strings.Builder
	keys := make([]string, 0, len(t.Files))
	for k := range t.Files {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		sb.WriteString(k)
		sb.WriteString(":")
		sb.WriteString(t.Files[k])
		sb.WriteRune('\n')
	}
	sum := sha256.Sum256([]byte(sb.String()))
	t.Hash = hex.EncodeToString(sum[:])

	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return "", err
	}
	objPath := filepath.Join(evoPath, "objects", t.Hash+".json")
	if err := os.WriteFile(objPath, data, 0644); err != nil {
		return "", err
	}
	return t.Hash, nil
}

// ReadRef / writeRef
func ReadRef(evoPath, refName string) (string, error) {
	if !strings.Contains(refName, "/") && refName != ActiveRef {
		refName = "refs/" + refName
	}
	refPath := filepath.Join(evoPath, refName)
	b, err := os.ReadFile(refPath)
	if err != nil {
		return "", nil
	}
	return strings.TrimSpace(string(b)), nil
}

func WriteRef(evoPath, refName, val string) error {
	if !strings.Contains(refName, "/") && refName != ActiveRef {
		refName = "refs/" + refName
	}
	p := filepath.Join(evoPath, refName)
	os.MkdirAll(filepath.Dir(p), 0755)
	return os.WriteFile(p, []byte(val), 0644)
}

// Helpers

func diffTrees(base, target *Tree) (added map[string]string, removed map[string]string, changed map[string][2]string) {
	added = make(map[string]string)
	removed = make(map[string]string)
	changed = make(map[string][2]string)

	// any file in target but not in base => added
	// any file in base but not in target => removed
	// if file in both but different hash => changed
	for f, bh := range target.Files {
		if bbh, ok := base.Files[f]; !ok {
			added[f] = bh
		} else if bbh != bh {
			changed[f] = [2]string{bbh, bh}
		}
	}
	for f, bh := range base.Files {
		if _, ok := target.Files[f]; !ok {
			removed[f] = bh
		}
	}
	return
}

func CloneTree(src *Tree) *Tree {
	cp := &Tree{
		Files: make(map[string]string),
	}
	for k, v := range src.Files {
		cp.Files[k] = v
	}
	return cp
}
