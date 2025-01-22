package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func lockRepo(repoPath string) error {
    return lockResource(repoPath, "")
}

func unlockRepo(repoPath string) error {
    return unlockResource(repoPath, "")
}

// If you still want subdir locks, pass subdir
func lockSubdir(repoPath, subdir string) error {
    return lockResource(repoPath, subdir)
}

func unlockSubdir(repoPath, subdir string) error {
    return unlockResource(repoPath, subdir)
}

func lockResource(repoPath, resource string) error {
    // resource might be "" or "objects"
    lockFile := getLockFilePath(repoPath, resource)
    // We attempt to create a lock file
    f, err := os.OpenFile(lockFile, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
    if err != nil {
        return fmt.Errorf("Cannot acquire lock on '%s': %v", resource, err)
    }
    defer f.Close()

    // write metadata
    fmt.Fprintf(f, "PID=%d\nTIME=%d\nRESOURCE=%s\n", os.Getpid(), time.Now().Unix(), resource)
    return nil
}

func unlockResource(repoPath, resource string) error {
    lockFile := getLockFilePath(repoPath, resource)
    if _, err := os.Stat(lockFile); os.IsNotExist(err) {
        // not locked
        return nil
    }
    // Optionally check if the lock belongs to the current process
    // For now, we just remove it
    return os.Remove(lockFile)
}

func getLockFilePath(repoPath, resource string) string {
    lockName := "LOCK"
    if resource != "" {
        lockName = "LOCK_" + strings.ReplaceAll(resource, "/", "_")
    }
    return filepath.Join(repoPath, EvoDir, lockName)
}

// checkLockFile => checks if there's a stale lock
func checkLockFile(repoPath string) error {
    // We'll just check the main repo lock
    lockFile := getLockFilePath(repoPath, "")
    info, err := os.Stat(lockFile)
    if os.IsNotExist(err) {
        return nil
    }
    if err != nil {
        return err
    }
    // read contents
    data, err := os.ReadFile(lockFile)
    if err != nil {
        return fmt.Errorf("cannot read lock file: %v", err)
    }
    lines := strings.Split(string(data), "\n")
    var pid int
    for _, line := range lines {
        if strings.HasPrefix(line, "PID=") {
            pid, _ = strconv.Atoi(strings.TrimPrefix(line, "PID="))
        }
    }
    // check if pid is alive
    if pid > 0 {
        // On Unix, we can do a signal 0 check
        err := syscall.Kill(pid, 0)
        if err == nil {
            return fmt.Errorf("Repo locked by process %d (lock file mod time: %v)", pid, info.ModTime())
        }
        // if err != nil => pid is stale => remove
    }
    os.Remove(lockFile)
    return nil
}
