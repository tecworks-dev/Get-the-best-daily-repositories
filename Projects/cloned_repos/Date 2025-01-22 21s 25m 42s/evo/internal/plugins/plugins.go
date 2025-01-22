package plugins

type MergePlugin interface {
	// Priority: if multiple plugins claim the same extension
	Priority() int
	// SupportedExtensions: which file extensions it can handle
	SupportedExtensions() []string
	// Merge: merges two byte slices, returns merged or error
	Merge(ours, theirs []byte) ([]byte, error)
}
