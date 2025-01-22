package plugins

import (
	"os"
	"path/filepath"
	"plugin"
	"sort"
	"strings"
)

var loadedPlugins []MergePlugin

func LoadAllPlugins(repoPath string) error {
	// load from ~/.evo/plugins
	home, _ := os.UserHomeDir()
	globalDir := filepath.Join(home, ".evo", "plugins")
	loadPluginsFromDir(globalDir)
	// load from .evo/plugins in the repo
	localDir := filepath.Join(repoPath, ".evo", "plugins")
	loadPluginsFromDir(localDir)

	// sort by priority descending
	sort.Slice(loadedPlugins, func(i, j int) bool {
		return loadedPlugins[i].Priority() > loadedPlugins[j].Priority()
	})
	return nil
}

func loadPluginsFromDir(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(e.Name(), ".so") {
			pluginPath := filepath.Join(dir, e.Name())
			p, err := plugin.Open(pluginPath)
			if err != nil {
				continue
			}
			sym, err := p.Lookup("Plugin")
			if err != nil {
				continue
			}

			mp, ok := sym.(MergePlugin)
			if !ok {
				continue
			}
			loadedPlugins = append(loadedPlugins, mp)
		}
	}
}
