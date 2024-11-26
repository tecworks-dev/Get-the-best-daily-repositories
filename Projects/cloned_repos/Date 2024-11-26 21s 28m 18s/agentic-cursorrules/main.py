from pathlib import Path
from typing import List, Set
import yaml
from gitignore_parser import parse_gitignore
import time
import argparse
import sys

class ProjectTreeGenerator:
    def __init__(self, project_root: Path):
        """
        Initializes the generator with gitignore-based exclusions and the project root.
        """
        self.project_root = project_root
        
        # Load config from YAML
        config_path = project_root / '.agentic-cursorrules' / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set file extensions from config
        self.INCLUDE_EXTENSIONS: Set[str] = set(config.get('include_extensions', []))
        self.IMPORTANT_DIRS = set(config.get('important_dirs', []))
        self.EXCLUDE_DIRS = set(config.get('exclude_dirs', []))
        
        # Initialize gitignore matcher
        gitignore_path = project_root / '.gitignore'
        if gitignore_path.exists():
            self.matches = parse_gitignore(gitignore_path)
        else:
            # Create temporary gitignore with exclude_dirs from config
            temp_gitignore = project_root / '.temp_gitignore'
            with open(temp_gitignore, 'w') as f:
                f.write('\n'.join(f'{dir}/' for dir in self.EXCLUDE_DIRS))
            self.matches = parse_gitignore(temp_gitignore)
            temp_gitignore.unlink()

    def generate_tree(self, directory: Path, file_types: List[str] = None, max_depth: int = 3, skip_dirs: Set[str] = None, config_paths: Set[str] = None) -> List[str]:
        """
        Generates a visual tree representation of the directory structure.
        
        Args:
            directory: Directory to generate tree for
            file_types: List of file extensions to include
            max_depth: Maximum depth to traverse
            skip_dirs: Set of directory paths to skip (already processed in parent trees)
            config_paths: Set of all paths from config.yaml for exclusion checking
        """
        tree_lines = []
        skip_dirs = skip_dirs or set()
        config_paths = config_paths or set()

        def _generate_tree(dir_path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return

            items = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_file(), x.name))
            for i, item in enumerate(items):
                rel_path = str(item.relative_to(self.project_root))
                
                if (item.name in self.EXCLUDE_DIRS or 
                    self.matches(str(item)) or 
                    rel_path in skip_dirs or
                    (item.is_dir() and any(cp.startswith(rel_path) for cp in config_paths))):
                    print(f"Skipping {rel_path}")  # Debug print
                    continue

                is_last = i == len(items) - 1
                display_path = item.name

                if item.is_dir():
                    tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{display_path}/")
                    _generate_tree(item, prefix + ('    ' if is_last else '│   '), depth + 1)
                elif item.is_file():
                    extensions_to_check = file_types if file_types else self.INCLUDE_EXTENSIONS
                    if any(item.name.endswith(ext) for ext in extensions_to_check):
                        tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{display_path}")

            return tree_lines

        return _generate_tree(directory)

    def find_focus_dirs(self, directory: Path, focus_dirs: List[str]) -> List[Path]:
        """
        Finds directories matching the focus names, handling nested paths and special cases.
        """
        found_dirs = []
        print("\nDebug - Input focus_dirs:", focus_dirs)
        
        # First, normalize all focus dirs and preserve special paths
        normalized_focus_dirs = []
        for fd in focus_dirs:
            # Preserve paths with double underscores
            if '__' in fd:
                normalized_focus_dirs.append(Path(fd))
            # Convert single underscores to paths
            elif '_' in fd and '/' not in fd:
                normalized_focus_dirs.append(Path(fd.replace('_', '/')))
            else:
                normalized_focus_dirs.append(Path(fd))
        
        print("Debug - Normalized dirs:", normalized_focus_dirs)
        
        # Sort by path depth (shortest first) to handle parent folders first
        normalized_focus_dirs.sort(key=lambda p: len(p.parts))
        print("Debug - Sorted dirs:", normalized_focus_dirs)
        
        for focus_path in normalized_focus_dirs:
            print(f"\nDebug - Processing: {focus_path}")
            
            # New skip condition: only skip if the exact path is already found
            if str(focus_path) in [str(found.relative_to(directory)) for found in found_dirs]:
                print(f"Debug - Skipping {focus_path} as it's already processed")
                continue
            
            # Handle both direct paths and nested paths
            target_path = directory / focus_path
            print(f"Debug - Looking for path: {target_path}")
            if target_path.exists() and target_path.is_dir():
                print(f"Debug - Found directory: {target_path}")
                found_dirs.append(target_path)
                continue
            
            # If not found directly, search one level deeper
            for item in directory.iterdir():
                if item.is_dir():
                    nested_path = item / focus_path.name
                    if nested_path.exists() and nested_path.is_dir():
                        print(f"Debug - Found nested directory: {nested_path}")
                        found_dirs.append(nested_path)
                        break
        
        print("\nDebug - Final found_dirs:", found_dirs)
        return found_dirs

def generate_agent_files(focus_dirs: List[str], agentic_dir: Path):
    """
    Generates agent-specific markdown files for each focus directory.
    """
    root_dir = agentic_dir.parent
    created_files = set()

    for dir_path in focus_dirs:
        try:
            # Convert string to Path if it's not already
            if isinstance(dir_path, str):
                dir_path = Path(dir_path)
            
            # Handle both Path objects and strings safely
            dir_name = dir_path.name if isinstance(dir_path, Path) else dir_path
            parent_path = dir_path.parent if isinstance(dir_path, Path) else Path(dir_path).parent
            parent_name = parent_path.name if parent_path != root_dir else None
            
            # Generate the agent file name based on the path structure
            if str(dir_path).count('/') > 0:
                parts = str(dir_path).split('/')
                agent_name = f"agent_{parts[0]}_{parts[-1]}.md"
            elif parent_name and not dir_name.startswith('__'):
                agent_name = f"agent_{parent_name}_{dir_name}.md"
            else:
                agent_name = f"agent_{dir_name}.md"
            
            if agent_name in created_files:
                continue
                
            # Use the last part of the path for the tree file name
            tree_file = agentic_dir / f'tree_{dir_path.name}.txt'
            tree_content = ""
            if tree_file.exists():
                with open(tree_file, 'r', encoding='utf-8') as f:
                    tree_content = f.read()
            
            # Generate appropriate directory description
            if '/' in str(dir_path):
                dir_description = f"the {dir_path.name} directory within {dir_path.parent.name}"
            elif parent_name:
                dir_description = f"the {dir_name} directory within {parent_name}"
            else:
                dir_description = f"the {dir_name} portion"
            
            agent_content = f"""You are an agent that specializes in {dir_description} of this project. Your expertise and responses should focus specifically on the code and files within this directory structure:

{tree_content}

When providing assistance, only reference and modify files within this directory structure. If you need to work with files outside this structure, list the required files and ask the user for permission first."""
            
            output_path = root_dir / agent_name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(agent_content)
            print(f"Created {output_path}")
            
            created_files.add(agent_name)
            
        except Exception as e:
            print(f"Error processing directory '{dir_path}': {str(e)}")
            print("Please ensure your config.yaml uses one of these formats:")
            print("  - Simple directory: 'api'")
            print("  - Nested directory: 'api/tests'")
            print("  - Special directory: '__tests__'")
            continue

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--recurring', action='store_true', help='Run the script every minute')
        args = parser.parse_args()
        
        while True:  # Add while loop for recurring execution
            # Get the .agentic-cursorrules directory path
            agentic_dir = Path(__file__).parent
            
            # Create default config.yaml in the .agentic-cursorrules directory
            config_path = agentic_dir / 'config.yaml'
            if not config_path.exists():
                default_config = {
                    'tree_focus': ['api', 'app']
                }
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f)
            
            # Load config with error handling
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if not isinstance(config, dict) or 'tree_focus' not in config:
                        raise ValueError("Invalid config format: 'tree_focus' list is required")
                    focus_dirs = config.get('tree_focus', [])
                    if not isinstance(focus_dirs, list):
                        raise ValueError("'tree_focus' must be a list of directories")
            except Exception as e:
                print(f"Error loading config.yaml: {str(e)}")
                print("Using default configuration...")
                focus_dirs = ['api', 'app']
            
            # Use parent directory of .agentic-cursorrules as the project root
            project_root = agentic_dir.parent
            generator = ProjectTreeGenerator(project_root)
            
            # Generate tree for each focus directory
            found_dirs = generator.find_focus_dirs(project_root, focus_dirs)
            
            # Keep track of processed directories
            processed_dirs = set()
            
            # Create a set of all configured paths for exclusion checking
            config_paths = {str(Path(fd)) for fd in focus_dirs}
            
            for focus_dir in found_dirs:
                # Calculate relative path from project root
                rel_path = focus_dir.relative_to(project_root)
                
                # Skip if this directory is already included in a parent tree
                if any(str(rel_path).startswith(str(pd)) for pd in processed_dirs 
                       if not any(part.startswith('__') for part in rel_path.parts)):
                    continue
                
                print(f"\nTree for {focus_dir.name}:")
                print("=" * (len(focus_dir.name) + 9))
                
                # Generate skip_dirs for subdirectories that will be processed separately
                skip_dirs = {str(d.relative_to(project_root)) for d in found_dirs 
                            if str(d.relative_to(project_root)).startswith(str(rel_path)) 
                            and d != focus_dir 
                            and any(part.startswith('__') for part in d.relative_to(project_root).parts)}
                
                # Pass the config_paths to generate_tree
                tree_content = generator.generate_tree(
                    focus_dir, 
                    skip_dirs=skip_dirs,
                    config_paths=config_paths
                )
                print('\n'.join(tree_content))
                
                # Save tree files in .agentic-cursorrules directory
                with open(agentic_dir / f'tree_{focus_dir.name}.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(tree_content))
                
                processed_dirs.add(rel_path)
            
            # Generate agent files in .agentic-cursorrules directory
            generate_agent_files([str(d.relative_to(project_root)) for d in found_dirs], agentic_dir)

            if not args.recurring:
                break
                
            time.sleep(60)  # Wait for 1 minute before next iteration
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
