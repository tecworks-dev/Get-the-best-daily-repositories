/*
 * Copyright 2023-2024 Juan Miguel Giraldo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;
use crate::alloc::string::ToString;
use lazy_static::lazy_static;
use crate::time::Timestamp;
use crate::Hash;
use crate::OperationProof;
use crate::verification::FSProof;
use crate::verification::ProofData;
use crate::vkfs::DirEntry;
use crate::serial_println;
use crate::verification::FSOpType;
use crate::verification::VERIFICATION_REGISTRY;
use crate::Verifiable;
use crate::verification::Operation;
use crate::VerificationError;
use crate::hash;
use crate::vkfs::Superblock;
use alloc::format;
use crate::tsc;
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use core::sync::atomic::Ordering;
use crate::VirtAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilePermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

enum ProcessResult {
    Regular(String),
    Symlink(String, String),
}

#[derive(Debug, Clone)]
pub enum FSOperation {
    Write {
        path: String,
        data: Vec<u8>,
    },
    Create {
        path: String,
    },
    Delete {
        path: String,
    },
}

#[derive(Debug)]
pub enum FsError {
    NotFound,
    AlreadyExists,
    InvalidName,
    PermissionDenied,
    NotADirectory,
    NotAFile,
    IsDirectory,
    IoError,
    InvalidPath,
    SymlinkLoop,
    DirectoryNotEmpty,
    ProcessError,
    FileSystemError,
    InvalidState,
}

impl From<FsError> for VerificationError {
    fn from(error: FsError) -> Self {
        match error {
            FsError::NotFound => VerificationError::InvalidState,
            FsError::AlreadyExists => VerificationError::InvalidState,
            FsError::InvalidPath => VerificationError::InvalidOperation,
            FsError::PermissionDenied => VerificationError::InvalidOperation,
            FsError::NotADirectory => VerificationError::InvalidState,
            FsError::NotAFile => VerificationError::InvalidState,
            FsError::IsDirectory => VerificationError::InvalidState,
            FsError::IoError => VerificationError::OperationFailed,
            FsError::SymlinkLoop => VerificationError::InvalidOperation,
            FsError::DirectoryNotEmpty => VerificationError::InvalidOperation,
            FsError::ProcessError => VerificationError::OperationFailed,
            FsError::InvalidName => VerificationError::InvalidOperation,
            FsError::FileSystemError => VerificationError::InvalidState,
            FsError::InvalidState => VerificationError::InvalidState,
        }
    }
}
#[derive(Debug, Clone)]
pub struct PathInfo {
    pub absolute: String,
    pub canonical: String,
    pub is_symlink: bool,
}

impl PathInfo {
    pub fn new(absolute: String, canonical: String, is_symlink: bool) -> Self {
        Self {
            absolute,
            canonical,
            is_symlink,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileTime(pub Timestamp);

impl FileTime {
    pub fn now() -> Self {
        serial_println!("DEBUG: Inside FileTime::now()");
        let timestamp = Timestamp::now();
        serial_println!("DEBUG: Timestamp created: secs={}", timestamp.secs);
        FileTime(timestamp)
    }

    pub fn as_timestamp(&self) -> Timestamp {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct FileStats {
    pub size: usize,
    pub permissions: FilePermissions,
    pub created: FileTime,
    pub modified: FileTime,
    pub is_directory: bool,
}

pub struct VKFileSystem {
    superblock: Mutex<Superblock>,
    
}

impl Default for FileStats {
    fn default() -> Self {
        Self {
            size: 0,
            permissions: FilePermissions {
                read: false,
                write: false,
                execute: false,
            },
            created: FileTime(Timestamp::now()),
            modified: FileTime(Timestamp::now()),
            is_directory: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathComponents {
    components: Vec<String>,
    is_absolute: bool,
}

impl PathComponents {
    pub fn new(path: &str) -> Self {
        let is_absolute = path.starts_with('/');
        let components: Vec<String> = path
            .split('/')
            .filter(|s| !s.is_empty() && *s != ".")
            .map(String::from)
            .collect();
        
        Self {
            components,
            is_absolute,
        }
    }

    pub fn resolve(&self, current_path: &str) -> Result<String, FsError> {
        let mut result = if self.is_absolute {
            Vec::new()
        } else {
            current_path
                .split('/')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect::<Vec<_>>()
        };

        for component in &self.components {
            match component.as_str() {
                ".." => {
                    if result.is_empty() && self.is_absolute {
                        return Err(FsError::InvalidPath);
                    }
                    result.pop();
                }
                _ => result.push(component.clone()),
            }
        }

        if result.is_empty() {
            Ok(String::from("/"))
        } else {
            Ok(format!("/{}", result.join("/")))
        }
    }
}

pub fn normalize_path(path: &str) -> String {
    let mut components = Vec::new();
    let is_absolute = path.starts_with('/');
    
    for component in path.split('/') {
        match component {
            "" | "." => continue,
            ".." => {
                if !components.is_empty() && components.last() != Some(&"..") {
                    components.pop();
                } else if !is_absolute {
                    components.push("..");
                }
            },
            name => components.push(name),
        }
    }
    
    let mut result = if is_absolute { "/".to_string() } else { String::new() };
    result.push_str(&components.join("/"));
    
    if result.is_empty() {
        "/".to_string()
    } else {
        result
    }
}

pub fn join_paths(base: &str, path: &str) -> String {
    
    if path.starts_with('/') {
        return normalize_path(path);
    }
    
    
    if path.is_empty() {
        return normalize_path(base);
    }
    
    
    let mut result = if base.is_empty() || base == "." {
        String::from("/")
    } else if base == "/" {
        String::from("/")
    } else {
        base.trim_end_matches('/').to_string()
    };
    
    
    if !result.ends_with('/') && !path.starts_with('/') {
        result.push('/');
    }
    
    
    result.push_str(path.trim_start_matches('/'));
    
    
    let normalized = normalize_path(&result);
    
    
    if !normalized.starts_with('/') {
        format!("/{}", normalized)
    } else {
        normalized
    }
}

pub fn validate_path(fs: &mut InMemoryFs, path: &str) -> Result<FileStats, FsError> {
    serial_println!("Validating path: {}", path);
    
    let normalized = normalize_path(path);
    serial_println!("Normalized path: {}", normalized);
    
    if normalized == "/" {
        return Ok(FileStats {
            size: 0,
            permissions: FilePermissions {
                read: true,
                write: true,
                execute: true,
            },
            created: FileTime::now(),
            modified: FileTime::now(),
            is_directory: true,
        });
    }

    let path_to_check = if normalized.starts_with('/') {
        &normalized[1..]
    } else {
        &normalized
    };
    
    let stats = fs.stat(path_to_check)?;
    
    if !stats.is_directory {
        serial_println!("validate_path: {} is not a directory", normalized);
        return Err(FsError::NotADirectory);
    }
    
    serial_println!("validate_path: {} is valid directory", normalized);
    Ok(stats)
}

pub trait FileSystem {
    fn create_file(&mut self, path: &str, permissions: FilePermissions) -> Result<(), FsError>;
    fn create_directory(&mut self, path: &str, permissions: FilePermissions) -> Result<(), FsError>;
    fn read_file(&mut self, path: &str) -> Result<Vec<u8>, FsError>;
    fn write_file(&mut self, path: &str, contents: &[u8]) -> Result<(), FsError>;
    fn remove_file(&mut self, path: &str) -> Result<(), FsError>;
    fn remove_directory(&mut self, path: &str) -> Result<(), FsError>;
    fn stat(&mut self, path: &str) -> Result<FileStats, FsError>;
    fn list_directory(&mut self, path: &str) -> Result<Vec<String>, FsError>;
}

pub trait VerifiedFileSystem: FileSystem {
    fn verify_operation(&self, operation: FSOperation) -> Result<OperationProof, FsError>;
    fn validate_state(&self) -> Result<Hash, FsError>;
    fn get_merkle_root(&self) -> Result<[u8; 32], FsError>;
    fn verify_path(&mut self, path: &str) -> Result<bool, FsError>;
    fn get_verification_status(&self) -> Result<VerificationStatus, FsError>;
}

impl VerifiedFileSystem for InMemoryFs {
    fn verify_operation(&self, operation: FSOperation) -> Result<OperationProof, FsError> {
        let proof = self.verify_operation(&operation)
            .map_err(|_| FsError::FileSystemError)?;
        
        VERIFICATION_REGISTRY.lock()
            .register_proof(proof.clone());
            
        Ok(proof)
    }

    fn validate_state(&self) -> Result<Hash, FsError> {
        Ok(Hash(self.fs_hash.load(Ordering::SeqCst)))
    }

    fn get_merkle_root(&self) -> Result<[u8; 32], FsError> {
        Ok(self.superblock.root_merkle_hash)
    }

    fn verify_path(&mut self, path: &str) -> Result<bool, FsError> {
        let components = PathComponents::new(path);
        let resolved = components.resolve("/")?;
    
        if resolved == "/" {
            return Ok(true);
        }
    
        
        match self.stat(&resolved) {
            Ok(stats) => Ok(stats.is_directory),
            Err(FsError::NotFound) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn get_verification_status(&self) -> Result<VerificationStatus, FsError> {
        Ok(VerificationStatus {
            last_verified: self.superblock.last_verification,
            total_proofs: VERIFICATION_REGISTRY.lock().get_proofs().len(),
            merkle_root: self.superblock.root_merkle_hash,
            state_hash: self.state_hash(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct VerificationStatus {
    pub last_verified: u64,
    pub total_proofs: usize,
    pub merkle_root: [u8; 32],
    pub state_hash: Hash,
}


#[derive(Debug)]
struct InodeData {
    data: Vec<u8>,
    stats: FileStats,
    symlink_target: Option<String>,
}

impl InodeData {
    fn validate_operation(&self, op: &FSOperation) -> Result<(), FsError> {
        match op {
            FSOperation::Write { path: _, data } => {
                if self.stats.is_directory {
                    return Err(FsError::IsDirectory);
                }
                if !self.stats.permissions.write {
                    return Err(FsError::PermissionDenied);
                }
                if data.len() > u64::MAX as usize {
                    return Err(FsError::IoError);
                }
                Ok(())
            },
            FSOperation::Create { path } => {
                if path.contains('\0') || path.contains('/') {
                    return Err(FsError::InvalidName);
                }
                if !self.stats.permissions.write {
                    return Err(FsError::PermissionDenied);
                }
                Ok(())
            },
            FSOperation::Delete { path: _ } => {
                if !self.stats.permissions.write {
                    return Err(FsError::PermissionDenied);
                }
                Ok(())
            }
        }
    }
}

impl Clone for InodeData {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            stats: self.stats.clone(),
            symlink_target: self.symlink_target.clone(),
        }
    }
}

#[derive(Debug)]
struct Inode {
    name: String,
    data: InodeData,
    children: Option<Vec<Inode>>, 
}

impl Clone for Inode {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.clone(),
            children: self.children.clone(),
        }
    }
}

pub struct InMemoryFs {
    root: Inode,
    fs_hash: AtomicU64,
    pub superblock: Superblock,
}

impl InMemoryFs {
    pub fn new() -> Self {
        serial_println!("InMemoryFs: Starting initialization");

        serial_println!("DEBUG: Starting root stats creation");
        let root_stats = {
            serial_println!("DEBUG: Creating FileTime::now()");
            let now = match FileTime::now() {
                time => {
                    serial_println!("DEBUG: FileTime created successfully");
                    time
                }
            };
    
            serial_println!("DEBUG: Creating permissions structure");
            let permissions = FilePermissions {
                read: true,
                write: true,
                execute: true,
            };
            serial_println!("DEBUG: Permissions created successfully");
    
            serial_println!("DEBUG: Creating complete FileStats structure");
            let stats = FileStats {
                size: 0,
                permissions,
                created: now,
                modified: now,
                is_directory: true,
            };
            serial_println!("DEBUG: Root stats structure created successfully");
            stats
        };
    
        serial_println!("DEBUG: Starting root inode data creation");
        let root_data = {
            let data = InodeData {
                data: Vec::new(),
                stats: root_stats.clone(),
                symlink_target: None,
            };
            serial_println!("DEBUG: Root inode data created successfully");
            data
        };
    
        serial_println!("DEBUG: Creating root inode structure");
        let root = {
            let inode = Inode {
                name: String::from("/"),
                data: root_data,
                children: Some(Vec::with_capacity(10)),
            };
            serial_println!("DEBUG: Root inode structure created successfully");
            inode
        };
    
        serial_println!("DEBUG: Creating filesystem structure");
        let fs = Self { 
            root,
            fs_hash: AtomicU64::new(0),
            superblock: Superblock::new(1024 * 1024, 1024),
        };
        
        if fs.root.children.is_none() {
            serial_println!("CRITICAL ERROR: Root children vector is None!");
        }
    
        serial_println!("DEBUG: Filesystem structure created");
        fs
    }

    pub fn init_directory_structure(&mut self) -> Result<(), FsError> {
        serial_println!("Initializing directory structure");
        
        let dir_permissions = FilePermissions {
            read: true,
            write: true,
            execute: true,
        };

        let base_dirs = [
            "/bin",
            "/home",
            "/tmp",
            "/usr",
            "/dev",
            "/etc",
        ];

        for dir in &base_dirs {
            match self.create_directory(dir, dir_permissions) {
                Ok(_) => serial_println!("Created directory {}", dir),
                Err(e) => serial_println!("Failed to create {}: {:?}", dir, e),
            }
        }

        let sub_dirs = [
            "/usr/bin",
            "/usr/lib",
        ];

        for dir in &sub_dirs {
            match self.create_directory(dir, dir_permissions) {
                Ok(_) => serial_println!("Created directory {}", dir),
                Err(e) => serial_println!("Failed to create {}: {:?}", dir, e),
            }
        }

        Ok(())
    }

    pub fn create_directory(&mut self, path: &str, permissions: FilePermissions) -> Result<(), FsError> {
        serial_println!("Creating directory: {}", path);

        if path == "/" {
            return Ok(());
        }

        let (parent_path, dir_name) = match path.rfind('/') {
            Some(pos) => {
                let parent = if pos == 0 {
                    "/"
                } else {
                    &path[..pos]
                };
                let name = &path[pos + 1..];
                (parent, name)
            },
            None => ("/", path)
        };
        
        serial_println!("Creating {} in parent {}", dir_name, parent_path);

        let parent_inode = self.find_inode(parent_path)?;
        
        if !parent_inode.data.stats.is_directory {
            serial_println!("Parent {} is not a directory", parent_path);
            return Err(FsError::NotADirectory);
        }
    
        let children = parent_inode.children.as_mut()
            .ok_or(FsError::NotADirectory)?;

        if children.iter().any(|node| node.name == dir_name) {
            serial_println!("Directory {} already exists in {}", dir_name, parent_path);
            return Err(FsError::AlreadyExists);
        }

        let new_inode = Inode {
            name: String::from(dir_name),
            data: InodeData {
                data: Vec::new(),
                stats: FileStats {
                    size: 0,
                    permissions,
                    created: FileTime::now(),
                    modified: FileTime::now(),
                    is_directory: true,
                },
                symlink_target: None,
            },
            children: Some(Vec::new()),
        };
    
        serial_println!("Created directory node: {}", dir_name);

        children.push(new_inode);
        serial_println!("Added {} to {}", dir_name, parent_path);
    
        Ok(())
    }
    
    pub fn remove_directory(&mut self, path: &str) -> Result<(), FsError> {
        let mut fs = FILESYSTEM.lock();
        
        
        let parent_path = path.rfind('/')
            .map(|pos| &path[..pos])
            .unwrap_or("");
            
        let dir_name = path.rfind('/')
            .map(|pos| &path[pos + 1..])
            .unwrap_or(path);
    
        
        let mut stored_dir = None;
        let mut stored_pos = None;
    
        if let Ok(parent_inode) = self.find_inode(parent_path) {
            if let Some(children) = &parent_inode.children {
                if let Some(pos) = children.iter().position(|node| node.name == dir_name) {
                    stored_dir = Some(children[pos].clone());
                    stored_pos = Some(pos);
                }
            }
        }
    
        
        if let Err(e) = fs.superblock.verify_tree_consistency() {
            
            if let (Some(dir), Some(pos)) = (stored_dir, stored_pos) {
                if let Ok(parent_inode) = self.find_inode(parent_path) {
                    if let Some(children) = &mut parent_inode.children {
                        children.insert(pos, dir);
                    }
                }
            }
            return Err(FsError::FileSystemError);
        }
        
        let inode = self.find_inode(path)?;
        
        if !inode.data.stats.is_directory {
            return Err(FsError::NotADirectory);
        }
    
        if let Some(children) = &inode.children {
            if !children.is_empty() {
                return Err(FsError::DirectoryNotEmpty);
            }
        }
    
        let parent_inode = self.find_inode(parent_path)?;
        
        if let Some(children) = &mut parent_inode.children {
            if let Some(pos) = children.iter().position(|node| node.name == dir_name) {
                children.remove(pos);
                return Ok(());
            }
        }
    
        Err(FsError::NotFound)
    }

    pub fn rename_directory(&mut self, old_path: &str, new_path: &str) -> Result<(), FsError> {
        let mut fs = FILESYSTEM.lock();
        let old_parent = old_path.rfind('/').map(|pos| &old_path[..pos]).unwrap_or("");
        
        if let Err(e) = fs.superblock.verify_tree_consistency() {
            if let Ok(src_parent) = self.find_inode(old_parent) {
                if let Some(ref mut src_children) = src_parent.children {
                    if !src_children.is_empty() {
                        
                        let removed_dir = src_children.last().unwrap().clone();
                        src_children.push(removed_dir.clone());
                    }
                }
            }
            return Err(FsError::FileSystemError);
        }

        let old_parent = old_path.rfind('/').map(|pos| &old_path[..pos]).unwrap_or("");
        let new_parent = new_path.rfind('/').map(|pos| &new_path[..pos]).unwrap_or("");
        let new_name = new_path.rfind('/').map(|pos| &new_path[pos + 1..]).unwrap_or(new_path);
        let src_name = old_path.rfind('/').map(|pos| &old_path[pos + 1..]).unwrap_or(old_path);
    
        
        if old_parent == new_parent {
            let parent_inode = self.find_inode(old_parent)?;
            if !parent_inode.data.stats.is_directory {
                return Err(FsError::NotADirectory);
            }
    
            let children = parent_inode.children.as_mut()
                .ok_or(FsError::NotADirectory)?;
            
            if let Some(dir_inode) = children.iter_mut()
                .find(|node| node.name == src_name) {
                dir_inode.name = String::from(new_name);
                return Ok(());
            }
            return Err(FsError::NotFound);
        }
    
        
        
        let (src_exists, dst_exists) = {
            let src_node = self.find_inode(old_path)?;
            if !src_node.data.stats.is_directory {
                return Err(FsError::NotADirectory);
            }
            
            
            let src_data = src_node.data.clone();
            let src_children = src_node.children.clone();
            
            let dst_exists = self.find_inode(new_path).is_ok();
            
            (Some((src_data, src_children)), dst_exists)
        };
    
        
        let dst_parent = self.find_inode(new_parent)?;
        if !dst_parent.data.stats.is_directory {
            return Err(FsError::NotADirectory);
        }
    
        if dst_exists {
            return Err(FsError::AlreadyExists);
        }
    
        let (src_data, src_children) = src_exists.ok_or(FsError::NotFound)?;
    
        
        let dst_children = dst_parent.children.as_mut()
            .ok_or(FsError::NotADirectory)?;
    
        let new_inode = Inode {
            name: String::from(new_name),
            data: src_data,
            children: src_children,
        };
        
        dst_children.push(new_inode);
    
        
        let src_parent = self.find_inode(old_parent)?;
        if let Some(src_children) = src_parent.children.as_mut() {
            src_children.retain(|node| node.name != src_name);
        }
    
        Ok(())
    }
    
    fn get_absolute_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{}", path)
        }
    }

    fn process_path_component(&mut self, path: &str, visited: &mut Vec<String>) -> Result<ProcessResult, FsError> {
        let inode = self.find_inode(path)?;
        
        if let Some(target) = &inode.data.symlink_target {
            visited.push(path.to_string());
            let canonical = if target.starts_with('/') {
                target.clone()
            } else {
                let parent = path.rsplitn(2, '/').nth(1).unwrap_or("");
                normalize_path(&format!("{}/{}", parent, target))
            };
            Ok(ProcessResult::Symlink(target.clone(), canonical))
        } else {
            Ok(ProcessResult::Regular(path.to_string()))
        }
    }

    pub fn resolve_path(&mut self, path: &str) -> Result<PathInfo, FsError> {
        let normalized = normalize_path(path);
        let mut components: Vec<String> = normalized.split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        
        let mut current_path = String::from("/");
        let mut canonical_path = String::from("/");
        let mut visited = Vec::new();
        let mut pending_components: Vec<String> = Vec::new();
        
        while let Some(component) = components.pop() {
            match component.as_str() {
                "." => continue,
                ".." => {
                    if current_path != "/" {
                        current_path = current_path.rsplitn(3, '/').nth(2)
                            .unwrap_or("/").to_string();
                        if !current_path.starts_with('/') {
                            current_path = format!("/{}", current_path);
                        }
                    }
                    if canonical_path != "/" {
                        canonical_path = canonical_path.rsplitn(3, '/').nth(2)
                            .unwrap_or("/").to_string();
                        if !canonical_path.starts_with('/') {
                            canonical_path = format!("/{}", canonical_path);
                        }
                    }
                    continue;
                },
                comp => {
                    
                    current_path = if current_path == "/" {
                        format!("/{}", comp)
                    } else {
                        format!("{}/{}", current_path, comp)
                    };
    
                    
                    if visited.contains(&current_path) {
                        return Err(FsError::SymlinkLoop);
                    }
    
                    
                    match self.process_path_component(&current_path, &mut visited)? {
                        ProcessResult::Regular(can_path) => {
                            canonical_path = can_path;
                        },
                        ProcessResult::Symlink(target, can_path) => {
                            canonical_path = can_path;
                            
                            
                            components.extend(pending_components.drain(..));
                            
                            
                            let mut target_components: Vec<String> = target.split('/')
                                .filter(|s| !s.is_empty())
                                .map(String::from)
                                .collect();
                            
                            if target.starts_with('/') {
                                
                                components.clear();
                            }
                            
                            
                            components.extend(target_components.into_iter().rev());
                        }
                    }
                }
            }
        }
    
        Ok(PathInfo {
            absolute: current_path.clone(),
            canonical: canonical_path.clone(),
            is_symlink: current_path != canonical_path,
        })
    }

    pub fn pwd(&mut self) -> Result<PathInfo, FsError> {
        serial_println!("PWD: Starting pwd() execution");
    
        
        let root = &self.root;
        serial_println!("PWD: Root inode name: {}", root.name);
        
        
        if !root.data.stats.is_directory {
            serial_println!("PWD: Root is not a directory!");
            return Err(FsError::NotADirectory);
        }
    
        
        let current_path = String::from("/");
        serial_println!("PWD: Current path determined: {}", current_path);
        
        
        let path_info = PathInfo {
            absolute: current_path.clone(),
            canonical: current_path.clone(),
            is_symlink: false,
        };
        
        serial_println!("PWD: Created PathInfo: {:?}", path_info);
        
        
        match self.stat(&path_info.absolute) {
            Ok(stats) => {
                serial_println!("PWD: Path stats verified: is_directory={}", stats.is_directory);
                if !stats.is_directory {
                    serial_println!("PWD: Path is not a directory");
                    return Err(FsError::NotADirectory);
                }
            },
            Err(e) => {
                serial_println!("PWD: Failed to stat path: {:?}", e);
                return Err(e);
            }
        }
        
        serial_println!("PWD: Successfully returning path_info");
        Ok(path_info)
    }


    pub fn compute_new_state(&self, op: &FSOperation) -> Hash {
        let current = Hash(self.fs_hash.load(AtomicOrdering::SeqCst));
        let op_hash = match op {
            FSOperation::Write { path, data } => {
                let mut hasher = hash::hash_memory(
                    VirtAddr::new(data.as_ptr() as u64),
                    data.len()
                );
                hasher.0 ^= hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                ).0;
                hasher
            },
            FSOperation::Create { path } => {
                hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                )
            },
            FSOperation::Delete { path } => {
                let mut hash = hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                );
                hash.0 = !hash.0;
                hash
            },
        };
        Hash(current.0 ^ op_hash.0)
    }

    pub fn generate_operation_proof(&self, op: &FSOperation) -> Result<OperationProof, VerificationError> {
        let prev_state = self.state_hash();
        let new_state = self.compute_new_state(op);
        
        
        let op_type = match op {
            FSOperation::Write { .. } => FSOpType::Modify,
            FSOperation::Create { .. } => FSOpType::Create,
            FSOperation::Delete { .. } => FSOpType::Delete,
        };

        
        let fs_proof = FSProof {
            operation: op_type,
            path: match op {
                FSOperation::Write { path, .. } |
                FSOperation::Create { path } |
                FSOperation::Delete { path } => path.clone(),
            },
            content_hash: new_state,
            prev_state,
            new_state,
            op: op.clone(),
        };

        
        Ok(OperationProof {
            op_id: tsc::read_tsc(),
            prev_state,
            new_state,
            data: ProofData::Filesystem(fs_proof),
            signature: [0u8; 64], 
        })
    }

    fn verify_operation_integrity(&mut self, op: &FSOperation) -> Result<bool, VerificationError> {        match op {
            FSOperation::Write { path, data } => {
                
                let data_hash = hash::hash_memory(
                    VirtAddr::new(data.as_ptr() as u64),
                    data.len()
                );
                
                
                if self.find_inode(path).is_err() {
                    return Ok(false);
                }
                
                Ok(true)
            },
            FSOperation::Create { path } => {
                
                match self.find_inode(path) {
                    Ok(_) => Ok(false),
                    Err(FsError::NotFound) => Ok(true),
                    Err(_) => Ok(false),
                }
            },
            FSOperation::Delete { path } => {
                
                match self.find_inode(path) {
                    Ok(inode) => Ok(!inode.data.stats.is_directory || 
                                  inode.children.as_ref().map_or(true, |c| c.is_empty())),
                    Err(_) => Ok(false),
                }
            },
        }
    }

    pub fn execute_verified_operation(&mut self, op: &FSOperation) -> Result<OperationProof, VerificationError> {
        
        let prev_state = self.state_hash();
        let op_proof = match op {
            FSOperation::Write { path, data } => {
                let data_hash = hash::hash_memory(
                    VirtAddr::new(data.as_ptr() as u64),
                    data.len()
                );
                FSProof {
                    operation: FSOpType::Modify,
                    path: path.clone(),
                    content_hash: data_hash,
                    prev_state,
                    new_state: Hash(prev_state.0 ^ data_hash.0),
                    op: op.clone(),
                }
            },
            FSOperation::Create { path } => {
                let path_hash = hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                );
                FSProof {
                    operation: FSOpType::Create,
                    path: path.clone(),
                    content_hash: path_hash,
                    prev_state,
                    new_state: Hash(prev_state.0 ^ path_hash.0),
                    op: op.clone(),
                }
            },
            FSOperation::Delete { path } => {
                let path_hash = hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                );
                FSProof {
                    operation: FSOpType::Delete,
                    path: path.clone(),
                    content_hash: path_hash,
                    prev_state,
                    new_state: Hash(!prev_state.0),
                    op: op.clone(),
                }
            },
        };
    
        let proof = OperationProof {
            op_id: tsc::read_tsc(),
            prev_state,
            new_state: op_proof.new_state,
            data: ProofData::Filesystem(op_proof.clone()),
            signature: [0; 64],
        };
    
        
        match op {
            FSOperation::Write { path, data } => {
                if let Err(e) = self.write_file(path, data) {
                    return Err(VerificationError::OperationFailed);
                }
            },
            FSOperation::Create { path } => {
                if let Err(e) = self.create_file(path, FilePermissions {
                    read: true,
                    write: true,
                    execute: false,
                }) {
                    return Err(VerificationError::OperationFailed);
                }
            },
            FSOperation::Delete { path } => {
                if let Err(e) = self.remove_file(path) {
                    return Err(VerificationError::OperationFailed);
                }
            },
        }
    
        
        let verification_hash = self.compute_new_state(op);
        if verification_hash != proof.new_state {
            return Err(VerificationError::OperationFailed);
        }
    
        
        self.fs_hash.store(proof.new_state.0, Ordering::SeqCst);
    
        let superblock = &mut FILESYSTEM.lock().superblock;
        superblock.track_state_transition(op.get_type())?;
        
        VERIFICATION_REGISTRY.lock().register_proof(proof.clone());
    
        Ok(proof)
    }

    pub fn verify_operation(&self, op: &FSOperation) -> Result<OperationProof, FsError> {
        let prev_hash = self.fs_hash.load(AtomicOrdering::SeqCst);
        let new_state = self.compute_new_state(op);
        
        let proof = FSProof {
            prev_state: Hash(prev_hash),
            new_state,
            op: op.clone(),
            content_hash: match op {
                FSOperation::Write { path, .. } |
                FSOperation::Create { path } |
                FSOperation::Delete { path } => hash::hash_memory(
                    VirtAddr::new(path.as_ptr() as u64),
                    path.len()
                ),
            },
            operation: match op {
                FSOperation::Write { .. } => FSOpType::Modify,
                FSOperation::Create { .. } => FSOpType::Create,
                FSOperation::Delete { .. } => FSOpType::Delete,
            },
            path: match op {
                FSOperation::Write { path, .. } |
                FSOperation::Create { path } |
                FSOperation::Delete { path } => path.clone(),
            },
        };
        
        Ok(OperationProof {
            op_id: tsc::read_tsc(),
            prev_state: Hash(prev_hash),
            new_state,
            data: ProofData::Filesystem(proof),
            signature: [0u8; 64],
        })
    }

    fn find_inode<'a>(&'a mut self, path: &str) -> Result<&'a mut Inode, FsError> {
        serial_println!("Finding inode for path: {}", path);
        if path == "/" {
            serial_println!("Returning root inode");
            return Ok(&mut self.root);
        }
    
        let mut current = &mut self.root;
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        
        serial_println!("Path components: {:?}", parts);
    
        for (i, part) in parts.iter().enumerate() {
            let children = current.children.as_mut()
                .ok_or_else(|| {
                    serial_println!("Not a directory at component {}: {}", i, current.name);
                    FsError::NotADirectory
                })?;
    
            current = children.iter_mut()
                .find(|node| node.name == *part)
                .ok_or_else(|| {
                    serial_println!("Component not found: {} (at position {})", part, i);
                    FsError::NotFound
                })?;
            
            serial_println!("Found component: {} is_directory={}", 
                part,
                current.children.is_some()
            );
        }
    
        Ok(current)
    }
}

impl FileSystem for InMemoryFs {
    fn create_file(&mut self, path: &str, permissions: FilePermissions) -> Result<(), FsError> {
        let (dir_path, file_name) = match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => return Err(FsError::InvalidName),
        };

        let parent = self.find_inode(if dir_path.is_empty() { "/" } else { dir_path })?;
        
        let children = parent.children.as_mut()
            .ok_or(FsError::NotADirectory)?;

        if children.iter().any(|node| node.name == file_name) {
            return Err(FsError::AlreadyExists);
        }

        let now = FileTime::now();
        let stats = FileStats {
            size: 0,
            permissions,
            created: now,
            modified: now,
            is_directory: false,
        };

        children.push(Inode {
            name: String::from(file_name),
            data: InodeData {
                data: Vec::new(),
                stats,
                symlink_target: None,
            },
            children: None,
        });

        Ok(())
    }

    fn create_directory(&mut self, path: &str, permissions: FilePermissions) -> Result<(), FsError> {
        let (dir_path, dir_name) = match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => return Err(FsError::InvalidName),
        };

        let parent = self.find_inode(if dir_path.is_empty() { "/" } else { dir_path })?;
        
        let children = parent.children.as_mut()
            .ok_or(FsError::NotADirectory)?;

        if children.iter().any(|node| node.name == dir_name) {
            return Err(FsError::AlreadyExists);
        }

        let now = FileTime::now();
        let stats = FileStats {
            size: 0,
            permissions,
            created: now,
            modified: now,
            is_directory: true,
        };

        children.push(Inode {
            name: String::from(dir_name),
            data: InodeData {
                data: Vec::new(),
                stats,
                symlink_target: None,
            },
            children: Some(Vec::new()),
        });

        Ok(())
    }

    fn read_file(&mut self, path: &str) -> Result<Vec<u8>, FsError> {
        let inode = self.find_inode(path)?;
        if inode.children.is_some() {
            return Err(FsError::IsDirectory);
        }
        
        
        let current_root = hash::hash_memory(
            VirtAddr::new(inode.data.data.as_ptr() as u64),
            inode.data.data.len()
        );
        
        if inode.data.stats.size > inode.data.data.len() {
            return Err(FsError::IoError);
        }
        
        Ok(inode.data.data.clone())
    }

    fn write_file(&mut self, path: &str, contents: &[u8]) -> Result<(), FsError> {
        let mut fs = FILESYSTEM.lock();
        if let Err(e) = fs.superblock.verify_tree_consistency() {
            return Err(FsError::FileSystemError);
        }

        
        if path.is_empty() || path.contains('\0') {
            return Err(FsError::InvalidName);
        }
    
        let op = FSOperation::Write {
            path: path.to_string(),
            data: contents.to_vec(),
        };
        
        
        let proof = self.verify_operation(&op)?;
        
        
        let (dir_path, file_name) = match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => return Err(FsError::InvalidName),
        };
    
        let parent = self.find_inode(if dir_path.is_empty() { "/" } else { dir_path })?;
        
        if !parent.data.stats.permissions.write {
            return Err(FsError::PermissionDenied);
        }
    
        
        let inode = self.find_inode(path)?;
        if let Err(e) = inode.data.validate_operation(&op) {
            return Err(e);
        }
        
        
        inode.data.data = contents.to_vec();
        inode.data.stats.size = contents.len();
        inode.data.stats.modified = FileTime::now();
        
        
        self.fs_hash.store(proof.new_state.0, Ordering::SeqCst);
        
        
        VERIFICATION_REGISTRY.lock().register_proof(proof);
        
        Ok(())
    }

    fn remove_file(&mut self, path: &str) -> Result<(), FsError> {
        let (dir_path, file_name) = match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => return Err(FsError::InvalidName),
        };

        let parent = self.find_inode(if dir_path.is_empty() { "/" } else { dir_path })?;
        
        let children = parent.children.as_mut()
            .ok_or(FsError::NotADirectory)?;

        let pos = children.iter()
            .position(|node| node.name == file_name && node.children.is_none())
            .ok_or(FsError::NotFound)?;

        children.remove(pos);
        Ok(())
    }

    fn remove_directory(&mut self, path: &str) -> Result<(), FsError> {
        let (dir_path, dir_name) = match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => return Err(FsError::InvalidName),
        };

        let parent = self.find_inode(if dir_path.is_empty() { "/" } else { dir_path })?;
        
        let children = parent.children.as_mut()
            .ok_or(FsError::NotADirectory)?;

        let pos = children.iter()
            .position(|node| node.name == dir_name && node.children.is_some())
            .ok_or(FsError::NotFound)?;

        children.remove(pos);
        Ok(())
    }

    fn stat(&mut self, path: &str) -> Result<FileStats, FsError> {
        serial_println!("Attempting to stat path: {}", path);
        let inode = self.find_inode(path)?;
        serial_println!("Found inode for path: {} is_directory={}", 
            path, 
            inode.data.stats.is_directory
        );
        
        Ok(inode.data.stats.clone())
    }

    fn list_directory(&mut self, path: &str) -> Result<Vec<String>, FsError> {
        serial_println!("Listing directory: {}", path);

        if path == "/" {
            if let Some(children) = &self.root.children {
                return Ok(children.iter()
                    .map(|node| node.name.clone())
                    .collect());
            }
        }

        let inode = self.find_inode(path)?;
        if !inode.data.stats.is_directory {
            return Err(FsError::NotADirectory);
        }

        match &inode.children {
            Some(children) => Ok(children.iter()
                .map(|node| node.name.clone())
                .collect()),
            None => Ok(Vec::new())
        }
    }
}

impl Verifiable for InMemoryFs {
    fn generate_proof(&self, operation: Operation) -> Result<OperationProof, VerificationError> {
        match operation {
            Operation::Filesystem { path, operation_type } => {
                let op = match operation_type {
                    FSOpType::Create => FSOperation::Create { path },
                    FSOpType::Delete => FSOperation::Delete { path },
                    FSOpType::Modify => FSOperation::Write { 
                        path: path.clone(),
                        data: Vec::new() 
                    },
                };
                self.verify_operation(&op)
                    .map_err(|_| VerificationError::OperationFailed)
            },
            _ => Err(VerificationError::InvalidOperation),
        }
    }

    fn verify_proof(&self, proof: &OperationProof) -> Result<bool, VerificationError> {
        match &proof.data {
            ProofData::Filesystem(fs_proof) => {
                let current_hash = self.compute_new_state(&fs_proof.op);
                Ok(current_hash == proof.new_state)
            },
            _ => Err(VerificationError::InvalidProof),
        }
    }

    fn state_hash(&self) -> Hash {
        Hash(self.fs_hash.load(AtomicOrdering::SeqCst))
    }
}

pub fn init() {
    lazy_static! {
        pub static ref FILESYSTEM: Mutex<InMemoryFs> = {
            serial_println!("Starting filesystem initialization...");
            let mut fs = InMemoryFs::new();
            serial_println!("InMemoryFs instance created successfully");

            if let Err(e) = fs.init_directory_structure() {
                serial_println!("Failed to create directory structure: {:?}", e);
            }

            match fs.list_directory("/") {
                Ok(entries) => {
                    serial_println!("Root directory contents:");
                    for entry in entries {
                        serial_println!("  - {}", entry);
                    }
                },
                Err(e) => serial_println!("Failed to list root directory: {:?}", e),
            }
    
            Mutex::new(fs)
        };
    }
}

pub fn cleanup() {
    let mut fs = FILESYSTEM.lock();

    let sb = &mut fs.superblock;
    sb.buffer_manager.lock().flush_all();

    sb.block_cache.lock().flush();

    *fs = InMemoryFs::new();
}

lazy_static! {
    pub static ref FILESYSTEM: Mutex<InMemoryFs> = {
        serial_println!("Starting filesystem initialization...");
        let mut fs = InMemoryFs::new();
        serial_println!("InMemoryFs instance created successfully");

        if let Err(e) = fs.init_directory_structure() {
            serial_println!("Failed to create directory structure: {:?}", e);
        }

        match fs.list_directory("/") {
            Ok(entries) => {
                serial_println!("Root directory contents:");
                for entry in entries {
                    serial_println!("  - {}", entry);
                }
            },
            Err(e) => serial_println!("Failed to list root directory: {:?}", e),
        }

        Mutex::new(fs)
    };
}