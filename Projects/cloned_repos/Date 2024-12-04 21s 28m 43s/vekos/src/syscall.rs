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

use crate::{
    print, println,
    memory::{MemoryError, MemoryZoneType, UserSpaceRegion},
    process::{Process, ProcessState, PROCESS_LIST},
    fs::{FileSystem, FILESYSTEM},
    elf::ElfLoader,
    MEMORY_MANAGER,
};

use x86_64::{
    registers::model_specific::{LStar, SFMask, Efer, EferFlags},
    registers::rflags::RFlags,
    VirtAddr,
    structures::paging::{Page, Size4KiB, PageTableFlags},
};

use x86_64::instructions::interrupts;
use crate::fs::VerifiedFileSystem;
use crate::process::SessionId;
use crate::process::Session;
use crate::fs::FSOperation;
use crate::serial_println;
use crate::VERIFICATION_REGISTRY;
use crate::memory::PAGE_REF_COUNTS;
use crate::fs;
use crate::memory::PageRefCount;
use x86_64::structures::paging::FrameDeallocator;
use x86_64::structures::paging::Mapper;
use crate::MemoryManager;
use crate::fs::FsError;
use crate::process::ProcessGroupId;
use crate::process::ProcessId;
use crate::signals;
use core::arch::naked_asm;
use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;
use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;
use core::alloc::Layout;

const USER_HEAP_START: u64 = 0x4000_0000;
const USER_HEAP_SIZE: usize = 1024 * 1024 * 16;
const MAX_PROCESS_MEMORY: usize = 1024 * 1024 * 1024;
const PAGE_SIZE: usize = 4096;
const MAX_READ_SIZE: usize = 1024 * 1024;  
const MAX_WRITE_SIZE: usize = 1024 * 1024; 
const MAX_ELF_SIZE: usize = 1024 * 1024 * 8; 

#[derive(Debug, Clone)]
pub struct FileDescriptor {
    pub path: String,
    pub offset: usize,
    pub flags: u32,
}

#[derive(Clone)]
pub struct FileDescriptorTable {
    next_fd: usize,
    descriptors: BTreeMap<usize, FileDescriptor>,
}

impl FileDescriptorTable {
    pub fn new() -> Self {
        Self {
            next_fd: 1, 
            descriptors: BTreeMap::new(),
        }
    }

    pub fn allocate(&mut self, path: String, flags: u32) -> usize {
        let fd = self.next_fd;
        self.next_fd += 1;
        
        self.descriptors.insert(fd, FileDescriptor {
            path,
            offset: 0,
            flags,
        });
        
        fd
    }

    pub fn get(&self, fd: usize) -> Option<&FileDescriptor> {
        self.descriptors.get(&fd)
    }

    pub fn get_mut(&mut self, fd: usize) -> Option<&mut FileDescriptor> {
        self.descriptors.get_mut(&fd)
    }

    pub fn remove(&mut self, fd: usize) -> Option<FileDescriptor> {
        self.descriptors.remove(&fd)
    }
}

#[repr(usize)]
#[derive(Debug, Copy, Clone)]
pub enum SyscallNumber {
    Exit = 1,
    Write = 2,
    Read = 3,
    Yield = 4,
    GetPid = 5,
    Fork = 6,
    Exec = 7,
    MemAlloc = 8,
    MemDealloc = 9,
    Open = 10,
    Close = 11,
    FileRead = 12,
    FileWrite = 13,
    Stat = 14,
    Create = 15,
    Delete = 16,
    Kill = 17,
    Signal = 18,
    SigReturn = 19,
    SigProcMask = 20,
    SigAction = 21,
    GetPgrp = 22,
    SetPgrp = 23,
    Wait = 24,
    Setsid = 25,
    Getsid = 26,
    Tcsetpgrp = 27,
    Tcgetpgrp = 28,
    VkfsCreate = 29,
    VkfsDelete = 30,
    VkfsRead = 31,
    VkfsWrite = 32,
    VkfsVerify = 33,
}

impl From<SyscallError> for u64 {
    fn from(error: SyscallError) -> u64 {
        match error {
            SyscallError::InvalidSyscall => 1,
            SyscallError::InvalidArgument => 2,
            SyscallError::MemoryError(_) => 3,
            SyscallError::ProcessError => 4,
            SyscallError::IOError => 5,
            SyscallError::InvalidAddress => 6,
            SyscallError::InvalidBuffer => 7,
            SyscallError::InvalidPermissions => 8,
            SyscallError::MemoryLimitExceeded => 9,
            SyscallError::NotFound => 10,
            SyscallError::FileSystemError => 11,
            SyscallError::InvalidState => 12,
        }
    }
}

#[derive(Debug)]
pub enum SyscallError {
    InvalidSyscall,
    InvalidArgument,
    MemoryError(MemoryError),
    ProcessError,
    IOError,
    InvalidAddress,
    InvalidBuffer,
    InvalidPermissions,
    MemoryLimitExceeded,
    NotFound,
    FileSystemError,
    InvalidState,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub size: usize,
    pub address: VirtAddr,
    pub layout: Layout,
}

#[derive(Clone)]
pub struct ProcessMemory {
    pub heap_start: VirtAddr,
    pub heap_size: usize,
    pub allocations: Vec<MemoryAllocation>,
    pub total_allocated: usize,
}

impl ProcessMemory {
    pub fn new() -> Self {
        Self {
            heap_start: VirtAddr::new(USER_HEAP_START),
            heap_size: 0,
            allocations: Vec::new(),
            total_allocated: 0,
        }
    }
}

static SYSCALL_COUNT: AtomicUsize = AtomicUsize::new(0);

lazy_static! {
    pub static ref KEYBOARD_BUFFER: Mutex<Vec<u8>> = Mutex::new(Vec::with_capacity(1024));
}

pub fn init() {
    unsafe {
        
        Efer::update(|efer| {
            efer.insert(EferFlags::SYSTEM_CALL_EXTENSIONS);
        });

        const KERNEL_CODE: u16 = 0x08;  
        const KERNEL_DATA: u16 = 0x10;  
        const USER_CODE: u16 = 0x1B;    
        const USER_DATA: u16 = 0x23;    

        
        let star_value: u64 = ((USER_CODE as u64) << 48) | ((KERNEL_CODE as u64) << 32);

        use x86_64::registers::model_specific::Msr;
        let mut star_msr = Msr::new(0xC0000081);
        star_msr.write(star_value);

        
        LStar::write(VirtAddr::new(syscall_entry as u64));

        
        SFMask::write(
            RFlags::INTERRUPT_FLAG | 
            RFlags::DIRECTION_FLAG | 
            RFlags::IOPL_LOW | 
            RFlags::IOPL_HIGH
        );
    }
}

#[naked]
unsafe extern "C" fn syscall_entry() {
    naked_asm!(
        "swapgs",
        "mov gs:16, rsp",
        "mov rsp, gs:8",
        
        "push rcx",
        "push r11",
        "push rax",
        "push rdi",
        "push rsi",
        "push rdx",
        "push r8",
        "push r9",
        "push r10",
        
        "mov rdi, rsp",
        "call {}",
        
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rdx",
        "pop rsi",
        "pop rdi",
        "pop rax",
        "pop r11",
        "pop rcx",
        
        "mov rsp, gs:16",
        "swapgs",
        "sysretq",
        
        sym handle_syscall
    );
}

#[derive(Debug)]
#[repr(C)]
struct SyscallRegisters {
    r10: u64,
    r9: u64,
    r8: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    rax: u64,
    r11: u64,
    rcx: u64,
}

fn validate_user_buffer(addr: *const u8, size: usize, write: bool) -> Result<(), SyscallError> {
    let virt_addr = VirtAddr::new(addr as u64);
    let _current_process = PROCESS_LIST.lock().current()
        .ok_or(SyscallError::ProcessError)?;
    
    if virt_addr.as_u64() >= 0x8000_0000_0000 {
        return Err(SyscallError::InvalidAddress);
    }

    let _flags = if write {
        PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE
    } else {
        PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE
    };

    let start_page = Page::<Size4KiB>::containing_address(virt_addr);
    let end_addr = VirtAddr::new(virt_addr.as_u64() + size as u64 - 1);
    let end_page = Page::<Size4KiB>::containing_address(VirtAddr::new(end_addr.as_u64()));

    for page in Page::range_inclusive(start_page, end_page) {
        if !page.start_address().as_u64() < 0x8000_0000_0000 {
            return Err(SyscallError::InvalidAddress);
        }
    }

    Ok(())
}

fn validate_kernel_access(addr: VirtAddr, size: usize) -> Result<(), SyscallError> {
    if addr.as_u64() < 0xffff800000000000 {
        return Err(SyscallError::InvalidAddress);
    }

    let end_addr = addr.as_u64().checked_add(size as u64)
        .ok_or(SyscallError::InvalidAddress)?;

    let start_page = Page::<Size4KiB>::containing_address(addr);
    let end_page = Page::containing_address(VirtAddr::new(end_addr));

    for page in Page::range_inclusive(start_page, end_page) {
        if !page.start_address().as_u64() >= 0xffff800000000000 {
            return Err(SyscallError::InvalidAddress);
        }
    }

    Ok(())
}


fn sys_open(path_ptr: *const u8, path_len: usize, flags: u32) -> u64 {
    if let Err(e) = validate_user_buffer(path_ptr, path_len, false) {
        return e.into();
    }

    let path = unsafe {
        let slice = core::slice::from_raw_parts(path_ptr, path_len);
        match core::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return u64::from(SyscallError::InvalidArgument),
        }
    };

    let operation = FSOperation::Create {
        path: String::from(path),
    };

    let mut fs = FILESYSTEM.lock();
    if let Err(e) = fs.verify_operation(&operation) {
        return translate_fs_error(e).into();
    }
    drop(fs);

    let mut fs = FILESYSTEM.lock();
    match fs.stat(path) {
        Ok(_) => {
            let mut process_list = PROCESS_LIST.lock();
            let current = match process_list.current_mut() {
                Some(p) => p,
                None => return translate_fs_error(FsError::ProcessError).into(),
            };
            
            let fd = current.fd_table.allocate(String::from(path), flags);
            fd as u64
        },
        Err(e) => translate_fs_error(e).into(),
    }
}

fn sys_close(fd: usize) -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return translate_fs_error(FsError::ProcessError).into(),
    };

    match current.fd_table.remove(fd) {
        Some(_) => 0,
        None => translate_fs_error(FsError::NotFound).into(),
    }
}

fn translate_fs_error(error: FsError) -> SyscallError {
    match error {
        FsError::NotFound => SyscallError::NotFound,
        FsError::AlreadyExists => SyscallError::InvalidArgument,
        FsError::InvalidName => SyscallError::InvalidArgument,
        FsError::PermissionDenied => SyscallError::InvalidPermissions,
        FsError::NotADirectory => SyscallError::InvalidArgument,
        FsError::NotAFile => SyscallError::InvalidArgument,
        FsError::IsDirectory => SyscallError::InvalidArgument,
        FsError::IoError => SyscallError::IOError,
        FsError::InvalidPath => SyscallError::InvalidArgument,
        FsError::SymlinkLoop => SyscallError::InvalidArgument,
        FsError::DirectoryNotEmpty => SyscallError::InvalidArgument,
        FsError::ProcessError => SyscallError::ProcessError,
        FsError::FileSystemError => SyscallError::FileSystemError,
        FsError::InvalidState => SyscallError::InvalidState,
    }
}
fn sys_read(fd: usize, buffer: *mut u8, size: usize) -> u64 {
    const MAX_READ_SIZE: usize = 1024 * 1024; 
    if size > MAX_READ_SIZE {
        return SyscallError::InvalidArgument.into();
    }

    if let Err(e) = validate_user_buffer(buffer as *const u8, size, true) {
        return e.into();
    }

    let operation = FSOperation::Write {
        path: String::new(), 
        data: Vec::new(),
    };

    let mut fs = FILESYSTEM.lock();
    if let Err(e) = fs.verify_operation(&operation) {
        return translate_fs_error(e).into();
    }
    drop(fs);

    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return translate_fs_error(FsError::ProcessError).into(),
    };

    let fd_entry = match current.fd_table.get_mut(fd) {
        Some(entry) => entry,
        None => return SyscallError::InvalidArgument.into(),
    };

    let mut fs = FILESYSTEM.lock();
    match fs.read_file(&fd_entry.path) {
        Ok(contents) => {
            if fd_entry.offset >= contents.len() {
                return 0; 
            }

            let available = contents.len() - fd_entry.offset;
            let read_size = core::cmp::min(size, available);

            unsafe {
                core::ptr::copy_nonoverlapping(
                    contents[fd_entry.offset..].as_ptr(),
                    buffer,
                    read_size
                );
            }

            fd_entry.offset = fd_entry.offset.checked_add(read_size)
                .unwrap_or(contents.len());

            read_size as u64
        },
        Err(e) => translate_fs_error(e).into(),
    }
}

fn sys_wait(pid: u64) -> u64 {
    let target_pid = if pid == u64::MAX {
        None
    } else {
        Some(ProcessId::from_u64(pid))
    };

    let mut process_list = PROCESS_LIST.lock();
    let current_pid = match process_list.current() {
        Some(p) => p.id(),
        None => return SyscallError::ProcessError.into(),
    };

    
    if let Some(current) = process_list.get_mut_by_id(current_pid) {
        
        let wait_result = if let Some((zombie_pid, status)) = current.reap_zombies()
            .into_iter()
            .find(|(pid, _)| target_pid.is_none() || target_pid == Some(*pid))
        {
            Ok((zombie_pid, status))
        } else {
            
            current.set_state(ProcessState::Blocked);
            
            
            let children = current.get_children();
            for child_pid in children {
                if target_pid.is_none() || target_pid == Some(child_pid) {
                    if let Some(child) = process_list.get_mut_by_id(child_pid) {
                        child.add_waiter(current_pid);
                    }
                }
            }
            Err(())
        };

        
        drop(process_list);

        match wait_result {
            Ok((child_pid, status)) => ((child_pid.0 << 32) | (status as u32) as u64) as u64,
            Err(()) => SyscallError::ProcessError.into(),
        }
    } else {
        SyscallError::ProcessError.into()
    }
}

fn sys_write(fd: usize, buffer: *const u8, size: usize) -> u64 {
    const MAX_WRITE_SIZE: usize = 1024 * 1024; 
    if size > MAX_WRITE_SIZE {
        return SyscallError::InvalidArgument.into();
    }

    if let Err(e) = validate_user_buffer(buffer, size, false) {
        return e.into();
    }

    let slice = unsafe {
        core::slice::from_raw_parts(buffer, size)
    };

    let operation = FSOperation::Write {
        path: String::new(),
        data: slice.to_vec(),
    };

    let mut fs = FILESYSTEM.lock();
    if let Err(e) = fs.verify_operation(&operation) {
        return translate_fs_error(e).into();
    }
    drop(fs);

    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return translate_fs_error(FsError::ProcessError).into(),
    };

    let fd_entry = match current.fd_table.get(fd) {
        Some(entry) => entry,
        None => return SyscallError::InvalidArgument.into(),
    };

    let slice = unsafe {
        core::slice::from_raw_parts(buffer, size)
    };

    let mut fs = FILESYSTEM.lock();
    match fs.write_file(&fd_entry.path, slice) {
        Ok(_) => size as u64,
        Err(e) => translate_fs_error(e).into(),
    }
}

fn sys_exit(status: i32) -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    let current_pid = match process_list.current() {
        Some(p) => p.id(),
        None => return 0,
    };

    
    let (parent_pid, children) = {
        if let Some(current) = process_list.get_by_id(current_pid) {
            (current.get_parent(), current.get_children())
        } else {
            (None, Vec::new())
        }
    };
    
    drop(process_list);

    fs::cleanup();

    
    {
        let mut process_list = PROCESS_LIST.lock();
        for child_pid in &children {
            if let Some(child) = process_list.get_mut_by_id(*child_pid) {
                child.relations.parent = Some(ProcessId(1));
            }
        }
    }

    
    let mut process_list = PROCESS_LIST.lock();
    if let Some(current) = process_list.get_mut_by_id(current_pid) {
        
        {
            let mut mm_lock = MEMORY_MANAGER.lock();
            if let Some(ref mut mm) = *mm_lock {
                if let Err(e) = current.cleanup(mm) {
                    serial_println!("Warning: Process cleanup failed: {:?}", e);
                }
                
                mm.page_table_cache.lock().release_page_table(current.page_table());
            }
        }

        
        current.make_zombie(status);
    }

    
    if let Some(parent_pid) = parent_pid {
        if let Some(parent) = process_list.get_mut_by_id(parent_pid) {
            parent.set_state(ProcessState::Ready);
        }
    }

    
    process_list.cleanup_process_relations(current_pid);

    0
}

fn sys_kill(pid: u64, sig: u32) -> u64 {
    let signal = match signals::Signal::from_u32(sig) {
        Some(s) => s,
        None => return SyscallError::InvalidArgument.into(),
    };

    let mut process_list = PROCESS_LIST.lock();
    let target = match process_list.get_mut_by_id(ProcessId::from_u64(pid)) {  
        Some(p) => p,
        None => return SyscallError::ProcessError.into(),
    };

    if let Err(_) = target.signal_state.send_signal(signal) {
        return SyscallError::InvalidArgument.into();
    }

    0
}

fn sys_sigaction(signum: u32, handler: u64, flags: u32) -> u64 {
    let signal = match signals::Signal::from_u32(signum) {
        Some(s) => s,
        None => return SyscallError::InvalidArgument.into(),
    };

    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return SyscallError::ProcessError.into(),
    };

    let handler = signals::SignalHandler {
        handler: VirtAddr::new(handler),
        mask: current.signal_state.mask,
        flags,
    };

    if let Err(_) = current.signal_state.set_handler(signal, handler) {
        return SyscallError::InvalidArgument.into();
    }

    0
}

fn sys_getpgrp() -> u64 {
    let process_list = PROCESS_LIST.lock();
    process_list.current()
        .map(|p| p.group_id.as_u64())
        .unwrap_or(0)
}

fn sys_setpgrp(pid: u64, pgid: u64) -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    let current_pid = match process_list.current() {
        Some(p) => p.id(),
        None => return SyscallError::ProcessError.into(),
    };

    
    let target_pid = if pid == 0 {
        current_pid
    } else {
        ProcessId::from_u64(pid)
    };

    
    let target_pgid = if pgid == 0 {
        ProcessGroupId::from_u64(target_pid.as_u64())
    } else {
        ProcessGroupId::from_u64(pgid)
    };

    
    if !process_list.group_contains_key(&target_pgid) {
        if process_list.create_group(target_pid).is_none() {
            return SyscallError::ProcessError.into();
        }
    }

    
    match process_list.add_to_group(target_pid, target_pgid) {
        Ok(_) => 0,
        Err(_) => SyscallError::ProcessError.into(),
    }
}

fn sys_setsid() -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    
    let (current_pid, current_group_id) = match process_list.current() {
        Some(p) => (p.id(), p.group_id),
        None => return SyscallError::ProcessError.into(),
    };

    
    if let Some(group) = process_list.group_get_mut(&current_group_id) {
        if group.get_leader() == current_pid {
            return SyscallError::InvalidPermissions.into();
        }
    }

    
    let new_group_id = ProcessGroupId::new();
    let new_session_id = SessionId::new();

    if let Some(current) = process_list.get_mut_by_id(current_pid) {
        current.group_id = new_group_id;
        current.session_id = Some(new_session_id);
        
        process_list.create_group(current_pid);
        if let Err(_) = process_list.add_to_group(current_pid, new_group_id) {
            return SyscallError::ProcessError.into();
        }

        let session = Session::new(current_pid, new_group_id);
        process_list.sessions.insert(new_session_id, session);
        
        new_session_id.as_u64()
    } else {
        SyscallError::ProcessError.into()
    }
}

fn sys_getsid(pid: u64) -> u64 {
    let process_list = PROCESS_LIST.lock();
    
    let target_pid = if pid == 0 {
        match process_list.current() {
            Some(p) => p.id(),
            None => return SyscallError::ProcessError.into(),
        }
    } else {
        ProcessId::from_u64(pid)
    };

    if let Some(process) = process_list.get_by_id(target_pid) {
        process.session_id.map_or(
            SyscallError::ProcessError.into(),
            |sid| sid.as_u64()
        )
    } else {
        SyscallError::ProcessError.into()
    }
}

fn sys_yield() -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    if let Some(current) = process_list.current_mut() {
        current.set_state(ProcessState::Ready);
    }
    0
}

fn sys_getpid() -> u64 {
    let process_list = PROCESS_LIST.lock();
    process_list.current()
        .map(|p| p.id().as_u64())
        .unwrap_or(0)
}

fn sys_fork() -> u64 {
    let mut parent_list = PROCESS_LIST.lock();
    let parent_pid = match parent_list.current() {
        Some(p) => p.id(),
        None => return SyscallError::ProcessError.into(),
    };

    let (parent_group_id, parent_memory) = {
        if let Some(parent) = parent_list.get_by_id(parent_pid) {
            let memory_clone = parent.memory.allocations.clone();
            (parent.group_id, memory_clone)
        } else {
            return SyscallError::ProcessError.into();
        }
    };

    let mut lock = MEMORY_MANAGER.lock();
    let mut memory_manager = lock.as_mut().unwrap();
    
    
    if let Some(parent) = parent_list.get_by_id(parent_pid) {
        for allocation in &parent_memory {
            let start_page: Page<Size4KiB> = Page::containing_address(allocation.address);
            let pages = (allocation.size + 4095) / 4096;
            
            for i in 0..pages {
                let page = start_page + i as u64;
                unsafe {
                    if let Ok(frame) = memory_manager.page_table.translate_page(page) {
                        
                        if let Ok((_, flush)) = memory_manager.page_table.unmap(page) {
                            flush.flush();
                            
                            
                            let flags = PageTableFlags::PRESENT 
                                | PageTableFlags::USER_ACCESSIBLE 
                                | PageTableFlags::BIT_9;  
                                
                            if let Ok(tlb) = memory_manager.page_table
                                .map_to(page, frame, flags, &mut memory_manager.frame_allocator)
                            {
                                tlb.flush();
                            } else {
                                
                                let original_flags = PageTableFlags::PRESENT 
                                    | PageTableFlags::WRITABLE 
                                    | PageTableFlags::USER_ACCESSIBLE;
                                if let Ok(tlb) = memory_manager.page_table
                                    .map_to(page, frame, original_flags, &mut memory_manager.frame_allocator)
                                {
                                    tlb.flush();
                                }
                                return SyscallError::MemoryError(MemoryError::PageMappingFailed).into();
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    let mut child = match Process::new_user(&mut memory_manager) {
        Ok(mut process) => {
            process.group_id = parent_group_id;
            
            if let Err(_) = parent_list.add_to_group(process.id(), parent_group_id) {
                if let Err(cleanup_err) = process.cleanup(&mut memory_manager) {
                    serial_println!("Warning: Failed to clean up child process: {:?}", cleanup_err);
                }
                return u64::from(SyscallError::ProcessError);
            }

            if let Err(e) = init_child_memory(&mut process, &parent_memory, &mut memory_manager) {
                if let Err(cleanup_err) = cleanup_failed_fork(&mut process, &mut memory_manager) {
                    serial_println!("Warning: Failed to clean up failed fork: {:?}", cleanup_err);
                }
                if let Err(cleanup_err) = process.cleanup(&mut memory_manager) {
                    serial_println!("Warning: Failed to clean up child process: {:?}", cleanup_err);
                }
                return u64::from(SyscallError::MemoryError(e));
            }

            process.fd_table = parent_list.get_by_id(parent_pid)
                .map(|p| p.fd_table.clone())
                .unwrap_or_else(|| FileDescriptorTable::new());
            
            process
        },
        Err(_) => return u64::from(SyscallError::ProcessError),
    };

    child.set_parent(Some(parent_pid));
    let child_pid = child.id();

    if let Err(_) = parent_list.add(child) {
        
        let mut child = Process::new_user(&mut memory_manager).unwrap();
        if let Err(e) = cleanup_failed_fork(&mut child, &mut memory_manager) {
            serial_println!("Warning: Failed to clean up failed fork: {:?}", e);
        }
        return u64::from(SyscallError::ProcessError);
    }

    if let Some(parent) = parent_list.get_mut_by_id(parent_pid) {
        parent.add_child(child_pid);
    }
    Process::update_sibling_links(&mut *parent_list, child_pid);

    child_pid.as_u64()
}

fn init_child_memory(
    child: &mut Process,
    parent_memory: &[MemoryAllocation],
    memory_manager: &mut MemoryManager,
) -> Result<(), MemoryError> {
    
    let mut process_list = PROCESS_LIST.lock();
    let parent_info = process_list.current()
        .ok_or(MemoryError::InvalidAddress)?;
    
    
    let parent_priority = parent_info.priority;
    let parent_time_slice = parent_info.remaining_time_slice;
    
    
    drop(process_list);

    for allocation in parent_memory {
        let region = UserSpaceRegion::new(
            allocation.address,
            allocation.size,
            PageTableFlags::PRESENT 
                | PageTableFlags::USER_ACCESSIBLE 
                | PageTableFlags::BIT_9  
        );
        
        memory_manager.map_user_region(region)?;

        let start_page: Page<Size4KiB> = Page::containing_address(allocation.address);
        let pages = (allocation.size + 4095) / 4096;
        
        for i in 0..pages {
            let page = start_page + i as u64;
            if let Ok(frame) = unsafe { memory_manager.page_table.translate_page(page) } {
                let mut ref_counts = PAGE_REF_COUNTS.lock();
                ref_counts.entry(frame.start_address())
                    .and_modify(|e| e.count += 1)
                    .or_insert(PageRefCount {
                        frame,
                        count: 2,
                    });
            }
        }

        child.memory.allocations.push(MemoryAllocation {
            address: allocation.address,
            size: allocation.size,
            layout: allocation.layout,
        });
        child.memory.total_allocated += allocation.size;
    }

    child.priority = parent_priority;
    child.remaining_time_slice = parent_time_slice;

    Ok(())
}

fn cleanup_failed_fork(
    child: &mut Process,
    memory_manager: &mut MemoryManager,
) -> Result<(), MemoryError> {
    for allocation in &child.memory.allocations {
        let start_page = Page::containing_address(allocation.address);
        let pages = (allocation.size + PAGE_SIZE - 1) / PAGE_SIZE;
        
        for i in 0..pages {
            let page = start_page + i as u64;
            if let Ok(frame) = unsafe { memory_manager.page_table.translate_page(page) } {
                
                let mut ref_counts = PAGE_REF_COUNTS.lock();
                if let Some(ref_count) = ref_counts.get_mut(&frame.start_address()) {
                    if ref_count.decrement() {
                        ref_counts.remove(&frame.start_address());
                        unsafe {
                            memory_manager.frame_allocator.deallocate_frame(frame);
                        }                    }
                }
            }
            
            unsafe {
                let _ = memory_manager.unmap_page(page);
            }
        }
    }
    
    
    child.memory.allocations.clear();
    child.memory.total_allocated = 0;
    
    Ok(())
}

fn sys_exec(program_ptr: *const u8) -> u64 {
    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return SyscallError::ProcessError.into(),
    };

    if let Err(_) = validate_user_buffer(program_ptr, core::mem::size_of::<u64>(), false) {
        return u64::from(SyscallError::InvalidAddress);
    }

    let elf_data = unsafe {
        core::slice::from_raw_parts(program_ptr, 4096)
    };

    let elf_loader = match ElfLoader::new(elf_data) {
        Ok(loader) => loader,
        Err(_) => return SyscallError::InvalidArgument.into(),
    };

    
    if let Err(_) = elf_loader.validate_segments() {
        return SyscallError::InvalidArgument.into();
    }

    let mut mm_lock = MEMORY_MANAGER.lock();
    if let Some(memory_manager) = mm_lock.as_mut() {
        
        if let Err(_) = current.cleanup(memory_manager) {
            return SyscallError::MemoryError(MemoryError::PageMappingFailed).into();
        }

        
        match elf_loader.load(memory_manager) {
            Ok(entry_point) => {
                current.set_instruction_pointer(entry_point.as_u64());
                current.set_stack_pointer(current.user_stack_top()
                    .map_or(0, |addr| addr.as_u64()));
                0
            },
            Err(_) => SyscallError::InvalidArgument.into(),
        }
    } else {
        SyscallError::ProcessError.into()
    }
}

fn sys_memalloc(size: usize) -> u64 {
    
    if size == 0 || size > MAX_PROCESS_MEMORY {
        return SyscallError::InvalidArgument.into();
    }

    
    let aligned_size = match size.checked_add(4095) {
        Some(s) => s & !4095,
        None => return SyscallError::InvalidArgument.into(),
    };

    
    let pages = match aligned_size.checked_div(4096) {
        Some(p) => p,
        None => return SyscallError::InvalidArgument.into(),
    };

    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return SyscallError::ProcessError.into(),
    };

    
    match current.memory.total_allocated.checked_add(aligned_size) {
        Some(total) if total <= MAX_PROCESS_MEMORY => {},
        _ => return SyscallError::MemoryLimitExceeded.into(),
    }

    let mut mm_lock = MEMORY_MANAGER.lock();
    let memory_manager = mm_lock.as_mut().unwrap();

    
    if !memory_manager.verify_memory_requirements(aligned_size) {
        if !memory_manager.handle_memory_pressure() {
            return SyscallError::MemoryLimitExceeded.into();
        }
    }

    let alloc_addr = current.memory.heap_start + current.memory.heap_size;

    let region = UserSpaceRegion::new(
        alloc_addr,
        aligned_size,
        PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
    );

    
    let frame = match memory_manager.allocate_in_zone(MemoryZoneType::User, pages) {
        Some(frame) => frame,
        None => return SyscallError::MemoryError(MemoryError::ZoneExhausted).into(),
    };

    
    if let Err(_) = memory_manager.map_user_region(region) {
        memory_manager.free_in_zone(frame, pages);
        return SyscallError::MemoryError(MemoryError::FrameAllocationFailed).into();
    }

    
    if let Some(new_heap_size) = current.memory.heap_size.checked_add(aligned_size) {
        current.memory.heap_size = new_heap_size;
    } else {
        return SyscallError::MemoryLimitExceeded.into();
    }

    if let Some(new_total) = current.memory.total_allocated.checked_add(aligned_size) {
        current.memory.total_allocated = new_total;
    } else {
        return SyscallError::MemoryLimitExceeded.into();
    }

    current.memory.allocations.push(MemoryAllocation {
        address: alloc_addr,
        size: aligned_size,
        layout: Layout::from_size_align(size, 4096).unwrap(),
    });

    alloc_addr.as_u64()
}

fn sys_memdealloc(addr: usize, size: usize) -> u64 {
    let virt_addr = VirtAddr::new(addr as u64);
    
    if let Err(e) = validate_user_buffer(addr as *const u8, size, true) {
        return e.into();
    }

    let mut process_list = PROCESS_LIST.lock();
    let current = match process_list.current_mut() {
        Some(p) => p,
        None => return SyscallError::ProcessError.into(),
    };

    let alloc_index = match current.memory.allocations.iter().position(|a| a.address == virt_addr) {
        Some(idx) => idx,
        None => return u64::from(SyscallError::InvalidAddress),
    };

    let allocation = current.memory.allocations.remove(alloc_index);
    if allocation.size != size {
        return SyscallError::InvalidArgument.into();
    }

    let mut mm_lock = MEMORY_MANAGER.lock();
    let memory_manager = mm_lock.as_mut().unwrap();
    let pages = (size + 4095) / 4096;
    let start_page = Page::containing_address(virt_addr);

    for i in 0..pages {
        let page = start_page + i as u64;
        unsafe {
            memory_manager.unmap_page(page).expect("Failed to unmap page");
        }
    }

    current.memory.total_allocated -= allocation.size;

    0
}

fn sys_vkfs_create(path_ptr: *const u8, path_len: usize) -> u64 {
    if let Err(e) = validate_user_buffer(path_ptr, path_len, false) {
        return e.into();
    }

    let path = unsafe {
        let slice = core::slice::from_raw_parts(path_ptr, path_len);
        match core::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return u64::from(SyscallError::InvalidArgument),
        }
    };

    let operation = FSOperation::Create {
        path: String::from(path),
    };

    let mut fs = FILESYSTEM.lock();
    match fs.verify_operation(&operation) {
        Ok(proof) => {
            VERIFICATION_REGISTRY.lock().register_proof(proof);
            0
        },
        Err(e) => translate_fs_error(e).into(),
    }
}

fn sys_vkfs_delete(path_ptr: *const u8, path_len: usize) -> u64 {
    if let Err(e) = validate_user_buffer(path_ptr, path_len, false) {
        return e.into();
    }

    let path = unsafe {
        let slice = core::slice::from_raw_parts(path_ptr, path_len);
        match core::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return u64::from(SyscallError::InvalidArgument),
        }
    };

    let operation = FSOperation::Delete {
        path: String::from(path),
    };

    let mut fs = FILESYSTEM.lock();
    match fs.verify_operation(&operation) {
        Ok(proof) => {
            VERIFICATION_REGISTRY.lock().register_proof(proof);
            0
        },
        Err(e) => translate_fs_error(e).into(),
    }
}

fn sys_vkfs_verify(path_ptr: *const u8, path_len: usize) -> u64 {
    if let Err(e) = validate_user_buffer(path_ptr, path_len, false) {
        return e.into();
    }

    let path = unsafe {
        let slice = core::slice::from_raw_parts(path_ptr, path_len);
        match core::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return u64::from(SyscallError::InvalidArgument),
        }
    };

    let mut fs = FILESYSTEM.lock();
    match fs.verify_path(path) {
        Ok(verified) => {
            if verified {
                0
            } else {
                u64::from(SyscallError::InvalidState)
            }
        },
        Err(e) => translate_fs_error(e).into(),
    }
}

#[no_mangle]
extern "C" fn handle_syscall(regs: &mut SyscallRegisters) -> u64 {
    SYSCALL_COUNT.fetch_add(1, Ordering::Relaxed);
    
    match regs.rax {
        1 => sys_exit(regs.rdi as i32),
        2 => sys_write(regs.rdi as usize, regs.rsi as *const u8, regs.rdx as usize),
        3 => sys_read(regs.rdi as usize, regs.rsi as *mut u8, regs.rdx as usize),
        4 => sys_yield(),
        5 => sys_getpid(),
        6 => sys_fork(),
        7 => sys_exec(regs.rdi as *const u8),
        8 => sys_memalloc(regs.rdi as usize),
        9 => sys_memdealloc(regs.rdi as usize, regs.rsi as usize),
        10 => sys_open(regs.rdi as *const u8, regs.rsi as usize, regs.rdx as u32),
        11 => sys_close(regs.rdi as usize),
        12 => sys_read(regs.rdi as usize, regs.rsi as *mut u8, regs.rdx as usize),
        13 => sys_write(regs.rdi as usize, regs.rsi as *const u8, regs.rdx as usize),
        17 => sys_kill(regs.rdi, regs.rsi as u32),
        18 => sys_sigaction(regs.rdi as u32, regs.rsi, regs.rdx as u32),
        22 => sys_getpgrp(),
        23 => sys_setpgrp(regs.rdi, regs.rsi),
        24 => sys_wait(regs.rdi),
        25 => sys_setsid(),
        26 => sys_getsid(regs.rdi),
        29 => sys_vkfs_create(regs.rdi as *const u8, regs.rsi as usize),
        30 => sys_vkfs_delete(regs.rdi as *const u8, regs.rsi as usize),
        31 => sys_read(regs.rdi as usize, regs.rsi as *mut u8, regs.rdx as usize),
        32 => sys_write(regs.rdi as usize, regs.rsi as *const u8, regs.rdx as usize),
        33 => sys_vkfs_verify(regs.rdi as *const u8, regs.rsi as usize),
        _ => u64::from(SyscallError::InvalidSyscall),
    }
}


pub fn push_to_keyboard_buffer(c: u8) {
    interrupts::without_interrupts(|| {
        let mut buffer = KEYBOARD_BUFFER.lock();
        if buffer.len() < 1024 {
            buffer.push(c);
            serial_println!("Added to keyboard buffer: {} ({})", c as char, c);
        } else {
            serial_println!("Keyboard buffer full!");
        }
    });
}
pub fn clear_keyboard_buffer() {
    let mut buffer = KEYBOARD_BUFFER.lock();
    buffer.clear();
}


pub fn get_syscall_count() -> usize {
    SYSCALL_COUNT.load(Ordering::Relaxed)
}

pub fn reset_syscall_count() {
    SYSCALL_COUNT.store(0, Ordering::Relaxed);
}

pub fn init_process_memory(process: &mut Process) -> Result<(), SyscallError> {
    process.memory = ProcessMemory::new();
    
    let region = UserSpaceRegion::new(
        process.memory.heap_start,
        USER_HEAP_SIZE,
        PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
    );

    let mut memory_manager_lock = MEMORY_MANAGER.lock();
    let memory_manager = memory_manager_lock.as_mut()
        .ok_or(SyscallError::ProcessError)?;

    memory_manager.map_user_region(region)
        .map_err(|e| SyscallError::MemoryError(e))?;
    Ok(())
}


pub const SYSCALL_TABLE: &[(&str, usize)] = &[
    ("exit", 1),
    ("write", 2),
    ("read", 3),
    ("yield", 4),
    ("getpid", 5),
    ("fork", 6),
    ("exec", 7),
    ("memalloc", 8),
    ("memdealloc", 9),
    ("open", 10),
    ("close", 11),
    ("file_read", 12),
    ("file_write", 13),
    ("stat", 14),
    ("create", 15),
    ("delete", 16),
    ("vkfs_create", 29),
    ("vkfs_delete", 30), 
    ("vkfs_read", 31),
    ("vkfs_write", 32),
    ("vkfs_verify", 33),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_syscall_validation() {
        let mut regs = SyscallRegisters {
            r10: 0, r9: 0, r8: 0, rdx: 0, rsi: 0, rdi: 0,
            rax: 999,
            r11: 0, rcx: 0,
        };
        assert_eq!(handle_syscall(&mut regs), SyscallError::InvalidSyscall as u64);
    }

    #[test_case]
    fn test_memory_validation() {
        let invalid_addr = 0xFFFF_FFFF_FFFF_FFFF as *const u8;
        assert!(validate_user_buffer(invalid_addr, 100, false).is_err());
    }
}