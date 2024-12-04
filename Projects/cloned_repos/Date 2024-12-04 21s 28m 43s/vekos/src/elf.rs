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

use crate::memory::{MemoryManager, MemoryError, UserSpaceRegion};
use x86_64::structures::paging::PageTableFlags;
use x86_64::VirtAddr;
use alloc::vec::Vec;

#[derive(Debug)]
pub enum ElfError {
    InvalidMagic,
    InvalidClass,
    InvalidFormat,
    UnsupportedArchitecture,
    MemoryError(MemoryError),
    LoadError,
    InvalidSegment,
    SecurityViolation,
    UnsupportedFeature,
    InvalidAlignment,
    DynamicLinkingUnsupported,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProgramHeaderTable {
    headers: &'static [ProgramHeader],
    count: usize,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentType {
    Null = 0,
    Load = 1,
    Dynamic = 2,
    Interp = 3,
    Note = 4,
    SharedLib = 5,
    PHdr = 6,
    TLS = 7,
}

impl TryFrom<u32> for SegmentType {
    type Error = ElfError;
    
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SegmentType::Null),
            1 => Ok(SegmentType::Load),
            2 => Ok(SegmentType::Dynamic),
            3 => Ok(SegmentType::Interp),
            4 => Ok(SegmentType::Note),
            5 => Ok(SegmentType::SharedLib),
            6 => Ok(SegmentType::PHdr),
            7 => Ok(SegmentType::TLS),
            _ => Err(ElfError::InvalidSegment),
        }
    }
}

#[repr(C)]
pub struct ElfHeader {
    magic: [u8; 4],
    class: u8,
    data: u8,
    version: u8,
    os_abi: u8,
    abi_version: u8,
    padding: [u8; 7],
    type_: u16,
    machine: u16,
    version2: u32,
    entry: u64,
    phoff: u64,
    shoff: u64,
    flags: u32,
    ehsize: u16,
    phentsize: u16,
    phnum: u16,
    shentsize: u16,
    shnum: u16,
    shstrndx: u16,
}

#[repr(C)]
#[derive(Debug)]
pub struct ProgramHeader {
    type_: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    paddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

pub struct ElfLoader {
    header: &'static ElfHeader,
    program_headers: &'static [ProgramHeader],
    binary: &'static [u8],
}

impl ElfLoader {
    pub fn new(binary: &'static [u8]) -> Result<Self, ElfError> {
        if binary.len() < core::mem::size_of::<ElfHeader>() {
            return Err(ElfError::InvalidFormat);
        }

        let header = unsafe { &*(binary.as_ptr() as *const ElfHeader) };
        
        
        if header.magic != [0x7f, 0x45, 0x4c, 0x46] {
            return Err(ElfError::InvalidMagic);
        }

        
        if header.class != 2 {
            return Err(ElfError::InvalidClass);
        }

        
        if header.machine != 0x3e {
            return Err(ElfError::UnsupportedArchitecture);
        }

        let ph_offset = header.phoff as usize;
        let ph_size = header.phentsize as usize;
        let ph_count = header.phnum as usize;

        if binary.len() < ph_offset + ph_size * ph_count {
            return Err(ElfError::InvalidFormat);
        }

        let program_headers = unsafe {
            core::slice::from_raw_parts(
                (binary.as_ptr().add(ph_offset)) as *const ProgramHeader,
                ph_count
            )
        };

        Ok(ElfLoader {
            header,
            program_headers,
            binary,
        })
    }

    pub fn load(&self, memory_manager: &mut MemoryManager) -> Result<VirtAddr, ElfError> {
        
        self.validate_segments()?;

        let mut entry_point = None;

        for ph in self.program_headers {
            match SegmentType::try_from(ph.type_)? {
                SegmentType::Load => {
                    self.load_segment(memory_manager, ph)?;
                }
                SegmentType::Dynamic => {
                    return Err(ElfError::DynamicLinkingUnsupported);
                }
                SegmentType::TLS => {
                    return Err(ElfError::UnsupportedFeature);
                }
                _ => continue,
            }
        }

        
        entry_point = Some(VirtAddr::new(self.header.entry));

        
        if let Some(entry) = entry_point {
            if entry.as_u64() >= 0xffff_8000_0000_0000 {
                return Err(ElfError::SecurityViolation);
            }
        }

        entry_point.ok_or(ElfError::LoadError)
    }

    fn load_segment(&self, memory_manager: &mut MemoryManager, ph: &ProgramHeader) 
        -> Result<(), ElfError> 
    {
        let mem_size = ph.memsz as usize;
        let file_size = ph.filesz as usize;
        let file_offset = ph.offset as usize;

        
        let mut flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
        if ph.flags & 0x2 != 0 { 
            flags |= PageTableFlags::WRITABLE;
        }
        if ph.flags & 0x1 == 0 { 
            flags |= PageTableFlags::NO_EXECUTE;
        }

        
        let region = UserSpaceRegion::new(
            VirtAddr::new(ph.vaddr),
            mem_size,
            flags,
        );

        
        memory_manager.map_user_region(region)
            .map_err(ElfError::MemoryError)?;

        
        if file_size > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.binary.as_ptr().add(file_offset),
                    ph.vaddr as *mut u8,
                    file_size
                );
            }

            
            if mem_size > file_size {
                unsafe {
                    core::ptr::write_bytes(
                        (ph.vaddr as *mut u8).add(file_size),
                        0,
                        mem_size - file_size
                    );
                }
            }
        }

        Ok(())
    }

    pub fn validate_segments(&self) -> Result<(), ElfError> {
        
        let mut sorted_segments: Vec<_> = self.program_headers.iter()
            .filter(|ph| SegmentType::try_from(ph.type_).unwrap_or(SegmentType::Null) == SegmentType::Load)
            .collect();
        sorted_segments.sort_by_key(|ph| ph.vaddr);

        for i in 1..sorted_segments.len() {
            let prev = sorted_segments[i - 1];
            let curr = sorted_segments[i];
            if prev.vaddr + prev.memsz > curr.vaddr {
                return Err(ElfError::SecurityViolation);
            }
        }

        
        for ph in self.program_headers {
            
            if ph.align > 0 && ph.align & (ph.align - 1) != 0 {
                return Err(ElfError::InvalidAlignment);
            }

            
            if ph.vaddr + ph.memsz < ph.vaddr {
                return Err(ElfError::SecurityViolation);
            }

            
            if ph.vaddr >= 0xffff_8000_0000_0000 {
                return Err(ElfError::SecurityViolation);
            }
        }

        Ok(())
    }

    pub fn entry_point(&self) -> VirtAddr {
        VirtAddr::new(self.header.entry)
    }
}