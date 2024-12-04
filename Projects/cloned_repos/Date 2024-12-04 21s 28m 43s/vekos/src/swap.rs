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

use alloc::vec::Vec;
use spin::Mutex;
use x86_64::{
    structures::paging::{Page, PageTableFlags, PhysFrame, Size4KiB},
    VirtAddr, PhysAddr,
};
use crate::{
    memory::{MemoryError, MemoryManager},
    fs::{FILESYSTEM, FileSystem, FilePermissions},
};

const SWAP_FILE: &str = "swap";
const PAGE_SIZE: usize = 4096;
const MAX_SWAP_PAGES: usize = 1024; 

pub struct SwapEntry {
    offset: usize,
    page: Page,
    flags: PageTableFlags,
}

pub struct SwapManager {
    free_slots: Vec<usize>,
    used_slots: Vec<Option<SwapEntry>>,
    total_slots: usize,
}

impl SwapManager {
    pub fn new() -> Self {
        
        let mut free_slots = Vec::with_capacity(MAX_SWAP_PAGES);
        for i in 0..MAX_SWAP_PAGES {
            free_slots.push(i);
        }

        Self {
            free_slots,
            used_slots: vec![None; MAX_SWAP_PAGES],
            total_slots: MAX_SWAP_PAGES,
        }
    }

    pub fn init(&self) -> Result<(), MemoryError> {
        let mut fs = FILESYSTEM.lock();
        
        
        if fs.stat(SWAP_FILE).is_err() {
            fs.create_file(
                SWAP_FILE,
                FilePermissions {
                    read: true,
                    write: true,
                    execute: false,
                }
            ).map_err(|_| MemoryError::SwapFileError)?;
            
            
            let zeros = vec![0u8; PAGE_SIZE * MAX_SWAP_PAGES];
            fs.write_file(SWAP_FILE, &zeros)
                .map_err(|_| MemoryError::SwapFileError)?;
        }
        
        Ok(())
    }

    pub fn swap_out(
        &mut self,
        page: Page,
        flags: PageTableFlags,
        memory_manager: &mut MemoryManager,
    ) -> Result<usize, MemoryError> {
        
        let slot = self.free_slots.pop()
            .ok_or(MemoryError::NoSwapSpace)?;
        
        let offset = slot * PAGE_SIZE;
        
        
        let virt_addr = page.start_address();
        let page_data = unsafe {
            core::slice::from_raw_parts(
                virt_addr.as_ptr::<u8>(),
                PAGE_SIZE
            )
        };

        
        let mut fs = FILESYSTEM.lock();
        fs.write_file(SWAP_FILE, page_data)
            .map_err(|_| MemoryError::SwapFileError)?;

        
        self.used_slots[slot] = Some(SwapEntry {
            offset,
            page,
            flags,
        });

        
        unsafe {
            memory_manager.unmap_page(page)?;
        }

        Ok(slot)
    }

    pub fn swap_in(
        &mut self,
        slot: usize,
        memory_manager: &mut MemoryManager,
    ) -> Result<(), MemoryError> {
        let entry = self.used_slots[slot].take()
            .ok_or(MemoryError::InvalidSwapSlot)?;

        
        let mut fs = FILESYSTEM.lock();
        let mut page_data = vec![0u8; PAGE_SIZE];
        fs.read_file(SWAP_FILE)
            .map_err(|_| MemoryError::SwapFileError)?;

        
        let frame = memory_manager.get_free_frame()?;
        unsafe {
            memory_manager.map_page(entry.page, frame, entry.flags)?;
            
            
            core::ptr::copy_nonoverlapping(
                page_data.as_ptr(),
                entry.page.start_address().as_mut_ptr::<u8>(),
                PAGE_SIZE
            );
        }

        
        self.free_slots.push(slot);
        
        Ok(())
    }
}


lazy_static::lazy_static! {
    pub static ref SWAP_MANAGER: Mutex<SwapManager> = Mutex::new(SwapManager::new());
}