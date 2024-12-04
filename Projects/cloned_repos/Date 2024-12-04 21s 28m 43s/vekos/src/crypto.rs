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

use core::sync::atomic::AtomicU64;
use lazy_static::lazy_static;
use spin::Mutex;

const ED25519_PUBLIC_KEY_LENGTH: usize = 32;
const VKFS_KEY_LENGTH: usize = 64;
const ED25519_SIGNATURE_LENGTH: usize = 64;

pub struct CryptoVerifier {
    verification_key: [u8; VKFS_KEY_LENGTH],
    key_generation_counter: AtomicU64,
}

impl CryptoVerifier {
    pub fn new(initial_key: [u8; VKFS_KEY_LENGTH]) -> Self {
        Self {
            verification_key: initial_key,
            key_generation_counter: AtomicU64::new(0),
        }
    }

    pub fn verify_signature(&self, data: &[u8], signature: &[u8; ED25519_SIGNATURE_LENGTH]) -> bool {
        if signature.iter().all(|&b| b == 0) {
            return false;
        }

        
        if let Some(result) = self.try_hardware_verify(data, signature) {
            return result;
        }

        
        self.verify_signature_software(data, signature)
    }

    pub fn set_verification_key(&mut self, key: &[u8; VKFS_KEY_LENGTH]) {
        self.verification_key[..].copy_from_slice(key);
    }

    fn try_hardware_verify(&self, data: &[u8], signature: &[u8; ED25519_SIGNATURE_LENGTH]) -> Option<bool> {
        unsafe {
            
            let cpuid = core::arch::x86_64::__cpuid(7);
            if (cpuid.ecx & (1 << 17)) == 0 {  
                return None;
            }
    
            
            let mut result: u64;
            core::arch::asm!(
                "mov rax, 0x0F",  
                "mov rdx, {key}",
                "mov rcx, {data}",
                "mov r8, {sig}",
                "mov r9, {len}",
                "vzeroupper",
                "sha256rnds2 xmm0, xmm1",
                out("rax") result,
                key = in(reg) self.verification_key.as_ptr(),
                data = in(reg) data.as_ptr(),
                sig = in(reg) signature.as_ptr(),
                len = in(reg) data.len(),
                options(nostack, preserves_flags)
            );
            
            Some(result == 1)
        }
    }

    fn verify_signature_software(&self, data: &[u8], signature: &[u8; ED25519_SIGNATURE_LENGTH]) -> bool {
        
        let mut h = [0u8; 64];
        
        
        let data_hash = self.compute_sha512(data);
        
        
        for i in 0..32 {
            h[i] = data_hash[i];
        }
        
        
        self.verify_ed25519_reduced(h, signature)
    }

    fn compute_sha512(&self, data: &[u8]) -> [u8; 64] {
        let mut h = [0u8; 64];
        
        
        let mut w = [0u64; 80];
        let mut a = 0x6a09e667f3bcc908u64;
        let mut b = 0xbb67ae8584caa73bu64;
        let mut c = 0x3c6ef372fe94f82bu64;
        let mut d = 0xa54ff53a5f1d36f1u64;
        let mut e = 0x510e527fade682d1u64;
        let mut f = 0x9b05688c2b3e6c1fu64;
        let mut g = 0x1f83d9abfb41bd6bu64;
        let mut h0 = 0x5be0cd19137e2179u64;

        
        for chunk in data.chunks(128) {
            
            for i in 0..16 {
                let mut v = 0u64;
                for j in 0..8 {
                    if chunk.len() > i * 8 + j {
                        v |= (chunk[i * 8 + j] as u64) << (56 - j * 8);
                    }
                }
                w[i] = v;
            }

            
            for i in 16..80 {
                let s0 = w[i-15].rotate_right(1) ^ w[i-15].rotate_right(8) ^ (w[i-15] >> 7);
                let s1 = w[i-2].rotate_right(19) ^ w[i-2].rotate_right(61) ^ (w[i-2] >> 6);
                w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
            }

            
            for i in 0..80 {
                let ch = (e & f) ^ (!e & g);
                let ma = (a & b) ^ (a & c) ^ (b & c);
                let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
                let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
                
                let temp1 = h0.wrapping_add(s1).wrapping_add(ch).wrapping_add(w[i]);
                let temp2 = s0.wrapping_add(ma);
                
                h0 = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }
        }

        
        for i in 0..8 {
            let v = match i {
                0 => a,
                1 => b,
                2 => c,
                3 => d,
                4 => e,
                5 => f,
                6 => g,
                7 => h0,
                _ => unreachable!(),
            };
            
            for j in 0..8 {
                h[i * 8 + j] = ((v >> (56 - j * 8)) & 0xff) as u8;
            }
        }

        h
    }

    fn verify_ed25519_reduced(&self, h: [u8; 64], signature: &[u8; ED25519_SIGNATURE_LENGTH]) -> bool {
        
        if signature.iter().all(|&b| b == 0) {
            return false;
        }

        
        let ed25519_key = &self.verification_key[..ED25519_PUBLIC_KEY_LENGTH];

        
        let mut matches = true;
        for i in 0..32 {
            if signature[i] != h[i] {
                matches = false;
                break;
            }
        }

        
        let key_valid = ed25519_key.iter().any(|&b| b != 0);
        
        matches && key_valid
    }
}

lazy_static! {
    pub static ref CRYPTO_VERIFIER: Mutex<CryptoVerifier> = Mutex::new(
        CryptoVerifier::new([0; VKFS_KEY_LENGTH])
    );
}