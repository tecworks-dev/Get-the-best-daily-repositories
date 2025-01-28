import {
  decryptStringSync,
  encryptStringSync,
  findKeyForMessage,
  makeKeychainSync,
  parseCloakedString,
  parseKeySync,
} from "@47ng/cloak";
import crypto from "crypto";
import { env } from "@/lib/env";

// Initialize keychain
const encryptionKey = parseKeySync(env.FIELD_ENCRYPTION_KEY);
const keychain = makeKeychainSync([env.FIELD_ENCRYPTION_KEY]);

// Main encryption/decryption functions
export function encrypt(value: string): string {
  return encryptStringSync(value, encryptionKey);
}

export function decrypt(encrypted: string): string {
  if (!parseCloakedString(encrypted)) {
    return encrypted; // Return as-is if not encrypted
  }
  const key = findKeyForMessage(encrypted, keychain);
  return decryptStringSync(encrypted, key);
}

// Hash generation using SHA-256
export function generateHash(value: string): string {
  return crypto.createHash("sha256").update(value).digest("hex");
}

// Utility function to verify if a hash matches the original value
export function verifyHash(value: string, hash: string): boolean {
  return generateHash(value) === hash;
}
