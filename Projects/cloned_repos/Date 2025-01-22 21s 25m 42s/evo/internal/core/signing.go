// Optioanl commit signing with ed25519
package core

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"golang.org/x/crypto/scrypt"
	"golang.org/x/crypto/ssh/terminal"
)

// signCommit uses a passphrase-protected key in .evo/keys/ed25519_priv.enc
func signCommit(repoPath, commitHash string, user UserConfig) (string, error) {
	evoPath := filepath.Join(repoPath, EvoDir)
	keyDir := filepath.Join(evoPath, "keys")
	privFile := filepath.Join(keyDir, "ed25519_priv.enc")
	pubFile := filepath.Join(keyDir, "ed25519_pub")

	var privKey ed25519.PrivateKey
	if _, err := os.Stat(privFile); os.IsNotExist(err) {
		// generate new key
		fmt.Println("No existing key found, generating a new ed25519 key pair...")
		pub, priv, _ := ed25519.GenerateKey(rand.Reader)

		// ask passphrase
		pass, err := getPassphrase("Enter passphrase for new key (leave blank for no pass): ")
		if err != nil {
			return "", err
		}
		encPriv, err := encryptPrivateKey(priv, pass)
		if err != nil {
			return "", err
		}
		os.WriteFile(privFile, encPriv, 0600)

		// store pub in hex
		os.WriteFile(pubFile, []byte(hex.EncodeToString(pub)), 0644)
		privKey = priv
	} else {
		// read pub, read priv, decrypt
		pass, err := getPassphrase("Enter passphrase for your existing key: ")
		if err != nil {
			return "", err
		}
		encPriv, err := os.ReadFile(privFile)
		if err != nil {
			return "", err
		}
		priv, err := decryptPrivateKey(encPriv, pass)
		if err != nil {
			return "", err
		}
		privKey = ed25519.PrivateKey(priv)
	}

	sig := ed25519.Sign(privKey, []byte(commitHash))
	return hex.EncodeToString(sig), nil
}

// VerifyCommit checks the stored signature against the commit hash
func VerifyCommit(repoPath string, c *Commit) bool {
	if c.Signature == "" {
		return false
	}
	evoPath := filepath.Join(repoPath, EvoDir)
	pubFile := filepath.Join(evoPath, "keys", "ed25519_pub")
	pubHex, err := os.ReadFile(pubFile)
	if err != nil {
		return false
	}
	pub, _ := hex.DecodeString(string(pubHex))
	signatureBytes, err := hex.DecodeString(c.Signature)
	if err != nil {
		return false
	}
	// recompute commit's raw
	raw := fmt.Sprintf("%s|%s|%v|%s", c.Message, c.Author, c.Timestamp.UnixNano(), c.TreeHash)
	for _, p := range c.Parents {
		raw += "|" + p
	}
	sum := ed25519.SignatureSize // dummy usage
	_ = sum                      // ignore
	// let's re-hash
	// Actually, we hashed string(c.Hash) to sign. But let's do consistent approach:
	// We'll just see if public key verifies commitHash = c.Hash
	// So we do:
	commitHashBytes, err := hex.DecodeString(c.Hash)
	if err != nil {
		return false
	}
	return ed25519.Verify(ed25519.PublicKey(pub), commitHashBytes, signatureBytes)
}

// We’ll store something like: MAGIC(6 bytes) || SALT(16 bytes) || NONCE(gcm.NonceSize()) || CIPHERTEXT
var magic = []byte("EVOPK1") // short “EVO Private Key v1”

func encryptPrivateKey(key ed25519.PrivateKey, pass []byte) ([]byte, error) {
	// If user supplied no passphrase, store in plaintext
	if len(pass) == 0 {
		return key, nil
	}

	// 1) Generate random salt
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return nil, fmt.Errorf("cannot generate salt: %w", err)
	}

	// 2) Derive key from pass + salt using scrypt (parameters can be tuned)
	derivedKey, err := scrypt.Key(pass, salt, 1<<15, 8, 1, 32)
	if err != nil {
		return nil, fmt.Errorf("scrypt.Key: %w", err)
	}

	// 3) Prepare AES-GCM
	block, err := aes.NewCipher(derivedKey)
	if err != nil {
		return nil, fmt.Errorf("newCipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("gcm: %w", err)
	}

	// 4) Generate random nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("nonce: %w", err)
	}

	// 5) Encrypt
	ciphertext := gcm.Seal(nil, nonce, key, nil)

	// 6) Combine everything:
	// MAGIC || SALT || NONCE || CIPHERTEXT
	out := append([]byte{}, magic...)
	out = append(out, salt...)
	out = append(out, nonce...)
	out = append(out, ciphertext...)
	return out, nil
}

func decryptPrivateKey(enc []byte, pass []byte) ([]byte, error) {
	// If no passphrase, we interpret data as plaintext
	if len(pass) == 0 {
		return enc, nil
	}

	// Check MAGIC header
	if len(enc) < len(magic) || !bytes.Equal(enc[:len(magic)], magic) {
		return nil, errors.New("invalid encrypted key format (missing magic header)")
	}
	enc = enc[len(magic):] // skip magic

	// Must have at least salt (16B) + some nonce + ciphertext
	if len(enc) < 16 {
		return nil, errors.New("invalid encrypted key format (no salt)")
	}
	salt := enc[:16]
	enc = enc[16:]

	// Derive AES key
	derivedKey, err := scrypt.Key(pass, salt, 1<<15, 8, 1, 32)
	if err != nil {
		return nil, err
	}
	block, err := aes.NewCipher(derivedKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	if len(enc) < gcm.NonceSize() {
		return nil, errors.New("invalid encrypted key (no nonce)")
	}
	nonce := enc[:gcm.NonceSize()]
	ciphertext := enc[gcm.NonceSize():]

	// Decrypt
	plain, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, errors.New("decryption failed (wrong passphrase or corrupt data)")
	}
	return plain, nil
}

func getPassphrase(prompt string) ([]byte, error) {
	fmt.Print(prompt)
	pass, err := terminal.ReadPassword(int(syscall.Stdin))
	fmt.Println()
	return pass, err
}
