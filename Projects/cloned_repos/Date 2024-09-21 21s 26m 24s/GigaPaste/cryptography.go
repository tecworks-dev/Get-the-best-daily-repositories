/*
This file is part of GigaPaste.

GigaPaste is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GigaPaste is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GigaPaste. If not, see <https://www.gnu.org/licenses/>.
*/

package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"io"
	"os"
	"golang.org/x/crypto/pbkdf2"
	"time"
)

func EncryptFile(srcPath string, aesKey []byte) error {
	
	srcFile, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(srcPath + ".tmp")
	if err != nil {
		return err
	}
	defer dstFile.Close()

	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return err
	}

	//make nonce
	nonce := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return err
	}

	//write nonce
	if _, err := dstFile.Write(nonce); err != nil {
		return err
	}

	stream := cipher.NewCTR(block, nonce)

	//create stream writer that can encrypt in real time
	streamWriter := &cipher.StreamWriter{S: stream, W: dstFile}
	buffer := make([]byte, 1024*Global.StreamSizeLimit)
	for {
		n, err := srcFile.Read(buffer)
		if err != nil && err != io.EOF {
			return err
		}
		if n == 0 {
			break
		}

		streamWriter.Write(buffer[:n])
		if(Global.StreamThrottle > 0){
			time.Sleep(time.Duration(Global.StreamThrottle) * time.Millisecond)
		}
	}

	srcFile.Close()
	err = os.Remove(srcPath)
	if err != nil {
		return err
	}

	dstFile.Close()
	err = os.Rename(srcPath+".tmp", srcPath)
	if err != nil {
		return err
	}

	return nil
}

func GetDecryptInfo(srcPath string, aesKey []byte) (error, []byte, cipher.Stream, int) {

	srcFile, err := os.Open(srcPath)
	if err != nil {
		return err, nil, nil, -1
	}
	defer srcFile.Close()

	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return err, nil, nil, -1
	}

	iv := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(srcFile, iv); err != nil {
		return err, nil, nil, -1
	}

	stream := cipher.NewCTR(block, iv)
	return nil, iv, stream, aes.BlockSize

}



func DecryptFileStream(buffer []byte, size int, iv []byte, stream cipher.Stream) (error, []byte) {

	decrypted := make([]byte, size)
	stream.XORKeyStream(decrypted, buffer[:size])

	return nil, decrypted

}

func GenerateSalt() ([]byte, error) {

	salt := make([]byte, 16)
	_, err := rand.Read(salt)
	if err != nil {
		return nil, err
	}
	return salt, nil

}


func GeneratePasswordHash(password string, salt []byte) []byte {

	return pbkdf2.Key([]byte(password), (salt), Global.Pbkdf2Iteraions, 32, sha256.New)

}
