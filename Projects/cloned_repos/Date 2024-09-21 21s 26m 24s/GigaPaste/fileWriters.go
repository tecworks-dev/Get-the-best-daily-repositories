/*
This file is part of GigaPaste.

GigaPaste is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GigaPaste is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GigaPaste. If not, see <https://www.gnu.org/licenses/>.
*/

package main

import (
	
	"fmt"
	"os"
	"time"
	"mime/multipart"
	"archive/zip"
	"io"

)

func MultipleFileWriter(files []*multipart.FileHeader, path string, aesKey []byte, callback func()){
	
	outFile, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer outFile.Close()
	
	//make new zip file
	zipWriter := zip.NewWriter(outFile)
	defer zipWriter.Close()

	for _, fileHeader := range files {

		file, err := fileHeader.Open()
		if err != nil {
			fmt.Println(err)
			return
		}
		defer file.Close()

		//create file inside the zip
		writer, err := zipWriter.Create(fileHeader.Filename)
		if err != nil {
			fmt.Println(err)
			return
		}

		buffer := make([]byte, 1024*Global.StreamSizeLimit)
		for {

			n, err := file.Read(buffer)
			if err != nil && err != io.EOF {
				fmt.Println(err)
				return
			}
			if n == 0 {
				break
			}

			// Write the chunk to the ZIP file
			_, err = writer.Write(buffer[:n])
			if err != nil {
				fmt.Println(err)
				return
			}

			//need to add check for > 0 because Sleep(0) will just trigger context switch
			if(Global.StreamThrottle > 0){
				time.Sleep(time.Duration(Global.StreamThrottle) * time.Millisecond)
			}

		}

	}

	zipWriter.Close()
	outFile.Close()

	if aesKey != nil {

		err = EncryptFile(path, aesKey)
		if err != nil {
			fmt.Println(err)
		}

	}	

	callback()

}

func SingleFileWriter(files []*multipart.FileHeader, path string, aesKey []byte, callback func()){

	outFile, err := os.Create(path) 
	if err != nil {
		fmt.Println(err)
		return
	}
	defer outFile.Close()

	file, err := files[0].Open()
	if err != nil {
		fmt.Println(err)
		return
	}

	defer file.Close()

	// Use a buffered reader to read the file in chunks
	buffer := make([]byte, 1024*Global.StreamSizeLimit)
	for {
		n, err := file.Read(buffer)
		if err != nil && err != io.EOF {
			fmt.Println(err)
			return
		}
		if n == 0 {
			break
		}

		// Write the chunk to the output file
		_, err = outFile.Write(buffer[:n])
		if err != nil {
			fmt.Println(err)
			return
		}

		//need to add check for > 0 because Sleep(0) will just trigger context switch
		if(Global.StreamThrottle > 0){
			time.Sleep(time.Duration(Global.StreamThrottle) * time.Millisecond)
		}

	}

	outFile.Close()
	if aesKey != nil {

		err = EncryptFile(path, aesKey)
		if err != nil {
			fmt.Println(err)
		}

	}

	callback()

}
