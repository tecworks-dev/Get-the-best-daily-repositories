/*
This file is part of GigaPaste.

GigaPaste is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GigaPaste is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GigaPaste. If not, see <https://www.gnu.org/licenses/>.
*/

package main

import (
	
	"database/sql"
	"fmt"
	"os"
	"time"
	"strconv"

)

func CheckExpiration(db *sql.DB){

	for{

		tx, err := db.Begin()
		if err != nil {
			fmt.Println(err)
		}

		//Sqlite doesn't support select for update, instead when we begin transaction it locks the whole db file
		rows, err := tx.Query("SELECT id, filePath FROM data WHERE expire <= ?", strconv.FormatInt(time.Now().Unix(), 10))
		if err != nil{
			tx.Rollback()
			fmt.Print(err)
			return
		}
	
		var toDelete = []struct {
			
			Id string
			FilePath string

		}{}

		for rows.Next() {

			var id string
			var filePath string

			err = rows.Scan(&id, &filePath)
			if err != nil {
				tx.Rollback()
				fmt.Println(err)
				rows.Close()
				return
			}

			//dont wanna clutter the file with type struct
			//we cant' directly remove the rows from database because it will be locked before we call rows.Close()
			//so we just put things we want to delete into array
			toDelete = append(toDelete, struct{Id string; FilePath string}{Id: id, FilePath: filePath})
		
		}

		rows.Close()

		for _, v := range toDelete {

			_, err = tx.Exec("DELETE FROM data WHERE id = ?", v.Id)
			if err != nil {
				tx.Rollback()
				fmt.Println(err)
				return
			}

			err = os.Remove(v.FilePath)
			if err != nil {
				fmt.Println(err)
			}
	
		}

		if err := tx.Commit(); err != nil {
			fmt.Println(err)
			return;
	    }

		time.Sleep(10 * time.Second)
		
		
	}

}
