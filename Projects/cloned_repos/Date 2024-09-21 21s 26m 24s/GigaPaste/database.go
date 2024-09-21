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
	_ "github.com/mattn/go-sqlite3"

)

func initDatabase() *sql.DB {

	db, err := sql.Open("sqlite3", "file:./data/database.db?cache=shared")
    if err != nil {
        fmt.Println(err)
		return nil
    }

	db.SetMaxOpenConns(1)

    // Create a table
    createTableSQL := `CREATE TABLE IF NOT EXISTS data (
        id TEXT NOT NULL,
        type TEXT NOT NULL,
        fileName TEXT NOT NULL,
		filePath TEXT NOT NULL,
		burn TEXT NOT NULL,
		expire TEXT NOT NULL,
		passwordHash TEXT NOT NULL,
		passwordSalt TEXT NOT NULL,
		encryptSalt TEXT NOT NULL
    );`

    _, err = db.Exec(createTableSQL)
    if err != nil {
        fmt.Println(err)
		return nil
    }	

	return db

}
