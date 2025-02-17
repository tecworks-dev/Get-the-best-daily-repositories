/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"net/http"
)

func main() {
	ch := make(chan struct{}, 1)

	go func() {
		http.HandleFunc("/preheated", func(w http.ResponseWriter, r *http.Request) {
			fmt.Println("request arrived")

			if r.Method != http.MethodPost {
				http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
				return
			}
			ch <- struct{}{}
		})

		fmt.Println("start the server")
		err := http.ListenAndServe("0.0.0.0:9090", nil)
		if err != nil {
			panic(err)
		}
	}()

	// waiting for callback
	<-ch
	fmt.Printf("preheat successfully")
}
