// A real HTTP server for push/pull. We'll store commits/object in the server's
// own .evo directory, or in memory. We'll illustrate a minimal appraoch.
package server

import (
	"fmt"
	"net/http"
	"strconv"
)

// EvoHTTPServer wraps Go's http.Server
type EvoHTTPServer struct {
	http.Server
}

func NewEvoHTTPServer(port int) *EvoHTTPServer {
	mux := http.NewServeMux()

	mux.HandleFunc("/pull", handlePull)
	mux.HandleFunc("/push", handlePush)
	mux.HandleFunc("/objects/", handleGetObject) // GET an object by hash
	mux.HandleFunc("/auth/login", handleLogin)
	mux.HandleFunc("/auth/register", handleRegister)

	return &EvoHTTPServer{
		http.Server{
			Addr:    ":" + strconv.Itoa(port),
			Handler: mux,
		},
	}
}

func (s *EvoHTTPServer) ListenAndServe() error {
	fmt.Println("Evo HTTP server listening on", s.Addr)
	return s.Server.ListenAndServe()
}
