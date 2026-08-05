package main

import (
	"log"
	"net/http"
	"sync"
	"io"
    "fmt"
    "os"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Client struct {
	conn *websocket.Conn
	send chan []byte
}

// Store active websocket clients per stream
var clients = make(map[string]map[*Client]bool)
var clientsMu sync.Mutex

var frameCounter int
var frameCounterMu sync.Mutex

func handlePushFrame(w http.ResponseWriter, r *http.Request) {
	streamID := r.URL.Query().Get("stream")
	if streamID == "" {
		http.Error(w, "missing stream id", 400)
		return
	}

	frameData, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "bad body", 400)
		return
	}
    
    frameCounterMu.Lock()
	frameCounter++
	frameCounterMu.Unlock()

	clientsMu.Lock()
	streamClients := clients[streamID]
	activeClientCount := len(streamClients)
	for client := range streamClients {
		select {
		case client.send <- frameData:
		default:
			// Buffer full, drop frame to avoid blocking!
		}
	}
	clientsMu.Unlock()
	
	if activeClientCount == 0 {
		http.Error(w, "no active clients", 410)
		return
	}
    fmt.Fprintf(w, "ok")
}

func handleStreamViewer(w http.ResponseWriter, r *http.Request) {
	streamID := r.URL.Query().Get("stream")
	if streamID == "" {
		http.Error(w, "missing stream id", 400)
		return
	}

	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
    
    log.Printf("New browser client connected to Stream %s\n", streamID)

	client := &Client{
		conn: ws,
		send: make(chan []byte, 10000), // Massive buffer to hold frames if browser is busy loading video
	}

	clientsMu.Lock()
	if clients[streamID] == nil {
		clients[streamID] = make(map[*Client]bool)
	}
	clients[streamID][client] = true
	clientsMu.Unlock()

	// Write pump
	go func() {
		defer func() {
			client.conn.Close()
		}()
		for message := range client.send {
			err := client.conn.WriteMessage(websocket.TextMessage, message)
			if err != nil {
				return
			}
		}
	}()

	// Read pump (keeps connection alive and detects disconnect)
	for {
		if _, _, err := client.conn.ReadMessage(); err != nil {
			clientsMu.Lock()
			delete(clients[streamID], client)
			clientsMu.Unlock()
            log.Printf("Browser client disconnected from Stream %s\n", streamID)
			break
		}
	}
}

func handleVideoServe(w http.ResponseWriter, r *http.Request) {
	// CORS headers must be set BEFORE any response body — Safari requires these for cross-origin srcdoc iframes
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Range")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")

	// Handle CORS preflight (Safari sends OPTIONS before GET for cross-origin binary files)
	if r.Method == "OPTIONS" {
		w.WriteHeader(200)
		return
	}

	path := r.URL.Query().Get("path")
    log.Printf("Browser requesting file: %s\n", path)
	if path == "" {
		http.Error(w, "missing path", 400)
		return
	}
	
	file, err := os.Open(path)
	if err != nil {
		log.Printf("Error opening file %s: %v\n", path, err)
		http.Error(w, "file not found", 404)
		return
	}
	defer file.Close()
	
	stat, err := file.Stat()
	if err != nil {
		http.Error(w, "file stat error", 500)
		return
	}
	
	http.ServeContent(w, r, stat.Name(), stat.ModTime(), file)
}

func main() {
	http.HandleFunc("/push_frame", handlePushFrame)
	http.HandleFunc("/view_stream", handleStreamViewer)
	http.HandleFunc("/video", handleVideoServe)
	
	// Serve static files to bypass Streamlit's MIME type bugs on macOS
	fs := http.FileServer(http.Dir("../static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	
	log.Println("VisionTrack Unclogged Gateway Starting on :8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

