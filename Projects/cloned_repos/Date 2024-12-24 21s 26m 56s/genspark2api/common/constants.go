package common

import "time"

var StartTime = time.Now().Unix() // unit: second
var Version = "v1.0.1"            // this hard coding will be replaced automatically when building, no need to manually change

var DefaultOpenaiModelList = []string{
	"gpt-4o",
	"gpt-4o-mini",
	"o1-preview",
	"claude-3-5-sonnet",
	"claude-3-5-haiku",
	"gemini-1.5-pro",
	"gemini-1.5-flash",

	"flux",
	"flux-speed",
	"flux-pro/ultra",
	"ideogram",
	"recraft-v3",
	"dall-e-3",
}
