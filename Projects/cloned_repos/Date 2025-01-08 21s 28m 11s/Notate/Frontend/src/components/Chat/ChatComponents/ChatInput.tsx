import { LibraryModal } from "@/components/CollectionModals/LibraryModal";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogTitle,
  DialogHeader,
  DialogContent,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Library, Send, X, Mic, Loader2 } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { useUser } from "@/context/useUser";
import { useSysSettings } from "@/context/useSysSettings";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WebAudioRecorder } from "@/utils/webAudioRecorder";
import { useLibrary } from "@/context/useLibrary";

export function ChatInput() {
  const {
    activeUser,
    handleChatRequest,
    cancelRequest,
    input,
    setInput,
    isLoading,
  } = useUser();
  const { openLibrary, setOpenLibrary } = useLibrary();
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptionLoading, setTranscriptionLoading] = useState(false);
  const [loadingDots, setLoadingDots] = useState("");
  const { isFFMPEGInstalled } = useSysSettings();
  const audioRecorder = useMemo(() => new WebAudioRecorder(), []);
  const { selectedCollection } = useLibrary();
  // Animate loading dots
  useEffect(() => {
    if (transcriptionLoading) {
      const interval = setInterval(() => {
        setLoadingDots((prev) => {
          if (prev === "...") return "";
          return prev + ".";
        });
      }, 500);
      return () => clearInterval(interval);
    } else {
      setLoadingDots("");
    }
  }, [transcriptionLoading]);

  const handleRecording = async () => {
    try {
      if (!isRecording) {
        await audioRecorder.startRecording();
        setIsRecording(true);
      } else {
        setTranscriptionLoading(true);
        const audioData = await audioRecorder.stopRecording();
        setIsRecording(false);
        setIsRecording(false);
        if (!activeUser?.id) {
          console.error("No active user ID found");
          setTranscriptionLoading(false);
          return;
        }
        const result = await window.electron.transcribeAudio(
          audioData,
          activeUser.id
        );

        if (!result.success) {
          console.error("Failed to transcribe audio:", result.error);
          setTranscriptionLoading(false);
          return;
        }

        if (result.transcription) {
          setInput((prev) => {
            const newInput = prev + (prev ? " " : "") + result.transcription;
            return newInput;
          });
        } else {
          console.warn("No transcription in result:", result);
        }

        setTranscriptionLoading(false);
      }
    } catch (error) {
      console.error("Error handling recording:", error);
      setIsRecording(false);
      setTranscriptionLoading(false);
    }
  };

  const getTooltipContent = () => {
    if (!isFFMPEGInstalled) {
      return "Please install FFMPEG to use voice-to-text";
    }
    if (transcriptionLoading) {
      return "Transcribing your audio...";
    }
    if (isRecording) {
      return "Click to stop recording";
    }
    return "Click to start voice recording";
  };

  return (
    <div className="p-4 bg-card border-t border-secondary">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (input.trim()) {
            handleChatRequest(selectedCollection?.id || undefined);
          }
        }}
        className="flex w-full items-center "
      >
        <div className="flex flex-col items-center">
          <Dialog open={openLibrary} onOpenChange={setOpenLibrary}>
            <DialogTrigger asChild>
              <Button
                type="button"
                size="icon"
                variant="outline"
                className="flex-shrink-0 rounded-none rounded-tl-[6px]"
              >
                <Library className="h-5 w-5" />
                <span className="sr-only">Library</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="max-h-[100vh] mt-4 overflow-y-auto p-6">
              <DialogHeader>
                <DialogTitle>Data Store Library</DialogTitle>
                <DialogDescription>
                  Select a data store to use for your chat.
                </DialogDescription>
              </DialogHeader>
              <LibraryModal />
            </DialogContent>
          </Dialog>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  size="icon"
                  disabled={transcriptionLoading}
                  variant={isRecording ? "destructive" : "outline"}
                  onClick={handleRecording}
                  className="flex-shrink-0 rounded-none rounded-bl-[6px] relative"
                >
                  {transcriptionLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Mic
                      className={`h-5 w-5 ${
                        isRecording ? "animate-pulse" : ""
                      } ${!isFFMPEGInstalled ? "opacity-50" : ""}`}
                    />
                  )}
                  {isRecording && (
                    <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-red-500 z-20 animate-pulse" />
                  )}
                  <span className="sr-only">
                    {transcriptionLoading
                      ? "Transcribing..."
                      : isRecording
                      ? "Stop Recording"
                      : "Start Recording"}
                  </span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{getTooltipContent()}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Textarea
          placeholder="Type your message here..."
          value={transcriptionLoading ? `Transcribing${loadingDots}` : input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (input.trim()) {
                handleChatRequest(selectedCollection?.id || undefined);
              }
            }
          }}
          disabled={transcriptionLoading}
          data-testid="chat-input"
          className={`z-10   max-h-[72px] min-h-[72px] flex-grow bg-background text-foreground placeholder-muted-foreground border-secondary rounded-none transition-opacity duration-200 ${
            transcriptionLoading ? "opacity-50" : "opacity-100"
          }`}
        />
        <Button
          type="button"
          size="icon"
          variant={isLoading ? "destructive" : "outline"}
          onClick={
            isLoading
              ? cancelRequest
              : () => {
                  if (input.trim()) {
                    handleChatRequest(selectedCollection?.id || undefined);
                  }
                }
          }
          data-testid="chat-submit"
          className="flex-shrink-0 h-[72px] w-[36px] rounded-none rounded-r-[6px]"
        >
          {isLoading ? <X className="h-5 w-5" /> : <Send className="h-5 w-5" />}
          <span className="sr-only">{isLoading ? "Cancel" : "Send"}</span>
        </Button>
      </form>
    </div>
  );
}
