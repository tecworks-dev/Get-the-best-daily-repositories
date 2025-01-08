import { Download, Loader2 } from "lucide-react";

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { Input } from "@/components/ui/input";
import { useSysSettings } from "@/context/useSysSettings";
import { useUser } from "@/context/useUser";
import { toast } from "@/hooks/use-toast";

export default function AddNewModel() {
  const [newModelName, setNewModelName] = useState("");
  const [isAddingModel, setIsAddingModel] = useState(false);
  const {
    localModalLoading,
    progressRef,
    setProgressLocalOutput,
    progressLocalOutput,
    handleRunOllama,
  } = useSysSettings();
  const { activeUser } = useUser();

  useEffect(() => {
    if (progressRef.current) {
      progressRef.current.scrollTop = progressRef.current.scrollHeight;
    }
  }, [progressLocalOutput, progressRef]);

  useEffect(() => {
    const handleProgress = (
      _event: Electron.IpcRendererEvent,
      data: OllamaProgressEvent | string
    ) => {
      if (typeof data === "string") {
        setProgressLocalOutput((prev) => [...prev, data]);
      } else {
        setProgressLocalOutput((prev) => [...prev, data.output]);
      }
    };

    window.electron.on("ollama-progress", handleProgress);

    return () => {
      window.electron.removeListener("ollama-progress", handleProgress);
    };
  }, [setProgressLocalOutput]);

  return (
    <Dialog open={isAddingModel} onOpenChange={setIsAddingModel}>
      <DialogTrigger asChild>
        <Button variant="outline" className="w-full mt-4">
          <Plus className="mr-2 h-4 w-4" />
          Add New Model
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add a New Model</DialogTitle>
          <DialogDescription>
            Enter the name of a model from{" "}
            <a
              href="https://ollama.com/search"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Ollama Library
            </a>
            , or use any Hugging Face GGUF model by prefixing with "hf.co/"
            (e.g. hf.co/TheBloke/Mistral-7B-v0.1-GGUF).
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <Input
            placeholder="Enter model name"
            value={newModelName}
            onChange={(e) => setNewModelName(e.target.value)}
          />
          <Button
            disabled={localModalLoading}
            onClick={() => {
              if (activeUser) {
                handleRunOllama(newModelName, activeUser);
                toast({
                  title: "Model pulling",
                  description: `Pulling ${newModelName}...`,
                });
              }
            }}
            className="w-full"
          >
            {localModalLoading ? (
              <div className="flex items-center justify-center gap-2">
                <Loader2 className="animate-spin h-4 w-4" />
                <span>Pulling Model...</span>
              </div>
            ) : (
              <>
                <Download className="mr-2 h-4 w-4" />
                Pull Model
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
