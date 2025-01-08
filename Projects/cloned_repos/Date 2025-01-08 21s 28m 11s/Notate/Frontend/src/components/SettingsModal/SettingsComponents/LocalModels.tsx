import AddNewModel from "./AddNewModel";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { useUser } from "@/context/useUser";
import { useSysSettings } from "@/context/useSysSettings";
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "@/hooks/use-toast";

export default function LocalModels() {
  const [selectedModel, setSelectedModel] = useState<string>("");
  const { activeUser } = useUser();
  const {
    localModels,
    handleRunOllama,
    localModalLoading,
    progressLocalOutput,
    progressRef,
  } = useSysSettings();

  return (
    <div className="space-y-4">
      <Select value={selectedModel} onValueChange={setSelectedModel}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select a local model" />
        </SelectTrigger>
        <SelectContent>
          {localModels.map((model) => (
            <SelectItem key={model.digest || model.name} value={model.name}>
              {model.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Button
        onClick={() => {
          if (activeUser) {
            handleRunOllama(selectedModel, activeUser);
            toast({
              title: "Model loading",
              description: `Loading ${selectedModel}...`,
            });
          }
        }}
        disabled={localModalLoading || !selectedModel}
        className={`w-full relative ${
          localModalLoading ? "animate-pulse bg-primary/80" : ""
        }`}
      >
        {localModalLoading ? (
          <div className="flex items-center justify-center gap-2">
            <Loader2 className="animate-spin h-4 w-4" />
            <span>Starting Model...</span>
          </div>
        ) : (
          <>Run Model</>
        )}
      </Button>

      {progressLocalOutput.length > 0 && (
        <div
          ref={progressRef}
          className="mt-4 bg-secondary/50 rounded-md p-4 h-48 overflow-y-auto font-mono text-xs"
        >
          {progressLocalOutput.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap">
              {line}
            </div>
          ))}
        </div>
      )}

      <AddNewModel />
    </div>
  );
}
