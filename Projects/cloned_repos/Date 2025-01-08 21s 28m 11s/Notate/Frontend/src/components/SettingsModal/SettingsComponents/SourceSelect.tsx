import { Button } from "@/components/ui/button";
import { useSysSettings } from "@/context/useSysSettings";
import { Globe, Shield, Sparkles } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function SourceSelect() {
  const { sourceType, setSourceType, isOllamaRunning, totalVRAM } =
    useSysSettings();

  return (
    <div className="flex gap-4">
      <Button
        variant={sourceType === "external" ? "default" : "outline"}
        className="flex-1 h-20"
        onClick={() => setSourceType("external")}
      >
        <div className="flex flex-col items-center gap-2">
          <Globe className="h-6 w-6" />
          <span>External API</span>
        </div>
      </Button>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex-1">
              <Button
                variant={sourceType === "local" ? "default" : "outline"}
                className={`w-full h-20 transition-all duration-300 ${
                  isOllamaRunning
                    ? "hover:shadow-lg hover:border-primary"
                    : "opacity-50 cursor-not-allowed"
                } ${
                  isOllamaRunning && sourceType !== "local"
                    ? "animate-pulse-subtle"
                    : ""
                }`}
                disabled={totalVRAM < 8 || !isOllamaRunning}
                onClick={() => setSourceType("local")}
              >
                <div className="flex flex-col items-center gap-2 relative">
                  {isOllamaRunning && sourceType !== "local" && (
                    <Sparkles className="h-4 w-4 text-primary absolute -right-6 -top-1 animate-sparkle" />
                  )}
                  <Shield className="h-6 w-6" />
                  <span>Local Models</span>
                </div>
              </Button>
            </div>
          </TooltipTrigger>
          {!isOllamaRunning && (
            <TooltipContent>
              <p>Ollama must be running to use local models</p>
            </TooltipContent>
          )}
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}
