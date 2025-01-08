import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Settings, Sparkles } from "lucide-react";
import { useSysSettings } from "@/context/useSysSettings";
import { SettingsModal } from "@/components/SettingsModal/SettingsModal";
import WindowControls from "./MainWindowControl";

export default function SettingsDialog() {
  const {
    settingsOpen,
    setSettingsOpen,
    platform,
    totalVRAM,
    isOllamaRunning,
    isMaximized,
    setIsMaximized,
  } = useSysSettings();
  const renderWindowControls = WindowControls({
    isMaximized,
    setIsMaximized,
    platform,
  });
  return (
    <div className="flex items-center justify-end">
      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogTrigger asChild className="clickable-header-section">
          <Button type="button" size="icon" variant="ghost">
            <Settings className="h-5 w-5" />
            <span className="sr-only">Chat Settings</span>
          </Button>
        </DialogTrigger>
        <DialogContent className="max-h-[100vh] mt-4 overflow-y-auto p-6">
          <DialogHeader className="sm:pb-4 pb-2">
            <DialogTitle className="text-xl font-semibold">
              Settings
            </DialogTitle>
            {totalVRAM > 8 && !isOllamaRunning && (
              <div className="animate-sparkle">
                <Sparkles className="h-4 w-4 text-primary" />
                Please Download and Run Ollama to use Notate in Local Mode
              </div>
            )}
            <DialogDescription className="text-muted-foreground">
              Configure your application preferences and settings
            </DialogDescription>
          </DialogHeader>
          <div className="overflow-y-auto overflow-x-hidden pr-2">
            <SettingsModal />
          </div>
        </DialogContent>
      </Dialog>
      {platform !== "darwin" && renderWindowControls}
    </div>
  );
}
