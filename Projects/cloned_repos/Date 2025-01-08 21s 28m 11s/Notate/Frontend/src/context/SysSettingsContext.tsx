import React, { createContext, useRef, useState } from "react";
import { toast } from "@/hooks/use-toast";

interface SysSettingsContextType {
  isOllamaRunning: boolean;
  setIsOllamaRunning: (isOllamaRunning: boolean) => void;
  systemSpecs: {
    cpu: string;
    vram: string;
    GPU_Manufacturer?: string;
  };
  setSystemSpecs: (systemSpecs: {
    cpu: string;
    vram: string;
    GPU_Manufacturer?: string;
  }) => void;
  settingsOpen: boolean;
  setSettingsOpen: React.Dispatch<React.SetStateAction<boolean>>;
  settings: UserSettings;
  setSettings: React.Dispatch<React.SetStateAction<UserSettings>>;
  platform: "win32" | "darwin" | "linux" | null;
  setPlatform: React.Dispatch<
    React.SetStateAction<"win32" | "darwin" | "linux" | null>
  >;
  sourceType: "local" | "external";
  setSourceType: React.Dispatch<React.SetStateAction<"local" | "external">>;
  users: User[];
  setUsers: React.Dispatch<React.SetStateAction<User[]>>;
  totalVRAM: number;
  localModels: OllamaModel[];
  setLocalModels: React.Dispatch<React.SetStateAction<OllamaModel[]>>;
  isRunningModel: boolean;
  setIsRunningModel: React.Dispatch<React.SetStateAction<boolean>>;
  isFFMPEGInstalled: boolean;
  setisFFMPEGInstalled: React.Dispatch<React.SetStateAction<boolean>>;
  localModalLoading: boolean;
  setLocalModalLoading: React.Dispatch<React.SetStateAction<boolean>>;
  progressRef: React.RefObject<HTMLDivElement>;
  progressLocalOutput: string[];
  setProgressLocalOutput: React.Dispatch<React.SetStateAction<string[]>>;
  handleRunOllama: (model: string, activeUser: User) => Promise<void>;
  isMaximized: boolean;
  setIsMaximized: React.Dispatch<React.SetStateAction<boolean>>;
  checkFFMPEG: () => Promise<void>;
  fetchLocalModels: () => Promise<void>;
  fetchSystemSpecs: () => Promise<void>;
  checkOllama: () => Promise<void>;
}

const SysSettingsContext = createContext<SysSettingsContextType | undefined>(
  undefined
);

const SysSettingsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [localModels, setLocalModels] = useState<OllamaModel[]>([]);
  const [isMaximized, setIsMaximized] = useState<boolean>(false);
  const [isOllamaRunning, setIsOllamaRunning] = useState<boolean>(false);
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [settings, setSettings] = useState<UserSettings>({});
  const [users, setUsers] = useState<User[]>([]);
  const progressRef = useRef<HTMLDivElement>(null);
  const [isRunningModel, setIsRunningModel] = useState(false);
  const [isFFMPEGInstalled, setisFFMPEGInstalled] = useState(false);
  const [localModalLoading, setLocalModalLoading] = useState(false);
  const [progressLocalOutput, setProgressLocalOutput] = useState<string[]>([]);
  const [sourceType, setSourceType] = useState<"local" | "external">(
    "external"
  );
  const [platform, setPlatform] = useState<"win32" | "darwin" | "linux" | null>(
    null
  );
  const [systemSpecs, setSystemSpecs] = useState<{
    cpu: string;
    vram: string;
    GPU_Manufacturer?: string;
  }>({
    cpu: "Unknown",
    vram: "Unknown",
    GPU_Manufacturer: "Unknown",
  });
  const totalVRAM = parseInt(systemSpecs.vram);

  const checkFFMPEG = async () => {
    try {
      const result = await window.electron.checkIfFFMPEGInstalled();
      if (result && typeof result.success === "boolean") {
        setisFFMPEGInstalled(result.success);
      } else {
        console.error("Invalid FFMPEG check result:", result);
        setisFFMPEGInstalled(false);
      }
    } catch (error) {
      console.error("Error checking FFMPEG:", error);
      setisFFMPEGInstalled(false);
    }
  };
  const fetchSystemSpecs = async () => {
    try {
      const { cpu, vram, GPU_Manufacturer } =
        await window.electron.systemSpecs();
      setSystemSpecs({ cpu, vram, GPU_Manufacturer });
    } catch (error) {
      console.error("Error fetching system specs:", error);
      setSystemSpecs({
        cpu: "Unknown",
        vram: "Unknown",
        GPU_Manufacturer: "Unknown",
      });
    }
  };

  const checkOllama = async () => {
    const { isOllamaRunning } = await window.electron.checkOllama();
    setIsOllamaRunning(isOllamaRunning);
    if (isOllamaRunning) {
      fetchLocalModels();
    }
  };

  const fetchLocalModels = async () => {
    try {
      const data = await window.electron.fetchOllamaModels();
      setLocalModels(
        Array.isArray(data.models)
          ? data.models.map((model: string | OllamaModel) => ({
              name: typeof model === "string" ? model : model.name || "",
              modified_at:
                typeof model === "string" ? "" : model.modified_at || "",
              size: typeof model === "string" ? 0 : model.size || 0,
              digest: typeof model === "string" ? "" : model.digest || "",
            }))
          : []
      );
    } catch (error) {
      console.error("Error fetching local models:", error);
    }
  };

  const handleRunOllama = async (model: string, activeUser: User) => {
    if (!model) {
      toast({
        title: "Error",
        description: "Please select a model first",
        variant: "destructive",
      });
      return;
    }
    setLocalModalLoading(true);
    setProgressLocalOutput([]);

    try {
      const result = await window.electron.runOllama(model, activeUser);

      if (!result.success && result.error) {
        toast({
          title: "Error",
          description: result.error,
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Success",
        description: `Started Ollama with model: ${model}`,
      });

      setSettings((prev) => ({
        ...prev,
        provider: "local",
        model: model,
      }));
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to start Ollama",
        variant: "destructive",
      });
    } finally {
      setLocalModalLoading(false);
    }
  };

  return (
    <SysSettingsContext.Provider
      value={{
        isOllamaRunning,
        setIsOllamaRunning,
        systemSpecs,
        setSystemSpecs,
        totalVRAM,
        settingsOpen,
        setSettingsOpen,
        settings,
        setSettings,
        platform,
        setPlatform,
        sourceType,
        setSourceType,
        users,
        setUsers,
        localModels,
        setLocalModels,
        isRunningModel,
        setIsRunningModel,
        isFFMPEGInstalled,
        setisFFMPEGInstalled,
        localModalLoading,
        setLocalModalLoading,
        progressRef,
        progressLocalOutput,
        setProgressLocalOutput,
        handleRunOllama,
        isMaximized,
        setIsMaximized,
        checkFFMPEG,
        fetchLocalModels,
        fetchSystemSpecs,
        checkOllama,
      }}
    >
      {children}
    </SysSettingsContext.Provider>
  );
};

export { SysSettingsProvider, SysSettingsContext };
