import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
} from "@/components/ui/command";
import { Check, ChevronDown, Plus, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
} from "@/components/ui/select";
import { useUser } from "@/context/useUser";
import { useState, useEffect } from "react";
import { useView } from "@/context/useView";
import { useSysSettings } from "@/context/useSysSettings";
import { toast } from "@/hooks/use-toast";
import { useLibrary } from "@/context/useLibrary";

export default function ChatSettings() {
  const { setApiKeys, setPrompts, setConversations, setActiveUser } = useUser();
  const { setSelectedCollection, setFiles } = useLibrary();
  const { activeUser, apiKeys, prompts } = useUser();
  const [open, setOpen] = useState<boolean>(false);
  const [value, setValue] = useState<string>("");
  const [showNewPrompt, setShowNewPrompt] = useState<boolean>(false);
  const [newPrompt, setNewPrompt] = useState<string>("");
  const { setActiveView } = useView();
  const {
    settings,
    setSettings,
    setSettingsOpen,
    localModels,
    handleRunOllama,
  } = useSysSettings();

  const handleSettingChange = async (
    setting: string,
    value: string | number | undefined
  ) => {
    setSettings((prev) => ({ ...prev, [setting]: value }));
    if (setting === "prompt") {
      const promptId = typeof value === "string" ? parseInt(value) : value;
      const selectedPromptName =
        prompts.find((p) => p.id === promptId)?.name || "";
      setValue(selectedPromptName);
    }
    try {
      if (activeUser) {
        await window.electron.updateUserSettings(
          activeUser.id,
          setting,
          value?.toString() ?? ""
        );
      }
    } catch (error) {
      console.error("Error updating user settings:", error);
    }
  };

  const modelOptions = {
    openai: ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    anthropic: [
      "claude-3-5-sonnet-20241022",
      "claude-3-5-haiku-20241022",
      "claude-3-opus-20240229",
      "claude-3-sonnet-20240229",
      "claude-3-haiku-20240307",
      "claude-2.1",
      "claude-2.0",
    ],
    gemini: ["gemini-1.5-flash", "gemini-1.5-pro"],
    xai: ["grok-beta"],
    local: localModels.map((model) => model.name),
  };

  const handleAddPrompt = async () => {
    if (activeUser) {
      const newPromptObject = await window.electron.addUserPrompt(
        activeUser.id,
        newPrompt,
        newPrompt
      );

      await handleSettingChange("prompt", newPromptObject.id.toString());
      setPrompts((prev) => [
        ...prev,
        {
          id: newPromptObject.id,
          name: newPromptObject.name,
          prompt: newPromptObject.prompt,
          userId: activeUser.id,
        },
      ]);
    }
  };

  useEffect(() => {
    if (settings.prompt) {
      const promptId = parseInt(settings.prompt);
      const selectedPromptName =
        prompts.find((p) => p.id === promptId)?.name || "";
      setValue(selectedPromptName);
    }
  }, [settings.prompt, prompts]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold mb-4">Chat Settings</h2>
        <div className="space-y-6">
          <div className="grid grid-cols-4 items-start gap-4">
            <Label
              htmlFor="prompt"
              className="text-right text-sm font-medium pt-2"
            >
              Prompt
            </Label>
            <div className="col-span-3 space-y-4">
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                  <Button
                    id="select-42"
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between bg-background px-3 font-normal"
                  >
                    <span
                      className={cn(
                        "truncate",
                        !value && "text-muted-foreground"
                      )}
                    >
                      {value || "Default Prompt"}
                    </span>
                    <ChevronDown size={16} className="opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-0" align="start">
                  <Command>
                    <CommandInput placeholder="Search prompts..." />
                    <CommandList>
                      <CommandEmpty>No prompt found.</CommandEmpty>
                      <CommandGroup>
                        <CommandItem
                          onSelect={() => {
                            setShowNewPrompt(true);
                            setOpen(false);
                            setValue("Adding New Prompt");
                          }}
                          className="flex items-center"
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Add New Prompt
                        </CommandItem>
                      </CommandGroup>
                      <CommandSeparator />
                      <CommandGroup>
                        {prompts.map((prompt) => (
                          <CommandItem
                            key={prompt.id}
                            value={prompt.name}
                            onSelect={(currentValue) => {
                              setValue(currentValue);
                              setOpen(false);
                              handleSettingChange(
                                "prompt",
                                prompt.id.toString()
                              );
                              toast({
                                title: "Prompt set",
                                description: `Prompt set to ${currentValue}`,
                              });
                            }}
                          >
                            {prompt.name.slice(0, 50)}
                            {value === prompt.name && (
                              <Check className="ml-auto h-4 w-4" />
                            )}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>

              {showNewPrompt && (
                <div className="flex gap-4">
                  <Textarea
                    id="newPrompt"
                    placeholder="Enter new prompt"
                    value={newPrompt}
                    onChange={(e) => setNewPrompt(e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    onClick={() => {
                      setShowNewPrompt(false);
                      handleAddPrompt();
                      setNewPrompt("");
                      toast({
                        title: "Prompt added",
                        description: `Prompt added to ${value}`,
                      });
                    }}
                  >
                    Add
                  </Button>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label
              htmlFor="temperature"
              className="text-right text-sm font-medium"
            >
              Temperature
            </Label>
            <div className="col-span-3 flex items-center gap-4">
              <Slider
                id="temperature"
                min={0}
                max={1}
                step={0.1}
                value={[settings.temperature ?? 0.7]}
                onValueChange={(value) => {
                  handleSettingChange("temperature", value[0]);
                }}
                className="flex-grow"
              />
              <span className="w-12 text-right text-sm tabular-nums">
                {settings.temperature?.toFixed(1) ?? "0.7"}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model" className="text-right text-sm font-medium">
              Model
            </Label>
            <Select
              value={settings.model}
              onValueChange={(value) => {
                const provider = Object.keys(modelOptions).find((key) =>
                  modelOptions[key as keyof typeof modelOptions].includes(value)
                ) as Provider;

                handleSettingChange("model", value);
                handleSettingChange("provider", provider);

                if (provider === "local" && activeUser) {
                  handleRunOllama(value, activeUser);
                  toast({
                    title: "Ollama model loading",
                    description: `Loading ${value}...`,
                  });
                } else {
                  toast({
                    title: "Model set",
                    description: `Model set to ${value}`,
                  });
                }
              }}
            >
              <SelectTrigger className="col-span-3">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {apiKeys.map((apiKey) => (
                  <SelectGroup key={apiKey.provider}>
                    <SelectLabel className="font-semibold">
                      {apiKey.provider.toUpperCase()}
                    </SelectLabel>
                    {modelOptions[
                      apiKey.provider as keyof typeof modelOptions
                    ]?.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                ))}
                <SelectGroup>
                  <SelectLabel className="font-semibold">LOCAL</SelectLabel>
                  {modelOptions.local.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <div className="flex flex-col space-y-4 pt-6 border-t">
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={() => setSettingsOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              setSettingsOpen(false);
              if (activeUser) {
                window.electron.updateUserSettings(
                  activeUser.id,
                  "vectorstore",
                  settings.vectorstore ?? ""
                );
                window.electron.updateUserSettings(
                  activeUser.id,
                  "temperature",
                  settings.temperature?.toString() ?? "0.7"
                );
                window.electron.updateUserSettings(
                  activeUser.id,
                  "model",
                  settings.model ?? ""
                );
              }
              toast({
                title: "Settings saved",
                description: `Settings saved`,
              });
            }}
          >
            Save Changes
          </Button>
        </div>

        <div className="flex">
          <Button
            variant="outline"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={() => {
              setActiveUser(null);
              setSelectedCollection(null);
              setApiKeys([]);
              setPrompts([]);
              setFiles([]);
              setConversations([]);
              setActiveView("SelectAccount");
              setSettingsOpen(false);
              toast({
                title: "Logged out",
                description: `Logged out of all accounts`,
              });
            }}
          >
            <LogOut size={14} className="mr-2" /> Logout
          </Button>
        </div>
      </div>
    </div>
  );
}
