import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MessageSquare, Cpu, Settings2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import LLMSettings from "./SettingsComponents/LLMSettings";
import { DevIntegration } from "./SettingsComponents/DevIntegration";
import ChatSettings from "./SettingsComponents/ChatSettings";

export function SettingsModal() {
  return (
    <Tabs defaultValue="chat" className="h-full flex flex-col">
      <TabsList className="w-full shrink-0 rounded-none border-b bg-transparent p-0">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <TabsTrigger
                value="chat"
                className="relative h-9 rounded-none border-b-2 border-b-transparent bg-transparent px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground data-[state=active]:border-b-primary data-[state=active]:text-foreground"
              >
                <MessageSquare className="h-4 w-4" />
                <p className="hidden md:block pl-2">Chat Settings</p>
              </TabsTrigger>
            </TooltipTrigger>
            <TooltipContent>
              <p>Chat Settings</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <TabsTrigger
                value="llm"
                className="relative h-9 rounded-none border-b-2 border-b-transparent bg-transparent px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground data-[state=active]:border-b-primary data-[state=active]:text-foreground"
              >
                <Settings2 className="h-4 w-4" />
                <p className="hidden md:block pl-2">LLM Integration</p>
              </TabsTrigger>
            </TooltipTrigger>
            <TooltipContent>
              <p>LLM Integration</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <TabsTrigger
                value="system"
                className="relative h-9 rounded-none border-b-2 border-b-transparent bg-transparent px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground data-[state=active]:border-b-primary data-[state=active]:text-foreground"
              >
                <Cpu className="h-4 w-4" />
                <p className="hidden md:block pl-2">Dev Integration</p>
              </TabsTrigger>
            </TooltipTrigger>
            <TooltipContent>
              <p>Developer Integration</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </TabsList>
      <div className="flex-1 overflow-hidden">
        <TabsContent
          value="chat"
          className="h-full m-0 overflow-auto border-none p-6 outline-none"
        >
          <ChatSettings />
        </TabsContent>
        <TabsContent
          value="llm"
          className="h-full m-0 overflow-auto border-none p-6 outline-none"
        >
          <LLMSettings />
        </TabsContent>
        <TabsContent
          value="system"
          className="h-full m-0 overflow-auto border-none p-6 outline-none"
        >
          <DevIntegration />
        </TabsContent>
      </div>
    </Tabs>
  );
}
