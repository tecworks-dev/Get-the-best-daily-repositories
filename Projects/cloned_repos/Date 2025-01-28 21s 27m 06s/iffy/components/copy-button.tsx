"use client";

import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";

export function CopyButton({ text, name }: { text: string; name: string }) {
  const { toast } = useToast();

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: `${name} copied to clipboard`,
      duration: 2000,
    });
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleCopy}>
            <Copy className="h-4 w-4 dark:text-stone-50" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Copy {name}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
