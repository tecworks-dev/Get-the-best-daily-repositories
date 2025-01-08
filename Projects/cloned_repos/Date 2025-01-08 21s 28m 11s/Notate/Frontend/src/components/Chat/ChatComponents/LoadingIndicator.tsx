import { Loader2 } from "lucide-react";

export function LoadingIndicator() {
  return (
    <div className="flex justify-center my-4">
      <div className="flex items-center bg-secondary/50 text-secondary-foreground rounded-full px-4 py-2 shadow-md">
        <Loader2 className="w-4 h-4 animate-spin mr-2" />
        <span className="text-sm">AI is processing...</span>
      </div>
    </div>
  );
}
