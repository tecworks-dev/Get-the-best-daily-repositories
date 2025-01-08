import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SyntaxHighlightedCode } from "@/components/Chat/ChatComponents/SyntaxHightlightedCode";
import { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  NotebookPenIcon,
  ExternalLink,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import { getYouTubeLink, formatTimestamp, getFileName } from "@/lib/utils";

export function ChatMessage({
  message,
  formatDate,
}: {
  message: Message;
  formatDate: (date: Date) => string;
}) {
  const isUser = message?.role === "user";
  const isRetrieval = message?.isRetrieval;
  const [isDataContentExpanded, setIsDataContentExpanded] = useState(false);

  const renderDataContent = (content: string) => {
    const dataContent: DataContent = JSON.parse(content);
    const topk = dataContent.top_k;

    return (
      <div className="flex flex-col gap-4 p-4 rounded-lg bg-secondary">
        <div className="text-sm font-medium">Top {topk} Results</div>
        {dataContent.results.map((result, index) => (
          <div
            key={index}
            className="flex flex-col gap-2 p-3 rounded border border-border"
          >
            <div className="font-medium text-foreground">
              Result {index + 1}
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex flex-row items-center gap-2 text-sm text-muted-foreground bg-background/40 rounded-[6px] border-2 border-green-700 p-2 overflow-hidden">
                <span className="font-medium">Source:</span>
                <span className="w-full text-green-700 truncate block flex flex-row items-center gap-2 justify-between">
                  {result.metadata.chunk_start ? (
                    <a
                      href={getYouTubeLink(
                        result.metadata.source,
                        result.metadata.chunk_start
                      )}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline truncate"
                    >
                      {result.metadata.title ||
                        getFileName(result.metadata.source)}
                    </a>
                  ) : result.metadata.source.startsWith("http") ? (
                    <a
                      href={result.metadata.source}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline truncate"
                    >
                      {result.metadata.title ||
                        getFileName(result.metadata.source)}
                    </a>
                  ) : (
                    <span className="truncate">
                      {result.metadata.title ||
                        getFileName(result.metadata.source)}
                    </span>
                  )}
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      const source = result.metadata.source;
                      if (
                        source.includes("youtube.com") ||
                        source.includes("youtu.be")
                      ) {
                        window.open(
                          getYouTubeLink(source, result.metadata.chunk_start),
                          "_blank"
                        );
                      } else if (source.startsWith("http")) {
                        window.open(source, "_blank");
                      } else {
                        window.electron.openCollectionFolder(source);
                      }
                    }}
                  >
                    <ExternalLink className="w-2 h-2" />
                  </Button>
                </span>
              </div>
              {result.metadata.chunk_start && result.metadata.chunk_end && (
                <div className="text-xs text-muted-foreground">
                  Timestamp: {formatTimestamp(result.metadata.chunk_start)} -{" "}
                  {formatTimestamp(result.metadata.chunk_end)}
                </div>
              )}
            </div>
            <div className="text-sm text-foreground whitespace-pre-wrap">
              {result.content}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderContent = (content: string) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push(content.slice(lastIndex, match.index));
      }

      const language = match[1] || "text";
      const code = match[2].trim();
      parts.push(
        <div key={match.index} className="max-w-full overflow-hidden">
          <SyntaxHighlightedCode code={code} language={language} />
        </div>
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      parts.push(content.slice(lastIndex));
    }

    return parts;
  };

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.value = message?.content || "";
    }
  }, [message?.content]);

  return (
    <div
      className={`flex ${
        isUser ? "justify-end" : "justify-start"
      } animate-in fade-in duration-300 mx-2 my-2`}
    >
      <div
        className={`flex ${
          isUser ? "flex-row-reverse" : "flex-row"
        } items-end max-w-[80%] group`}
      >
        <Avatar
          className={`w-8 h-8 ${
            isUser
              ? "ring-2 ring-primary"
              : isRetrieval
              ? "ring-2 ring-emerald-500"
              : "ring-2 ring-secondary"
          }`}
        >
          <AvatarFallback>
            {isUser ? "U" : isRetrieval ? "D" : "AI"}
          </AvatarFallback>
          <AvatarImage
            src={
              isUser
                ? "/src/assets/avatars/user-avatar.svg"
                : isRetrieval
                ? "/src/assets/avatars/database-avatar.svg"
                : "/src/assets/avatars/ai-avatar.png"
            }
          />
        </Avatar>
        <div
          className={`mx-2 my-1 p-3 rounded-2xl whitespace-pre-wrap break-words ${
            isUser
              ? "bg-primary text-primary-foreground rounded-br-none"
              : isRetrieval
              ? "bg-emerald-100 text-emerald-900 rounded-bl-none border-2 border-emerald-500"
              : "bg-secondary text-secondary-foreground rounded-bl-none"
          } shadow-md transition-all duration-300 group-hover:shadow-lg max-w-[calc(100% - 3rem)]`}
        >
          {message.data_content && (
            <div
              className="flex items-center justify-between mb-2 text-emerald-700 cursor-pointer"
              onClick={() => setIsDataContentExpanded(!isDataContentExpanded)}
            >
              <div className="flex items-center">
                <NotebookPenIcon className="mr-2" />
                <span className="font-semibold">Notations</span>
              </div>
              {isDataContentExpanded ? <ChevronUp /> : <ChevronDown />}
            </div>
          )}
          {message.data_content && isDataContentExpanded && (
            <div className="bg-white bg-opacity-50 p-2 rounded mb-2 text-sm">
              {renderDataContent(message.data_content)}
            </div>
          )}
          {!isRetrieval && (
            <div
              className={`text-sm whitespace-pre-wrap break-words text-left overflow-hidden ${
                isRetrieval ? "bg-white bg-opacity-50 p-2 rounded" : ""
              }`}
              data-testid={`message-content-${message.role}`}
            >
              {renderContent(message?.content || "")}
              <div className="sr-only">{message?.content}</div>
            </div>
          )}
          <span
            className={`text-xs mt-1 block group-hover:opacity-100 transition-opacity text-right ${
              isUser
                ? "text-primary-foreground"
                : isRetrieval
                ? "text-emerald-700"
                : "text-secondary-foreground"
            }`}
          >
            {message?.timestamp ? formatDate(message?.timestamp) : ""}
          </span>
          <textarea
            ref={textareaRef}
            className="sr-only"
            readOnly
            aria-hidden="true"
          />
        </div>
      </div>
    </div>
  );
}
