import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SyntaxHighlightedCode } from "@/components/Chat/ChatComponents/SyntaxHightlightedCode";
import { useState, useEffect } from "react";

export function StreamingMessage({ content }: { content: string }) {
  const [parsedContent, setParsedContent] = useState<(string | JSX.Element)[]>(
    []
  );

  useEffect(() => {
    const parts: (string | JSX.Element)[] = [];
    let codeBlock = "";
    let isInCodeBlock = false;
    let language = "";

    const lines = content.split("\n");

    for (const line of lines) {
      if (line.startsWith("```")) {
        if (isInCodeBlock) {
          parts.push(
            <SyntaxHighlightedCode
              key={parts.length}
              code={codeBlock.trim()}
              language={language}
            />
          );
          codeBlock = "";
          isInCodeBlock = false;
          language = "";
        } else {
          isInCodeBlock = true;
          language = line.slice(3).trim() || "text";
        }
      } else if (isInCodeBlock) {
        codeBlock += line + "\n";
      } else {
        parts.push(
          ...line.split("").map((char, index) => (
            <span key={`${parts.length}-${index}`} className="animate-fade-in">
              {char}
            </span>
          ))
        );
        parts.push(<br key={`${parts.length}-br`} />);
      }
    }

    if (isInCodeBlock) {
      parts.push(
        <SyntaxHighlightedCode
          key={parts.length}
          code={codeBlock.trim()}
          language={language}
        />
      );
    }

    setParsedContent(parts);
  }, [content]);

  return (
    <div className="flex justify-start animate-in fade-in duration-300 mx-2 my-2">
      <div className="flex flex-row items-end max-w-[80%]">
        <Avatar className="w-8 h-8 ring-2 ring-secondary">
          <AvatarFallback>AI</AvatarFallback>
          <AvatarImage src="/src/assets/avatars/ai-avatar.png" />
        </Avatar>
        <div className="mx-2 my-1 p-3 rounded-2xl bg-secondary text-secondary-foreground shadow-md rounded-bl-none">
          <div className="text-sm whitespace-pre-wrap break-words text-left">
            {parsedContent}
          </div>
        </div>
      </div>
    </div>
  );
}
