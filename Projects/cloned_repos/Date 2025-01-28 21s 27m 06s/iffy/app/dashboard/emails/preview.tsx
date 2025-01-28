"use client";

import { cn } from "@/lib/utils";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

const INITIAL_HEIGHT = 800;

export const Preview = ({ html, className, ...props }: { html: string } & React.HTMLAttributes<HTMLDivElement>) => {
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(INITIAL_HEIGHT + "px");

  const resize = () => {
    if (ref.current?.contentWindow) {
      const height = ref.current.contentWindow.document.body.scrollHeight;
      if (height > 0) {
        setHeight(height + "px");
      }
    }
  };

  useLayoutEffect(() => {
    resize();
  }, [html]);

  return (
    <iframe
      ref={ref}
      className={cn("w-full bg-white dark:bg-zinc-900", className)}
      srcDoc={html}
      onLoad={resize}
      style={{ height }}
      {...props}
    />
  );
};
