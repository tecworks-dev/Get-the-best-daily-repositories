import * as React from "react"

import { cn } from "~/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<
  HTMLInputElement,
  InputProps & { wrapperClassName?: string; endContent?: React.ReactNode }
>(({ className, type, wrapperClassName, endContent, ...props }, ref) => {
  return (
    <div
      className={cn(
        "inline-flex h-7 w-full items-center rounded-md border border-input bg-transparent px-2 py-1 text-sm transition-colors placeholder:text-muted-foreground focus-within:border-ring focus-within:outline-none focus-within:ring-1 focus-within:ring-ring disabled:cursor-not-allowed disabled:opacity-50 dark:border-neutral-700 dark:bg-neutral-800 dark:focus-within:border-ring",
        wrapperClassName,
      )}
    >
      <input
        type={type}
        className={cn("grow bg-transparent outline-none", className)}
        ref={ref}
        {...props}
      />

      {endContent}
    </div>
  )
})
Input.displayName = "Input"

export { Input }
