"use client";

import { Toaster as Sonner } from "sonner";

const Toaster = ({ ...props }) => {
  return (
    <Sonner
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-foreground-50/30 group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-foreground-100",
          actionButton: "group-[.toast]:bg-primary group-[.toast]:text-black",
          cancelButton:
            "group-[.toast]:bg-transparent group-[.toast]:border group-[.toast]:border-primary group-[.toast]:text-primary",
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
