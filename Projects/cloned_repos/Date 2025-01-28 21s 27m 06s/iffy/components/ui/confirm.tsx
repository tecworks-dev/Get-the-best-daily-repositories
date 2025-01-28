"use client";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { ReactNode, createContext, useCallback, useContext, useState } from "react";

interface ConfirmOptions {
  title: string;
  description: string;
}

const ConfirmContext = createContext<(options: ConfirmOptions) => Promise<boolean>>(() => Promise.resolve(false));

export function useConfirm() {
  return useContext(ConfirmContext);
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [options, setOptions] = useState<ConfirmOptions>({ title: "", description: "" });
  const [resolve, setResolve] = useState<(value: boolean) => void>();

  const confirm = useCallback((options: ConfirmOptions) => {
    setOptions(options);
    setOpen(true);
    return new Promise<boolean>((resolve) => setResolve(() => resolve));
  }, []);

  const handleOpenChange = (open: boolean) => {
    setOpen(open);
    if (!open) {
      resolve?.(false);
    }
  };

  const handleConfirm = () => {
    setOpen(false);
    resolve?.(true);
  };

  const handleCancel = () => {
    setOpen(false);
    resolve?.(false);
  };

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <AlertDialog open={open} onOpenChange={handleOpenChange}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{options.title}</AlertDialogTitle>
            <AlertDialogDescription>{options.description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleCancel}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirm}>Continue</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ConfirmContext.Provider>
  );
}
