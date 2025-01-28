"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";
import {
  SelectButton,
  SelectButtonContent,
  SelectButtonMenuTrigger,
  SelectButtonMenuContent,
  SelectButtonMenuItem,
  SelectButtonMenuSeparator,
} from "@/components/select-button";

type CloseAction = "default" | "completed" | "not-planned";

interface CloseIssueButtonProps {
  // onClose: (action: CloseAction) => void;
}

export default function AppealActionButton() {
  return (
    <SelectButton defaultSelectedItem="Reply">
      <SelectButtonContent className="group-hover:bg-destructive-hover" />
      <SelectButtonMenuTrigger className="border-destructive-hover group-hover:bg-destructive-hover focus:ring-offset-0">
        <ChevronDown className="h-4 w-4" />
      </SelectButtonMenuTrigger>
      <SelectButtonMenuContent className="w-56">
        <SelectButtonMenuItem value="Close as completed">Reply and reject</SelectButtonMenuItem>
        <SelectButtonMenuItem value="Close as not planned">Reply and confirm</SelectButtonMenuItem>
        <SelectButtonMenuSeparator />
        <SelectButtonMenuItem value="Close issue">Reply</SelectButtonMenuItem>
      </SelectButtonMenuContent>
    </SelectButton>
  );
}
