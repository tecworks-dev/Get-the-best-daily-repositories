"use client";
import React from "react";
import { cn } from "@/app/_lib/utils/uiUtils.js";
import { MovingBorder } from "./MovingBorder.js";

export function BorderButton({
  borderRadius = "8px",
  children,
  containerClassName,
  borderClassName,
  duration,
  className,
  shadowAnimation = true,
  darken = false,
  movingAnimation = true,
  ...otherProps
}) {
  return (
    <button
      className="relative group w-fit transition-transform duration-300 active:scale-95"
      {...otherProps}
    >
      {shadowAnimation && (
        <span className="pointer-events-none absolute -inset-0 z-10 transform-gpu rounded-2xl bg-gradient-to-br from-primary to-secondary opacity-20 blur-lg transition-all duration-300 group-hover:opacity-90 group-active:opacity-40" />
      )}

      <div
        className={cn(
          "bg-transparent relative text-sm p-[1px] overflow-hidden z-20",
          containerClassName
        )}
        style={{
          borderRadius: borderRadius,
        }}
      >
        {movingAnimation && (
          <div
            className="absolute inset-0"
            style={{ borderRadius: `calc(${borderRadius} * 0.96)` }}
          >
            <MovingBorder duration={duration} rx="30%" ry="30%">
              <div
                className={cn(
                  "h-20 w-20 opacity-[0.8] bg-[radial-gradient(theme(colors.primary.DEFAULT)_30%,transparent_60%)]",
                  borderClassName
                )}
              />
            </MovingBorder>
          </div>
        )}

        <div
          className={cn(
            "relative bg-primary border border-background text-black backdrop-blur-xl flex items-center justify-center w-full h-full antialiased font-sMedium text-lg",
            `${
              darken
                ? "group-hover:bg-black group-hover:border-primary-600 group-hover:text-white transition-colors duration-300"
                : ""
            }`,
            className
          )}
          style={{
            borderRadius: `calc(${borderRadius} * 0.96)`,
          }}
        >
          {children}
        </div>
      </div>
    </button>
  );
}
