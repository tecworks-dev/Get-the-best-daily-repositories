import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { Entries } from "type-fest";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// https://stackoverflow.com/a/74405465
export function entries<T extends object>(obj: T) {
  return Object.entries(obj) as Entries<T>;
}
