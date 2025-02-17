"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BsCalendar, BsGear, BsListTask } from "react-icons/bs";
import { cn } from "@/lib/utils";

interface AppNavProps {
  className?: string;
}

export function AppNav({ className }: AppNavProps) {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Calendar", icon: BsCalendar },
    { href: "/tasks", label: "Tasks", icon: BsListTask },
    { href: "/settings", label: "Settings", icon: BsGear },
  ];

  return (
    <nav
      className={cn(
        "h-16 bg-white border-b border-gray-200 flex-none z-10",
        className
      )}
    >
      <div className="h-full px-4">
        <div className="h-full flex items-center justify-center">
          <div className="flex items-center gap-8">
            {links.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={cn(
                    "inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md",
                    isActive
                      ? "bg-blue-50 text-blue-700"
                      : "text-gray-900 hover:bg-gray-50"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
