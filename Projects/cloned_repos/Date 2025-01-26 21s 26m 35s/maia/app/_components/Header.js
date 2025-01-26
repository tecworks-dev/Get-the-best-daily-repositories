"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";
import { BorderButton } from "./ui/BorderButton";
import { Icons } from "./icon/icon";

import { GITHUB_LINK } from "@/config.js";
import { Sidebar } from "./ui/Sidebar.js";

export default function Header() {
  const [isScroll, setIsScroll] = useState(false);

  const handleIsScroll = () => {
    if (window !== undefined) {
      let windowHeight = window.scrollY;
      if (windowHeight > 1) {
        setIsScroll(true);
      } else {
        setIsScroll(false);
      }
    }
  };

  useEffect(() => {
    window.addEventListener("scroll", handleIsScroll);

    return () => {
      window.removeEventListener("scroll", handleIsScroll);
    };
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 w-full z-50 flex items-center justify-between duration-300 max-md:px-10 ${
        isScroll
          ? "bg-background/50 backdrop-blur-sm px-16 py-4"
          : "bg-background/0 p-16 max-md:py-8"
      } `}
    >
      <Link href={"/"}>
        <h1 className="font-oSemibold text-primary text-[32px]">MAIA</h1>
      </Link>
      <nav className="flex items-center gap-6 max-md:hidden">
        {NAV_ITEMS.map((item, index) => (
          <Link
            key={index}
            href={item.href}
            className="text-foreground-50 hover:text-primary transition-all text-sm"
          >
            {item.label}
          </Link>
        ))}
        <BorderButton
          onClick={() => window.open(GITHUB_LINK)}
          className={
            "font-sMedium text-sm flex items-center gap-1.5 h-[35px] w-24"
          }
        >
          <Icons.github /> Github
        </BorderButton>
      </nav>
      <div className="md:hidden">
        <Sidebar />
      </div>
    </header>
  );
}

export const NAV_ITEMS = [
  { label: "Analyzer", href: "#analyzer" },
  { label: "Features", href: "#features" },
];
