import { useState } from "react";
import { NAV_ITEMS } from "../Header.js";
import { Icons } from "../icon/icon.js";
import { OutlineButton } from "./OutlineButton";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "./Sheet";
import Link from "next/link.js";
import { BorderButton } from "./BorderButton.js";

export const Sidebar = () => {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <OutlineButton className={"px-0 w-[35px]"}>
          <Icons.menu />
        </OutlineButton>
      </SheetTrigger>
      <SheetContent className="flex flex-col justify-between">
        <div className="space-y-10">
          <SheetHeader>
            <SheetTitle>
              <Link href={"/"}>
                <h1 className="font-oSemibold text-primary text-[32px] text-left">
                  MAIA
                </h1>
              </Link>
            </SheetTitle>
          </SheetHeader>
          <div className="flex flex-col gap-4">
            {NAV_ITEMS.map((item, index) => (
              <Link
                key={index}
                href={item.href}
                onClick={() => setOpen(false)}
                className="text-foreground-50 hover:text-primary transition-all"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
        <SheetFooter>
          <SheetClose asChild>
            <BorderButton
              onClick={() => window.open(GITHUB_LINK)}
              className={
                "font-sMedium text-sm flex items-center gap-1.5 h-[35px] w-24"
              }
            >
              <Icons.github /> Github
            </BorderButton>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
};
