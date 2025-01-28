"use client";

import Image from "next/image";
import { useState } from "react";
import iffy from "./iffy.png";

export function IffyImage() {
  const [clicked, setClicked] = useState(false);

  return (
    <div className="relative">
      <Image
        src={iffy}
        alt="Iffy"
        className={`w-full cursor-pointer transition-all duration-300 will-change-transform ${clicked ? "" : "blur-xl hover:blur-lg"}`}
        onClick={() => setClicked(true)}
      />
      {!clicked && <div className="mt-2 uppercase text-gray-800">Click to moderate</div>}
      {clicked && <div className="mt-2 uppercase text-green-500">Compliant</div>}
    </div>
  );
}
