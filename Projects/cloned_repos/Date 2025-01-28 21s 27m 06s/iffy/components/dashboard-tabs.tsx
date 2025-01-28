"use client";

import Image, { StaticImageData } from "next/image";
import { useState } from "react";

interface DashboardTabsProps {
  moderationsImage: StaticImageData;
  usersImage: StaticImageData;
  rulesImage: StaticImageData;
}

export function DashboardTabs({ moderationsImage, usersImage, rulesImage }: DashboardTabsProps) {
  const [activeTab, setActiveTab] = useState<"moderations" | "users" | "rules">("moderations");

  return (
    <div className="space-y-4">
      <div className="overflow-hidden rounded-lg border border-gray-200 shadow-lg">
        <div className="relative">
          <Image
            src={moderationsImage}
            alt="Moderations dashboard"
            className={`transition-opacity duration-200 ${activeTab === "moderations" ? "opacity-100" : "opacity-0"}`}
          />
          <Image
            src={usersImage}
            alt="Users dashboard"
            className={`absolute inset-0 transition-opacity duration-200 ${activeTab === "users" ? "opacity-100" : "opacity-0"}`}
          />
          <Image
            src={rulesImage}
            alt="Rules dashboard"
            className={`absolute inset-0 transition-opacity duration-200 ${activeTab === "rules" ? "opacity-100" : "opacity-0"}`}
          />
        </div>
      </div>
      <div className="flex w-full justify-center gap-12">
        <button
          onClick={() => setActiveTab("moderations")}
          className={`text-lg ${
            activeTab === "moderations"
              ? "text-black underline decoration-gray-300 underline-offset-8"
              : "text-gray-400 hover:text-gray-600"
          }`}
        >
          Moderations
        </button>
        <button
          onClick={() => setActiveTab("users")}
          className={`text-lg ${
            activeTab === "users"
              ? "text-black underline decoration-gray-300 underline-offset-8"
              : "text-gray-400 hover:text-gray-600"
          }`}
        >
          Users
        </button>
        <button
          onClick={() => setActiveTab("rules")}
          className={`text-lg ${
            activeTab === "rules"
              ? "text-black underline decoration-gray-300 underline-offset-8"
              : "text-gray-400 hover:text-gray-600"
          }`}
        >
          Rules
        </button>
      </div>
    </div>
  );
}
