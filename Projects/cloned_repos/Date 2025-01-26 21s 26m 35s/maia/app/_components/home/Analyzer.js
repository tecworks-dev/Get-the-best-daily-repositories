"use client";

import Image from "next/image";
import { useState } from "react";
import dynamic from "next/dynamic";
import LoadingLottie from "../../../data/lotties/Loading.json";
import { toast } from "sonner";
import { Tab, Tabs } from "@nextui-org/react";
import { Chat } from "../ui/Chat.js";
const Lottie = dynamic(() => import("react-lottie"), { ssr: false });

const {
  PlaceholdersAndVanishInput,
} = require("../ui/PlaceholdersAndVanishInput");

const cursorClasses = {
  Cute: "group-data-[selected=true]:bg-[#00FFF6]",
  Evil: "group-data-[selected=true]:bg-[#FF4000]",
  Wise: "group-data-[selected=true]:bg-[#29FF42]",
};

const Analyzer = () => {
  const [value, setValue] = useState("");
  const [data, setData] = useState("");
  const [selectedPersonality, setPersonality] = useState("Cute");
  const [isLoading, setIsLoading] = useState(false);
  const placeholders = ["Enter token Contact address"];

  const personalities = ["Cute", "Evil", "Wise"];
  const colors = ["#00FFF6", "#FF4000", "#29FF42"];
  const bgColors = ["#11141D", "#1D1111", "#111D1D"];
  const images = [
    "/assets/hero.png",
    "/assets/hero-evil.jpeg",
    "/assets/hero-wise.jpeg",
  ];

  const handleChange = (e) => {
    setValue(e.target.value);
  };

  const onSubmit = async (e) => {
    setIsLoading(true);
    e.preventDefault();
    try {
      const response = await fetch("/api/token", {
        method: "POST",
        body: JSON.stringify({
          address: value,
          personality: selectedPersonality.toLocaleLowerCase(),
        }),
      });

      const newData = await response.json();

      setData(newData.data.result);
    } catch (error) {
      toast.error("Something went wrong!");
    } finally {
      setIsLoading(false);
    }
  };

  const lottieOptions = {
    loop: true,
    autoplay: true,
    animationData: LoadingLottie,
  };

  return (
    <section
      id="analyzer"
      className="min-h-screen flex flex-col justify-center items-center px-6  max-w-[1440px] mx-auto w-full"
    >
      <h2
        data-aos="fade-right"
        className="text-center text-3xl sm:text-5xl text-white font-oSemibold uppercase mb-[27px]"
      >
        MAIA Analyzer
      </h2>

      <Tabs
        data-aos="fade-out"
        fullWidth
        className="mb-3 max-w-4xl"
        radius="md"
        classNames={{
          cursor: `w-full ${cursorClasses[selectedPersonality]}`,
        }}
        onSelectionChange={(index) => {
          setPersonality(personalities[index]);
          setData("");
        }}
      >
        {personalities.map((personality, index) => (
          <Tab
            key={index}
            className="h-10"
            title={
              <span
                className="font-sBold text-lg"
                style={{
                  color: personality === selectedPersonality ? "#000" : "#fff",
                }}
              >
                {personality}
              </span>
            }
          />
        ))}
      </Tabs>

      <div
        data-aos="fade-up"
        className="h-[55vh] max-w-4xl w-full  flex items-center justify-center border-2 rounded-2xl mb-[27px]"
        style={{
          borderColor: colors[personalities.indexOf(selectedPersonality)],
          backgroundColor: bgColors[personalities.indexOf(selectedPersonality)],
        }}
      >
        {isLoading ? (
          <Lottie options={lottieOptions} />
        ) : data ? (
          <div className="h-full p-8 hide-scroll overflow-y-auto overflow-x-hidden w-full">
            <Chat result={data} />
          </div>
        ) : (
          <Image
            src={images[personalities.indexOf(selectedPersonality)]}
            alt="analyzer"
            width={700}
            height={700}
            className="size-[205px] rounded-full object-cover"
          />
        )}
      </div>
      <div data-aos="fade-up" className="max-w-4xl w-full">
        <PlaceholdersAndVanishInput
          className="w-full"
          placeholders={placeholders}
          onChange={handleChange}
          onSubmit={onSubmit}
        />
      </div>
    </section>
  );
};

export default Analyzer;
