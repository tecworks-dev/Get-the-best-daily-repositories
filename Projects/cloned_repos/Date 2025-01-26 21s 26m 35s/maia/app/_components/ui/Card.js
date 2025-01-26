"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";

export function Card({ title, description, children }) {
  return (
    <CardX title={title} description={description}>
      {children}
    </CardX>
  );
}

const CardX = ({ title, description, children }) => {
  const [hovered, setHovered] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="bg-background border border-foreground/[0.2] group/canvas-card flex items-center justify-center max-w-sm w-full mx-auto p-4 relative h-[30rem]"
    >
      <Icon className="absolute h-6 w-6 -top-3 -left-3 text-foreground" />
      <Icon className="absolute h-6 w-6 -bottom-3 -left-3 text-foreground" />
      <Icon className="absolute h-6 w-6 -top-3 -right-3 text-foreground" />
      <Icon className="absolute h-6 w-6 -bottom-3 -right-3 text-foreground" />
      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="h-full w-full absolute inset-0"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
      <div className="relative z-20">
        <div className="absolute top-0 bottom-0 text-center group-hover/canvas-card:-translate-y-4 group-hover/canvas-card:opacity-0 transition duration-200 w-full  mx-auto flex items-center justify-center">
          {title}
        </div>
        <h2 className="text-center text-xl opacity-0 group-hover/canvas-card:opacity-100 relative z-10 text-foreground mt-4  font-bold group-hover/canvas-card:text-foreground group-hover/canvas-card:-translate-y-2 transition duration-200">
          {description}
        </h2>
      </div>
    </div>
  );
};

export const Icon = ({ className, ...rest }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth="1.5"
      stroke="currentColor"
      className={className}
      {...rest}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
    </svg>
  );
};
