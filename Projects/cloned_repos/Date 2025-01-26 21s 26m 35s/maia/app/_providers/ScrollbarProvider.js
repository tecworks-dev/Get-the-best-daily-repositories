"use client";

import "overlayscrollbars/styles/overlayscrollbars.css";
import { useEffect } from "react";
import { OverlayScrollbars } from "overlayscrollbars";

export function ScrollbarProvider(props) {
  function init() {
    const elm = document.querySelector("[data-overlayscrollbars-initialize]");
    if (elm) {
      OverlayScrollbars(elm, {
        scrollbars: {
          theme: "os-theme-custom",
        },
      });
    }
  }

  useEffect(() => {
    init();
  }, []);

  return props.children;
}
