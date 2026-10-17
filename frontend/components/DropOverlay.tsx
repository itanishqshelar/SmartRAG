"use client";

import { UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";
import { useLibrary } from "@/lib/documents";

/** Full-window drop target so files can be dropped anywhere in the app. */
export function DropOverlay() {
  const [active, setActive] = useState(false);
  const upload = useLibrary((s) => s.upload);

  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files");
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setActive(true);
    };
    const onLeave = () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) setActive(false);
    };
    const onOver = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      depth = 0;
      setActive(false);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length) upload(files);
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [upload]);

  if (!active) return null;
  return (
    <div className="pointer-events-none fixed inset-0 z-[70] grid animate-fade-in place-items-center bg-bg/70 p-6 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-3 rounded-3xl border-2 border-dashed border-accent bg-panel/80 px-16 py-12 text-center shadow-2xl">
        <div className="grid size-14 place-items-center rounded-full bg-accent-soft text-accent">
          <UploadCloud className="size-7" />
        </div>
        <p className="text-lg font-semibold">Drop to add to your library</p>
        <p className="text-sm text-muted">Documents, images and audio are processed locally</p>
      </div>
    </div>
  );
}
