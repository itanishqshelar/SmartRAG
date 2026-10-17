"use client";

import { AudioLines, FileText, FileType2, Image as ImageIcon, NotebookText, type LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes } from "react";
import type { DocumentType } from "@/lib/api";
import { cn } from "@/lib/utils";

export function Logo({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "grid size-8 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-[#8b7dff] via-[#6d5dfc] to-[#3b82f6] shadow-[0_0_0_1px_rgba(255,255,255,0.12)_inset,0_6px_20px_-6px_rgba(109,93,252,0.6)]",
        className,
      )}
    >
      <svg viewBox="0 0 24 24" className="size-[55%] text-white" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round">
        <path d="M5 7.5h9M5 12h14M5 16.5h6" />
        <circle cx="18" cy="7.5" r="1.6" fill="currentColor" stroke="none" />
      </svg>
    </div>
  );
}

const TYPE_STYLES: Record<DocumentType, { icon: LucideIcon; className: string; label: string }> = {
  pdf: { icon: FileText, className: "bg-red-500/12 text-red-500 dark:text-red-400", label: "PDF" },
  docx: { icon: FileType2, className: "bg-blue-500/12 text-blue-600 dark:text-blue-400", label: "Word" },
  text: { icon: NotebookText, className: "bg-zinc-500/12 text-zinc-600 dark:text-zinc-300", label: "Text" },
  image: { icon: ImageIcon, className: "bg-emerald-500/12 text-emerald-600 dark:text-emerald-400", label: "Image" },
  audio: { icon: AudioLines, className: "bg-amber-500/12 text-amber-600 dark:text-amber-400", label: "Audio" },
};

export const typeLabel = (type: DocumentType) => TYPE_STYLES[type]?.label ?? "File";

export function FileIcon({ type, className }: { type: DocumentType; className?: string }) {
  const { icon: Icon, className: tone } = TYPE_STYLES[type] ?? TYPE_STYLES.text;
  return (
    <div className={cn("grid size-9 shrink-0 place-items-center rounded-lg", tone, className)}>
      <Icon className="size-[1.1rem]" strokeWidth={1.8} />
    </div>
  );
}

export function IconButton({
  className,
  label,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "grid size-8 shrink-0 cursor-pointer place-items-center rounded-lg text-muted transition-colors hover:bg-hover hover:text-fg disabled:pointer-events-none disabled:opacity-40",
        className,
      )}
      {...props}
    />
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={cn("size-4 animate-spin", className)} fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <span key={i} className="size-1.5 animate-pulse-dot rounded-full bg-muted" style={{ animationDelay: `${i * 0.16}s` }} />
      ))}
    </span>
  );
}
