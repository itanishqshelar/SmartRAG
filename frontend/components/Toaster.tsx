"use client";

import { CircleAlert, CircleCheck, X } from "lucide-react";
import { useToasts } from "@/lib/toast";

export function Toaster() {
  const { toasts, dismiss } = useToasts();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          className="pointer-events-auto flex animate-slide-up items-start gap-3 rounded-xl border border-border bg-panel p-3 pr-2 shadow-xl shadow-black/10"
        >
          {t.kind === "success" ? (
            <CircleCheck className="mt-0.5 size-4 shrink-0 text-success" />
          ) : (
            <CircleAlert className="mt-0.5 size-4 shrink-0 text-danger" />
          )}
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{t.title}</p>
            {t.description && <p className="mt-0.5 text-xs text-muted">{t.description}</p>}
          </div>
          <button onClick={() => dismiss(t.id)} className="cursor-pointer rounded p-1 text-subtle hover:text-fg" aria-label="Dismiss">
            <X className="size-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
