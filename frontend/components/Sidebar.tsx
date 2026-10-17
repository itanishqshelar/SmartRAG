"use client";

import { MessageSquare, Moon, PanelLeftClose, Plus, Sun, Trash2 } from "lucide-react";
import { useMemo } from "react";
import { useLibrary } from "@/lib/documents";
import { useChatStore, type Conversation } from "@/lib/store";
import { cn } from "@/lib/utils";
import { IconButton, Logo } from "./ui";

function groupConversations(conversations: Conversation[]) {
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const groups: { label: string; items: Conversation[] }[] = [
    { label: "Today", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Older", items: [] },
  ];
  for (const c of [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (c.updatedAt >= startOfToday) groups[0].items.push(c);
    else if (c.updatedAt >= startOfToday - 7 * 86400_000) groups[1].items.push(c);
    else groups[2].items.push(c);
  }
  return groups.filter((g) => g.items.length);
}

export function Sidebar({ onClose }: { onClose?: () => void }) {
  const { conversations, activeId, newConversation, selectConversation, deleteConversation, theme, toggleTheme } =
    useChatStore();
  const groups = useMemo(() => groupConversations(conversations), [conversations]);

  return (
    <aside className="flex h-full w-full flex-col bg-panel">
      <div className="flex items-center gap-2.5 px-4 pb-3 pt-4">
        <Logo />
        <div className="min-w-0 flex-1">
          <p className="text-[15px] font-semibold tracking-tight">SmartRAG</p>
          <p className="text-[11px] text-muted">Private document intelligence</p>
        </div>
        {onClose && <IconButton label="Close sidebar" onClick={onClose}><PanelLeftClose className="size-4" /></IconButton>}
      </div>

      <div className="px-3 pb-2">
        <button
          onClick={() => {
            newConversation();
            onClose?.();
          }}
          className="flex w-full cursor-pointer items-center gap-2 rounded-xl border border-border bg-bg px-3 py-2.5 text-sm font-medium transition-colors hover:border-border-strong hover:bg-hover"
        >
          <Plus className="size-4" />
          New chat
          <kbd className="ml-auto rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-subtle">Ctrl K</kbd>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 pb-3">
        {groups.length === 0 && (
          <div className="mt-8 flex flex-col items-center px-4 text-center text-xs text-subtle">
            <MessageSquare className="mb-2 size-5" strokeWidth={1.5} />
            Your conversations will appear here
          </div>
        )}
        {groups.map((group) => (
          <div key={group.label} className="mt-3">
            <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wider text-subtle">{group.label}</p>
            {group.items.map((c) => (
              <div
                key={c.id}
                className={cn(
                  "group relative flex items-center rounded-lg transition-colors",
                  c.id === activeId ? "bg-elevated text-fg" : "text-muted hover:bg-hover hover:text-fg",
                )}
              >
                <button
                  onClick={() => {
                    selectConversation(c.id);
                    onClose?.();
                  }}
                  className="min-w-0 flex-1 cursor-pointer truncate py-2 pl-2.5 pr-8 text-left text-[13.5px]"
                >
                  {c.title}
                </button>
                <button
                  onClick={() => deleteConversation(c.id)}
                  aria-label="Delete conversation"
                  className="absolute right-1 hidden cursor-pointer rounded-md p-1.5 text-subtle hover:text-danger group-hover:block"
                >
                  <Trash2 className="size-3.5" />
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>

      <div className="flex items-center gap-1 border-t border-border p-3">
        <StatusIndicator />
        <IconButton label={theme === "dark" ? "Light mode" : "Dark mode"} onClick={toggleTheme}>
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </IconButton>
      </div>
    </aside>
  );
}

function StatusIndicator() {
  const { health, healthError } = useLibrary();

  let tone = "bg-amber-400";
  let label = "Connecting…";
  let hint = "";
  if (healthError) {
    tone = "bg-red-500";
    label = "API offline";
    hint = "Start the backend: uvicorn api:app --port 8000";
  } else if (health && !health.ollama) {
    tone = "bg-red-500";
    label = "Ollama offline";
    hint = "Start Ollama: ollama serve";
  } else if (health && !health.llm) {
    tone = "bg-amber-400";
    label = "Model missing";
    hint = `Run: ollama pull ${health.llm_model}`;
  } else if (health && !health.embeddings) {
    tone = "bg-amber-400";
    label = "Embeddings missing";
    hint = `Run: ollama pull ${health.embedding_model}`;
  } else if (health) {
    tone = "bg-emerald-500";
    label = health.llm_model;
    hint = `Embeddings: ${health.embedding_model} · ${health.chunks} chunks indexed`;
  }

  return (
    <div className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1.5" title={hint}>
      <span className="relative flex size-2 shrink-0">
        {tone === "bg-emerald-500" && <span className={cn("absolute inset-0 animate-ping rounded-full opacity-40", tone)} />}
        <span className={cn("relative size-2 rounded-full", tone)} />
      </span>
      <div className="min-w-0">
        <p className="truncate text-xs font-medium">{label}</p>
        {hint && tone !== "bg-emerald-500" && <p className="truncate font-mono text-[10.5px] text-muted">{hint}</p>}
        {tone === "bg-emerald-500" && <p className="text-[10.5px] text-muted">Running locally</p>}
      </div>
    </div>
  );
}
