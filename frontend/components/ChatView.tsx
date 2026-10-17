"use client";

import { ArrowDown, Filter, Library, Menu, X } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useChat } from "@/lib/chat";
import { useLibrary } from "@/lib/documents";
import { useChatStore, type Message as MessageType } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Composer } from "./Composer";
import { Message } from "./Message";
import { IconButton, Logo } from "./ui";

const NO_MESSAGES: MessageType[] = [];

const SUGGESTIONS = [
  { title: "Summarize", prompt: "Summarize the key points across my documents." },
  { title: "Find dates & deadlines", prompt: "List every date, deadline or time-sensitive item mentioned in my documents." },
  { title: "Key numbers", prompt: "What are the most important figures or numbers in my documents?" },
  { title: "Explain simply", prompt: "Explain the main topic of my documents as if I'm new to it." },
];

export function ChatView({
  libraryOpen,
  onToggleLibrary,
  onOpenSidebar,
}: {
  libraryOpen: boolean;
  onToggleLibrary: () => void;
  onOpenSidebar: () => void;
}) {
  const conversation = useChatStore((s) => s.conversations.find((c) => c.id === s.activeId));
  const scoped = useChatStore((s) => s.scopedDocumentIds);
  const setScoped = useChatStore((s) => s.setScopedDocuments);
  const documents = useLibrary((s) => s.documents);
  const messages = conversation?.messages ?? NO_MESSAGES;

  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);

  // Keep the view pinned to the bottom while streaming, unless the user scrolled up
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Switching conversations jumps to the latest message (the resulting scroll event resets showJump)
  useEffect(() => {
    pinned.current = true;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [conversation?.id]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    pinned.current = nearBottom;
    setShowJump(!nearBottom);
  };

  const jumpToBottom = () => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  };

  return (
    <main className="relative flex h-full min-w-0 flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-2 px-3 lg:px-5">
        <IconButton label="Open sidebar" onClick={onOpenSidebar} className="lg:hidden">
          <Menu className="size-[1.1rem]" />
        </IconButton>
        <h1 className="min-w-0 flex-1 truncate text-sm font-medium text-muted">{conversation?.title ?? ""}</h1>
        {scoped.length > 0 && (
          <span className="flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent-soft py-1 pl-2.5 pr-1 text-xs font-medium text-accent">
            <Filter className="size-3" />
            {scoped.length} document{scoped.length > 1 ? "s" : ""}
            <button onClick={() => setScoped([])} className="cursor-pointer rounded-full p-0.5 hover:bg-accent/15" aria-label="Search all documents">
              <X className="size-3" />
            </button>
          </span>
        )}
        <button
          onClick={onToggleLibrary}
          className={cn(
            "flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
            libraryOpen ? "bg-elevated text-fg" : "text-muted hover:bg-hover hover:text-fg",
          )}
        >
          <Library className="size-4" />
          <span className="hidden sm:inline">Library</span>
          {documents.length > 0 && (
            <span className="rounded-md bg-accent-soft px-1.5 font-mono text-[11px] font-semibold text-accent">{documents.length}</span>
          )}
        </button>
      </header>

      <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-7 px-4 pb-8 pt-4 lg:px-6">
            {messages.map((m) => (
              <Message key={m.id} message={m} />
            ))}
          </div>
        )}
      </div>

      <div className="relative mx-auto w-full max-w-3xl px-4 pb-4 lg:px-6">
        {showJump && (
          <button
            onClick={jumpToBottom}
            aria-label="Scroll to bottom"
            className="absolute -top-12 left-1/2 grid size-9 -translate-x-1/2 animate-fade-in cursor-pointer place-items-center rounded-full border border-border bg-panel text-muted shadow-lg hover:text-fg"
          >
            <ArrowDown className="size-4" />
          </button>
        )}
        <OfflineNotice />
        <Composer />
        <p className="mt-2 text-center text-[11px] text-subtle">
          Runs entirely on your machine · Answers can be wrong, check the cited sources
        </p>
      </div>
    </main>
  );
}

function OfflineNotice() {
  const { health, healthError } = useLibrary();
  let message: React.ReactNode = null;
  if (healthError) message = <>Can&apos;t reach the SmartRAG API on port 8000. Start it with <code>python start.py</code>.</>;
  else if (health && !health.ollama) message = <>Ollama isn&apos;t running. Start it with <code>ollama serve</code>.</>;
  else if (health && !health.llm) message = <>Model not found. Run <code>ollama pull {health.llm_model}</code>.</>;
  else if (health && !health.embeddings) message = <>Embedding model not found. Run <code>ollama pull {health.embedding_model}</code>.</>;
  if (!message) return null;
  return (
    <div className="mb-2 animate-fade-in rounded-xl border border-amber-500/25 bg-amber-500/10 px-3.5 py-2 text-[13px] text-amber-700 dark:text-amber-300 [&_code]:rounded [&_code]:bg-amber-500/15 [&_code]:px-1 [&_code]:font-mono [&_code]:text-[12px]">
      {message}
    </div>
  );
}

function EmptyState() {
  const send = useChat((s) => s.send);
  const { documents, loaded, upload } = useLibrary();
  const hasDocs = documents.length > 0;

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col items-center justify-center px-6 py-10 text-center">
      <div className="relative mb-6">
        <div className="absolute inset-0 -z-10 scale-[2.5] rounded-full bg-accent/20 blur-3xl" />
        <Logo className="size-14 rounded-2xl" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">What would you like to know?</h2>
      <p className="mt-2 max-w-md text-[15px] text-muted">
        {hasDocs
          ? `Ask questions across your ${documents.length} document${documents.length > 1 ? "s" : ""}. Every answer links back to its sources.`
          : "Add documents, images or audio, then ask questions about them. Everything stays on your machine."}
      </p>

      {hasDocs ? (
        <div className="mt-8 grid w-full gap-2.5 sm:grid-cols-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s.title}
              onClick={() => send(s.prompt)}
              className="cursor-pointer rounded-xl border border-border bg-panel px-4 py-3 text-left transition-all hover:-translate-y-0.5 hover:border-border-strong hover:shadow-md"
            >
              <p className="text-sm font-medium">{s.title}</p>
              <p className="mt-0.5 line-clamp-1 text-xs text-muted">{s.prompt}</p>
            </button>
          ))}
        </div>
      ) : (
        loaded && (
          <label className="mt-8 flex w-full max-w-md cursor-pointer flex-col items-center gap-2 rounded-2xl border border-dashed border-border-strong bg-panel/50 px-6 py-8 transition-colors hover:border-accent/60 hover:bg-accent-soft/40">
            <span className="text-sm font-medium">
              Drop files anywhere, or <span className="text-accent">browse</span>
            </span>
            <span className="text-xs text-subtle">PDF · DOCX · TXT · MD · PNG · JPG · MP3 · WAV · M4A</span>
            <input
              type="file"
              multiple
              hidden
              onChange={(e) => {
                if (e.target.files?.length) upload(Array.from(e.target.files));
                e.target.value = "";
              }}
            />
          </label>
        )
      )}
    </div>
  );
}
