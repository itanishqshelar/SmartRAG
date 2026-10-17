"use client";

import { Check, ChevronDown, CircleAlert, Copy } from "lucide-react";
import { useMemo, useState } from "react";
import type { Source } from "@/lib/api";
import { useLibrary } from "@/lib/documents";
import type { Message as MessageType } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Markdown } from "./Markdown";
import { FileIcon, IconButton, Logo, TypingDots } from "./ui";

const NO_SOURCES: Source[] = [];

export function Message({ message }: { message: MessageType }) {
  if (message.role === "user") {
    return (
      <div className="flex animate-slide-up justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-elevated px-4 py-2.5 text-[0.95rem] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }
  return <AssistantMessage message={message} />;
}

function AssistantMessage({ message }: { message: MessageType }) {
  const streaming = message.status === "streaming";
  const sources = message.sources ?? NO_SOURCES;
  const [copied, setCopied] = useState(false);

  const cited = useMemo(
    () => new Set(sources.filter((s) => new RegExp(`\\[${s.index}\\]`).test(message.content)).map((s) => s.index)),
    [sources, message.content],
  );

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group flex animate-slide-up gap-3.5">
      <Logo className="mt-0.5 size-7 rounded-lg" />
      <div className="min-w-0 flex-1">
        {streaming && !message.content ? (
          <div className="flex h-7 items-center gap-2.5 text-sm text-muted">
            <TypingDots />
            {message.sources === undefined ? "Searching your library…" : sources.length ? `Reading ${sources.length} excerpt${sources.length > 1 ? "s" : ""}…` : "Thinking…"}
          </div>
        ) : (
          message.content && <Markdown content={message.content} sources={sources} streaming={streaming} />
        )}

        {message.status === "error" && (
          <div className="mt-2 flex items-start gap-2 rounded-xl border border-danger/25 bg-danger/8 px-3.5 py-2.5 text-sm text-danger">
            <CircleAlert className="mt-0.5 size-4 shrink-0" />
            <span>{message.error ?? "Something went wrong."}</span>
          </div>
        )}

        {!streaming && sources.length > 0 && <SourceList sources={sources} cited={cited} />}

        {!streaming && message.content && (
          <div className="mt-1.5 flex opacity-0 transition-opacity group-hover:opacity-100">
            <IconButton label="Copy answer" onClick={copy} className="size-7">
              {copied ? <Check className="size-3.5 text-success" /> : <Copy className="size-3.5" />}
            </IconButton>
          </div>
        )}
      </div>
    </div>
  );
}

function SourceList({ sources, cited }: { sources: Source[]; cited: Set<number> }) {
  const [showAll, setShowAll] = useState(false);
  const primary = sources.filter((s) => cited.has(s.index));
  const others = sources.filter((s) => !cited.has(s.index));
  const visible = showAll ? [...primary, ...others] : primary;

  return (
    <div className="mt-4">
      {visible.length > 0 && (
        <div className="grid gap-2 sm:grid-cols-2">
          {visible.map((s) => (
            <SourceCard key={s.index} source={s} dimmed={!cited.has(s.index)} />
          ))}
        </div>
      )}
      {others.length > 0 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="mt-2 flex cursor-pointer items-center gap-1 text-xs text-muted transition-colors hover:text-fg"
        >
          <ChevronDown className={cn("size-3.5 transition-transform", showAll && "rotate-180")} />
          {showAll
            ? "Hide other excerpts"
            : `${others.length} ${primary.length ? "more " : ""}retrieved excerpt${others.length > 1 ? "s" : ""}`}
        </button>
      )}
    </div>
  );
}

function SourceCard({ source, dimmed }: { source: Source; dimmed: boolean }) {
  const preview = useLibrary((s) => s.preview);
  return (
    <button
      onClick={() => source.document_id && preview(source.document_id, source.excerpt, source.page)}
      className={cn(
        "flex cursor-pointer items-start gap-2.5 rounded-xl border border-border bg-panel p-2.5 text-left transition-all hover:border-border-strong hover:shadow-sm",
        dimmed && "opacity-70 hover:opacity-100",
      )}
    >
      <div className="relative">
        <FileIcon type={source.document_type} className="size-8" />
        <span className="absolute -right-1.5 -top-1.5 grid size-4 place-items-center rounded-full bg-accent font-mono text-[9px] font-bold text-accent-fg">
          {source.index}
        </span>
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium">{source.filename}</p>
        <p className="line-clamp-2 text-xs leading-snug text-muted">
          {source.page ? <span className="text-subtle">p.{source.page} · </span> : null}
          {source.excerpt}
        </p>
      </div>
    </button>
  );
}
