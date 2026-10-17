"use client";

import { ExternalLink, Quote, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type SmartDocument } from "@/lib/api";
import { useLibrary } from "@/lib/documents";
import { formatBytes } from "@/lib/utils";
import { FileIcon, IconButton, Spinner, typeLabel } from "./ui";

export function DocumentPreview() {
  const { previewing, closePreview } = useLibrary();

  useEffect(() => {
    if (!previewing) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closePreview();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previewing, closePreview]);

  if (!previewing) return null;
  const { document: doc, excerpt, page } = previewing;
  const url = api.fileUrl(doc.id);

  return (
    <div className="fixed inset-0 z-50 flex animate-fade-in items-center justify-center bg-black/50 p-4 backdrop-blur-sm" onClick={closePreview}>
      <div
        role="dialog"
        aria-label={`Preview of ${doc.filename}`}
        onClick={(e) => e.stopPropagation()}
        className="flex h-[min(88vh,900px)] w-full max-w-4xl animate-slide-up flex-col overflow-hidden rounded-2xl border border-border bg-panel shadow-2xl"
      >
        <header className="flex items-center gap-3 border-b border-border px-4 py-3">
          <FileIcon type={doc.document_type} />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold">{doc.filename}</p>
            <p className="text-xs text-muted">
              {typeLabel(doc.document_type)} · {formatBytes(doc.size)} · {doc.chunks} chunks
            </p>
          </div>
          <a href={url} target="_blank" rel="noreferrer" className="grid size-8 place-items-center rounded-lg text-muted hover:bg-hover hover:text-fg" title="Open in new tab">
            <ExternalLink className="size-4" />
          </a>
          <IconButton label="Close" onClick={closePreview}>
            <X className="size-4" />
          </IconButton>
        </header>

        {excerpt && (
          <div className="flex gap-3 border-b border-border bg-accent-soft/60 px-5 py-3.5">
            <Quote className="mt-0.5 size-4 shrink-0 text-accent" />
            <div className="min-w-0">
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-accent">
                Cited excerpt{page ? ` · page ${page}` : ""}
              </p>
              <p className="line-clamp-4 text-[13px] leading-relaxed">{excerpt}</p>
            </div>
          </div>
        )}

        <div className="min-h-0 flex-1 bg-bg">
          <PreviewBody doc={doc} url={url} page={page} />
        </div>
      </div>
    </div>
  );
}

function PreviewBody({ doc, url, page }: { doc: SmartDocument; url: string; page?: number | null }) {
  switch (doc.document_type) {
    case "pdf":
      return <iframe src={`${url}#page=${page ?? 1}&view=FitH`} title={doc.filename} className="size-full" />;
    case "image":
      return (
        <div className="grid size-full place-items-center overflow-auto p-6">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={url} alt={doc.filename} className="max-h-full max-w-full rounded-lg object-contain shadow-lg" />
        </div>
      );
    case "audio":
      return (
        <div className="grid size-full place-items-center p-6">
          <div className="flex w-full max-w-md flex-col items-center gap-5">
            <FileIcon type="audio" className="size-16 rounded-2xl" />
            <audio src={url} controls className="w-full" />
          </div>
        </div>
      );
    case "text":
      return <TextPreview url={url} />;
    default:
      return (
        <div className="grid size-full place-items-center p-6 text-center">
          <div>
            <p className="text-sm text-muted">This file type can&apos;t be previewed in the browser.</p>
            <a href={url} download={doc.filename} className="mt-3 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-fg hover:brightness-110">
              Download {doc.filename}
            </a>
          </div>
        </div>
      );
  }
}

function TextPreview({ url }: { url: string }) {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => r.text())
      .then((t) => !cancelled && setText(t))
      .catch(() => !cancelled && setText("Couldn't load file."));
    return () => {
      cancelled = true;
    };
  }, [url]);

  if (text === null) return <div className="grid size-full place-items-center text-muted"><Spinner /></div>;
  return <pre className="size-full overflow-auto whitespace-pre-wrap p-6 font-mono text-[13px] leading-relaxed">{text}</pre>;
}
