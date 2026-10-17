"use client";

import { Check, CircleAlert, Eye, Trash2, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";
import type { SmartDocument } from "@/lib/api";
import { useLibrary } from "@/lib/documents";
import { useChatStore } from "@/lib/store";
import { cn, formatBytes, timeAgo } from "@/lib/utils";
import { FileIcon, IconButton, Spinner, typeLabel } from "./ui";

export function LibraryPanel({ onClose }: { onClose: () => void }) {
  const { documents, loaded, uploads, upload, dismissUpload, removeAll, health } = useLibrary();
  const { scopedDocumentIds, setScopedDocuments } = useChatStore();
  const [confirmClear, setConfirmClear] = useState(false);

  return (
    <aside className="flex h-full w-full flex-col bg-panel">
      <div className="flex items-center gap-2 px-4 pb-2 pt-4">
        <div className="flex-1">
          <h2 className="text-[15px] font-semibold tracking-tight">Library</h2>
          <p className="text-xs text-muted">
            {documents.length} item{documents.length === 1 ? "" : "s"}
            {health ? ` · ${health.chunks} chunks indexed` : ""}
          </p>
        </div>
        <IconButton label="Close library" onClick={onClose}>
          <X className="size-4" />
        </IconButton>
      </div>

      <div className="px-4 py-2">
        <DropZone onFiles={upload} accept={health?.supported_extensions} />
      </div>

      {documents.length > 0 && (
        <div className="mx-4 mb-1 mt-2 flex items-center justify-between text-xs">
          <span className="text-muted">
            {scopedDocumentIds.length
              ? `Chat uses ${scopedDocumentIds.length} selected`
              : "Chat searches everything"}
          </span>
          {scopedDocumentIds.length > 0 && (
            <button onClick={() => setScopedDocuments([])} className="cursor-pointer font-medium text-accent hover:underline">
              Use all
            </button>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {uploads.map((u) => (
          <div key={u.id} className="mx-2 my-1 flex animate-fade-in items-center gap-3 rounded-xl border border-border bg-bg p-2.5">
            <div
              className={cn(
                "grid size-9 shrink-0 place-items-center rounded-lg",
                u.status === "error" ? "bg-danger/10 text-danger" : "bg-accent-soft text-accent",
              )}
            >
              {u.status === "error" ? <CircleAlert className="size-4" /> : <Spinner />}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-medium">{u.filename}</p>
              <p className={cn("truncate text-xs", u.status === "error" ? "text-danger" : "text-muted")}>
                {u.status === "error" ? u.error : "Extracting & indexing…"}
              </p>
            </div>
            {u.status === "error" && (
              <IconButton label="Dismiss" className="size-7" onClick={() => dismissUpload(u.id)}>
                <X className="size-3.5" />
              </IconButton>
            )}
          </div>
        ))}

        {loaded && documents.length === 0 && uploads.length === 0 && (
          <p className="px-6 py-8 text-center text-xs leading-relaxed text-subtle">
            Nothing here yet. Add PDFs, Word docs, notes, images or audio and SmartRAG will index them locally.
          </p>
        )}

        {documents.map((doc) => (
          <DocumentRow key={doc.id} doc={doc} />
        ))}
      </div>

      {documents.length > 0 && (
        <div className="border-t border-border p-3">
          <button
            onClick={() => {
              if (!confirmClear) {
                setConfirmClear(true);
                setTimeout(() => setConfirmClear(false), 3000);
              } else {
                removeAll();
                setConfirmClear(false);
              }
            }}
            className={cn(
              "w-full cursor-pointer rounded-lg py-2 text-xs font-medium transition-colors",
              confirmClear ? "bg-danger/10 text-danger" : "text-muted hover:bg-hover hover:text-fg",
            )}
          >
            {confirmClear ? "Click again to delete everything" : "Clear library"}
          </button>
        </div>
      )}
    </aside>
  );
}

function DocumentRow({ doc }: { doc: SmartDocument }) {
  const { remove, preview } = useLibrary();
  const { scopedDocumentIds, toggleScopedDocument } = useChatStore();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const scoped = scopedDocumentIds.includes(doc.id);

  return (
    <div
      className={cn(
        "group mx-0 my-0.5 flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-hover",
        scoped && "bg-accent-soft hover:bg-accent-soft",
      )}
    >
      <button
        onClick={() => toggleScopedDocument(doc.id)}
        aria-label={scoped ? "Remove from chat scope" : "Limit chat to this document"}
        title={scoped ? "Remove from chat scope" : "Limit chat to this document"}
        className="relative cursor-pointer"
      >
        <FileIcon type={doc.document_type} />
        <span
          className={cn(
            "absolute -bottom-1 -right-1 grid size-4 place-items-center rounded-full border-2 border-panel transition-all",
            scoped ? "scale-100 bg-accent text-accent-fg" : "scale-0 bg-elevated group-hover:scale-100",
          )}
        >
          <Check className="size-2.5" strokeWidth={3.5} />
        </span>
      </button>
      <button onClick={() => preview(doc.id)} className="min-w-0 flex-1 cursor-pointer text-left">
        <p className="truncate text-[13px] font-medium" title={doc.filename}>
          {doc.filename}
        </p>
        <p className="truncate text-[11.5px] text-muted">
          {typeLabel(doc.document_type)} · {formatBytes(doc.size)} · {doc.chunks} chunk{doc.chunks === 1 ? "" : "s"} ·{" "}
          {timeAgo(doc.created_at)}
        </p>
      </button>
      <div className={cn("flex items-center opacity-0 transition-opacity group-hover:opacity-100", confirmDelete && "opacity-100")}>
        <IconButton label="Preview" className="size-7" onClick={() => preview(doc.id)}>
          <Eye className="size-3.5" />
        </IconButton>
        <IconButton
          label={confirmDelete ? "Click again to delete" : "Delete"}
          className={cn("size-7", confirmDelete && "bg-danger/12 text-danger hover:bg-danger/20 hover:text-danger")}
          onClick={() => {
            if (confirmDelete) return remove(doc.id);
            setConfirmDelete(true);
            setTimeout(() => setConfirmDelete(false), 3000);
          }}
        >
          <Trash2 className="size-3.5" />
        </IconButton>
      </div>
    </div>
  );
}

function DropZone({ onFiles, accept }: { onFiles: (files: File[]) => void; accept?: string[] }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <>
      <button
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOver(false);
          if (e.dataTransfer.files.length) onFiles(Array.from(e.dataTransfer.files));
        }}
        className={cn(
          "flex w-full cursor-pointer flex-col items-center gap-1.5 rounded-xl border border-dashed px-4 py-5 text-center transition-all",
          over ? "border-accent bg-accent-soft" : "border-border-strong hover:border-accent/60 hover:bg-hover",
        )}
      >
        <div className="grid size-9 place-items-center rounded-full bg-accent-soft text-accent">
          <UploadCloud className="size-[1.1rem]" />
        </div>
        <p className="text-[13px] font-medium">
          Drop files or <span className="text-accent">browse</span>
        </p>
        <p className="text-[11px] text-subtle">PDF, DOCX, TXT, MD · images · audio</p>
      </button>
      <input
        ref={input}
        type="file"
        multiple
        hidden
        accept={accept?.join(",")}
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />
    </>
  );
}
