"use client";

import { ArrowUp, Mic, Paperclip, Square, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useChat } from "@/lib/chat";
import { useLibrary } from "@/lib/documents";
import { toast } from "@/lib/toast";
import { useRecorder } from "@/lib/useRecorder";
import { cn } from "@/lib/utils";
import { IconButton, Spinner } from "./ui";

export function Composer() {
  const [text, setText] = useState("");
  const [transcribing, setTranscribing] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const { streaming, send, stop } = useChat();
  const { health, upload } = useLibrary();
  const recorder = useRecorder();

  // Auto-grow up to a max height
  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  useEffect(() => textarea.current?.focus(), []);

  const submit = () => {
    if (!text.trim() || streaming) return;
    send(text);
    setText("");
  };

  const toggleRecording = async () => {
    if (!recorder.recording) {
      try {
        await recorder.start();
      } catch {
        toast.error("Microphone unavailable", "Allow microphone access in your browser to use voice input.");
      }
      return;
    }
    const audio = await recorder.stop();
    if (!audio?.size) return;
    setTranscribing(true);
    try {
      const { text: transcript } = await api.transcribe(audio);
      if (transcript) setText((t) => (t ? `${t} ${transcript}` : transcript));
      else toast.error("No speech detected");
    } catch (e) {
      toast.error("Transcription failed", e instanceof Error ? e.message : undefined);
    } finally {
      setTranscribing(false);
      textarea.current?.focus();
    }
  };

  const canSend = text.trim().length > 0 && !transcribing;

  return (
    <div className="rounded-2xl border border-border bg-panel shadow-[0_8px_30px_-12px_rgba(0,0,0,0.25)] transition-colors focus-within:border-border-strong">
      {recorder.recording ? (
        <div className="flex h-[52px] items-center gap-3 px-4">
          <span className="relative flex size-2.5">
            <span className="absolute inset-0 animate-ping rounded-full bg-red-500 opacity-60" />
            <span className="relative size-2.5 rounded-full bg-red-500" />
          </span>
          <span className="text-sm">Listening…</span>
          <span className="font-mono text-sm tabular-nums text-muted">
            {Math.floor(recorder.seconds / 60)}:{String(recorder.seconds % 60).padStart(2, "0")}
          </span>
        </div>
      ) : (
        <textarea
          ref={textarea}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
          disabled={transcribing}
          placeholder={transcribing ? "Transcribing…" : "Ask anything about your documents…"}
          className="block max-h-[220px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[0.95rem] leading-relaxed outline-none placeholder:text-subtle"
        />
      )}

      <div className="flex items-center gap-1 px-2 pb-2 pt-1">
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          accept={health?.supported_extensions.join(",")}
          onChange={(e) => {
            if (e.target.files?.length) upload(Array.from(e.target.files));
            e.target.value = "";
          }}
        />
        <IconButton label="Add files to library" onClick={() => fileInput.current?.click()} disabled={recorder.recording}>
          <Paperclip className="size-[1.05rem]" />
        </IconButton>
        {recorder.recording ? (
          <IconButton label="Cancel recording" onClick={recorder.cancel}>
            <X className="size-[1.05rem]" />
          </IconButton>
        ) : null}
        <IconButton
          label={recorder.recording ? "Stop and transcribe" : "Voice input"}
          onClick={toggleRecording}
          disabled={transcribing}
          className={cn(recorder.recording && "bg-red-500/12 text-red-500 hover:bg-red-500/20 hover:text-red-500")}
        >
          {transcribing ? <Spinner /> : <Mic className="size-[1.05rem]" />}
        </IconButton>

        <div className="ml-auto">
          {streaming ? (
            <button
              onClick={stop}
              aria-label="Stop generating"
              className="grid size-8 cursor-pointer place-items-center rounded-lg bg-fg text-bg transition-opacity hover:opacity-85"
            >
              <Square className="size-3 fill-current" />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!canSend}
              aria-label="Send message"
              className="grid size-8 cursor-pointer place-items-center rounded-lg bg-accent text-accent-fg transition-all hover:brightness-110 disabled:cursor-default disabled:bg-elevated disabled:text-subtle"
            >
              <ArrowUp className="size-4" strokeWidth={2.4} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
