export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type DocumentType = "text" | "pdf" | "docx" | "image" | "audio";

export interface SmartDocument {
  id: string;
  filename: string;
  content_type: string;
  document_type: DocumentType;
  size: number;
  chunks: number;
  created_at: string;
}

export interface Source {
  index: number;
  document_id: string | null;
  filename: string;
  document_type: DocumentType;
  page: number | null;
  score: number;
  excerpt: string;
}

export interface Health {
  ready: boolean;
  ollama: boolean;
  llm: boolean;
  embeddings: boolean;
  llm_model: string;
  embedding_model: string;
  documents: number;
  chunks: number;
  supported_extensions: string[];
  max_upload_size_mb: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export type ChatEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${res.status})`);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

function formData(file: Blob, filename?: string) {
  const form = new FormData();
  form.append("file", file, filename ?? (file instanceof File ? file.name : "upload"));
  return form;
}

export const api = {
  health: () => request<Health>("/api/health", { cache: "no-store" }),
  documents: () => request<SmartDocument[]>("/api/documents", { cache: "no-store" }),
  upload: (file: File) =>
    request<SmartDocument>("/api/documents", { method: "POST", body: formData(file) }),
  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),
  deleteAllDocuments: () => request<void>("/api/documents", { method: "DELETE" }),
  transcribe: (audio: Blob) =>
    request<{ text: string }>("/api/transcribe", { method: "POST", body: formData(audio, "recording.webm") }),
  fileUrl: (id: string) => `${API_URL}/api/documents/${id}/file`,

  /** POST a chat message and yield streamed events. */
  async *chat(
    body: { message: string; history: ChatTurn[]; document_ids?: string[] },
    signal?: AbortSignal,
  ): AsyncGenerator<ChatEvent> {
    const res = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`Chat request failed (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) yield JSON.parse(line) as ChatEvent;
      }
    }
    if (buffer.trim()) yield JSON.parse(buffer) as ChatEvent;
  },
};
