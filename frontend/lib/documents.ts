import { create } from "zustand";
import { api, type Health, type SmartDocument } from "./api";
import { toast } from "./toast";
import { uid, useChatStore } from "./store";

export interface Upload {
  id: string;
  filename: string;
  size: number;
  status: "processing" | "error";
  error?: string;
}

interface LibraryState {
  documents: SmartDocument[];
  loaded: boolean;
  uploads: Upload[];
  health: Health | null;
  healthError: boolean;
  previewing: { document: SmartDocument; excerpt?: string; page?: number | null } | null;

  refresh: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  upload: (files: File[]) => Promise<void>;
  dismissUpload: (id: string) => void;
  remove: (id: string) => Promise<void>;
  removeAll: () => Promise<void>;
  preview: (documentId: string, excerpt?: string, page?: number | null) => void;
  closePreview: () => void;
}

export const useLibrary = create<LibraryState>()((set, get) => ({
  documents: [],
  loaded: false,
  uploads: [],
  health: null,
  healthError: false,
  previewing: null,

  refresh: async () => {
    try {
      set({ documents: await api.documents(), loaded: true });
    } catch {
      set({ loaded: true });
    }
  },

  refreshHealth: async () => {
    try {
      set({ health: await api.health(), healthError: false });
    } catch {
      set({ healthError: true });
    }
  },

  upload: async (files) => {
    const { health } = get();
    const allowed = health?.supported_extensions;
    const maxBytes = (health?.max_upload_size_mb ?? Infinity) * 1024 * 1024;

    await Promise.all(
      files.map(async (file) => {
        const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;
        const id = uid();
        const fail = (error: string) =>
          set((s) => ({
            uploads: [
              ...s.uploads.filter((u) => u.id !== id),
              { id, filename: file.name, size: file.size, status: "error", error },
            ],
          }));

        if (allowed && !allowed.includes(ext)) return fail(`${ext} files aren't supported`);
        if (file.size > maxBytes) return fail(`Larger than ${health?.max_upload_size_mb} MB`);

        set((s) => ({ uploads: [...s.uploads, { id, filename: file.name, size: file.size, status: "processing" }] }));
        try {
          const doc = await api.upload(file);
          set((s) => ({
            documents: [doc, ...s.documents],
            uploads: s.uploads.filter((u) => u.id !== id),
          }));
          toast.success(`Added ${doc.filename}`, `${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"} indexed`);
        } catch (e) {
          fail(e instanceof Error ? e.message : "Upload failed");
        }
      }),
    );
    get().refreshHealth();
  },

  dismissUpload: (id) => set((s) => ({ uploads: s.uploads.filter((u) => u.id !== id) })),

  remove: async (id) => {
    const doc = get().documents.find((d) => d.id === id);
    try {
      await api.deleteDocument(id);
      set((s) => ({ documents: s.documents.filter((d) => d.id !== id) }));
      const chat = useChatStore.getState();
      chat.setScopedDocuments(chat.scopedDocumentIds.filter((d) => d !== id));
      toast.success(`Removed ${doc?.filename ?? "document"}`);
    } catch (e) {
      toast.error("Couldn't remove document", e instanceof Error ? e.message : undefined);
    }
    get().refreshHealth();
  },

  removeAll: async () => {
    try {
      await api.deleteAllDocuments();
      set({ documents: [] });
      useChatStore.getState().setScopedDocuments([]);
      toast.success("Library cleared");
    } catch (e) {
      toast.error("Couldn't clear library", e instanceof Error ? e.message : undefined);
    }
    get().refreshHealth();
  },

  preview: (documentId, excerpt, page) => {
    const document = get().documents.find((d) => d.id === documentId);
    if (document) set({ previewing: { document, excerpt, page } });
    else toast.error("This document is no longer in your library");
  },

  closePreview: () => set({ previewing: null }),
}));
