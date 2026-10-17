import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Source } from "./api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  status?: "streaming" | "done" | "error";
  error?: string;
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  /** Documents the chat is scoped to; empty means search everything. */
  scopedDocumentIds: string[];
  theme: "dark" | "light";

  newConversation: () => void;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  /** Append a message to the active conversation, creating one if needed. Returns the conversation id. */
  addMessage: (message: Message) => string;
  updateMessage: (conversationId: string, messageId: string, patch: Partial<Message>) => void;
  toggleScopedDocument: (id: string) => void;
  setScopedDocuments: (ids: string[]) => void;
  toggleTheme: () => void;
}

export const uid = () => crypto.randomUUID();

const titleFrom = (text: string) => {
  const line = text.trim().split("\n")[0];
  return line.length > 48 ? `${line.slice(0, 48).trimEnd()}…` : line || "New chat";
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeId: null,
      scopedDocumentIds: [],
      theme: "dark",

      newConversation: () => set({ activeId: null }),

      selectConversation: (id) => set({ activeId: id }),

      deleteConversation: (id) =>
        set((s) => ({
          conversations: s.conversations.filter((c) => c.id !== id),
          activeId: s.activeId === id ? null : s.activeId,
        })),

      addMessage: (message) => {
        const { activeId, conversations } = get();
        const now = Date.now();
        if (activeId && conversations.some((c) => c.id === activeId)) {
          set({
            conversations: conversations.map((c) =>
              c.id === activeId ? { ...c, messages: [...c.messages, message], updatedAt: now } : c,
            ),
          });
          return activeId;
        }
        const conversation: Conversation = {
          id: uid(),
          title: titleFrom(message.content),
          messages: [message],
          updatedAt: now,
        };
        set({ conversations: [conversation, ...conversations], activeId: conversation.id });
        return conversation.id;
      },

      updateMessage: (conversationId, messageId, patch) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id !== conversationId
              ? c
              : { ...c, messages: c.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m)) },
          ),
        })),

      toggleScopedDocument: (id) =>
        set((s) => ({
          scopedDocumentIds: s.scopedDocumentIds.includes(id)
            ? s.scopedDocumentIds.filter((d) => d !== id)
            : [...s.scopedDocumentIds, id],
        })),

      setScopedDocuments: (ids) => set({ scopedDocumentIds: ids }),

      toggleTheme: () => set((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),
    }),
    {
      name: "smartrag",
      version: 1,
      // A reload mid-stream would otherwise leave a message stuck "streaming"
      onRehydrateStorage: () => (state) => {
        state?.conversations.forEach((c) =>
          c.messages.forEach((m) => {
            if (m.status === "streaming") m.status = m.content ? "done" : "error";
          }),
        );
      },
    },
  ),
);
