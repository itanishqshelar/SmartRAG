import { create } from "zustand";
import { api, type ChatTurn } from "./api";
import { uid, useChatStore, type Message } from "./store";

interface StreamState {
  streaming: boolean;
  controller: AbortController | null;
  send: (text: string) => Promise<void>;
  stop: () => void;
}

export const useChat = create<StreamState>()((set, get) => ({
  streaming: false,
  controller: null,

  send: async (text) => {
    const message = text.trim();
    if (!message || get().streaming) return;

    const store = useChatStore.getState();
    const prior = store.conversations.find((c) => c.id === store.activeId)?.messages ?? [];
    const history: ChatTurn[] = prior
      .filter((m) => m.status !== "error" && m.content)
      .map(({ role, content }) => ({ role, content }));

    const conversationId = store.addMessage({ id: uid(), role: "user", content: message, createdAt: Date.now() });
    const assistantId = uid();
    useChatStore.getState().addMessage({
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
      createdAt: Date.now(),
    });
    const update = (patch: Partial<Message>) =>
      useChatStore.getState().updateMessage(conversationId, assistantId, patch);

    const controller = new AbortController();
    set({ streaming: true, controller });
    let content = "";
    try {
      const scoped = useChatStore.getState().scopedDocumentIds;
      for await (const event of api.chat(
        { message, history, document_ids: scoped.length ? scoped : undefined },
        controller.signal,
      )) {
        if (event.type === "sources") update({ sources: event.sources });
        else if (event.type === "token") update({ content: (content += event.content) });
        else if (event.type === "error") throw new Error(event.message);
      }
      update({ status: "done" });
    } catch (e) {
      if (controller.signal.aborted) {
        update({ status: "done", content: content || "_Stopped._" });
      } else {
        const error =
          e instanceof TypeError ? "Can't reach the SmartRAG API. Is the backend running?" : (e as Error).message;
        update({ status: "error", content, error });
      }
    } finally {
      set({ streaming: false, controller: null });
    }
  },

  stop: () => get().controller?.abort(),
}));
