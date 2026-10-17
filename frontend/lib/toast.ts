import { create } from "zustand";

export interface Toast {
  id: number;
  kind: "success" | "error";
  title: string;
  description?: string;
}

interface ToastState {
  toasts: Toast[];
  push: (toast: Omit<Toast, "id">) => void;
  dismiss: (id: number) => void;
}

let nextId = 1;

export const useToasts = create<ToastState>()((set, get) => ({
  toasts: [],
  push: (toast) => {
    const id = nextId++;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { ...toast, id }] }));
    setTimeout(() => get().dismiss(id), toast.kind === "error" ? 6000 : 3500);
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export const toast = {
  success: (title: string, description?: string) => useToasts.getState().push({ kind: "success", title, description }),
  error: (title: string, description?: string) => useToasts.getState().push({ kind: "error", title, description }),
};
