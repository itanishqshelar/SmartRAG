"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { useLibrary } from "@/lib/documents";
import { useChatStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { ChatView } from "./ChatView";
import { DocumentPreview } from "./DocumentPreview";
import { DropOverlay } from "./DropOverlay";
import { LibraryPanel } from "./LibraryPanel";
import { Sidebar } from "./Sidebar";
import { Toaster } from "./Toaster";

const HEALTH_POLL_MS = 8000;

// Conversations live in localStorage, so render only on the client to avoid hydration mismatches
const useIsClient = () =>
  useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );

export function App() {
  const isClient = useIsClient();
  const [sidebarOpen, setSidebarOpen] = useState(false); // mobile drawer
  // Open the library by default on wide screens
  const [libraryOpen, setLibraryOpen] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 1280px)").matches,
  );
  const theme = useChatStore((s) => s.theme);
  const newConversation = useChatStore((s) => s.newConversation);
  const { refresh, refreshHealth } = useLibrary();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    refresh();
    refreshHealth();
    const id = setInterval(refreshHealth, HEALTH_POLL_MS);
    return () => clearInterval(id);
  }, [refresh, refreshHealth]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        newConversation();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newConversation]);

  if (!isClient) return <div className="h-full bg-bg" />;

  return (
    <div className="flex h-full">
      {/* Conversations: fixed column on desktop, drawer on mobile */}
      <div className="hidden w-[272px] shrink-0 border-r border-border lg:block">
        <Sidebar />
      </div>
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 animate-fade-in bg-black/40" onClick={() => setSidebarOpen(false)} />
          <div className="relative h-full w-[280px] animate-slide-up border-r border-border shadow-2xl">
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </div>
        </div>
      )}

      <ChatView
        libraryOpen={libraryOpen}
        onToggleLibrary={() => setLibraryOpen((v) => !v)}
        onOpenSidebar={() => setSidebarOpen(true)}
      />

      {/* Library: side panel on wide screens, overlay on narrow ones */}
      {libraryOpen && (
        <div className="fixed inset-0 z-30 animate-fade-in bg-black/40 xl:hidden" onClick={() => setLibraryOpen(false)} />
      )}
      <div
        className={cn(
          "shrink-0 border-l border-border",
          libraryOpen
            ? "fixed inset-y-0 right-0 z-30 w-[340px] max-w-full animate-slide-in-right shadow-2xl xl:static xl:z-auto xl:shadow-none"
            : "hidden",
        )}
      >
        <LibraryPanel onClose={() => setLibraryOpen(false)} />
      </div>

      <DocumentPreview />
      <DropOverlay />
      <Toaster />
    </div>
  );
}
