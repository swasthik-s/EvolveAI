"use client";

import { useState } from "react";
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { BrainCircuit, ChevronDown, MoreHorizontal, PanelLeftClose, Plus, Search, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const aui = useAui();
  const loading = useAuiState((state) => state.threads.isLoading);
  const [query, setQuery] = useState("");
  const [about, setAbout] = useState(false);

  function newChat() {
    aui.threads.switchToNewThread();
    onNavigate?.();
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary text-primary-foreground"><BrainCircuit size={18} /></span>
          <span className="font-display text-[15px] font-semibold tracking-tight">Evolve<span className="text-primary">AI</span></span>
        </div>
        {onNavigate && (
          <Button variant="ghost" size="icon-sm" aria-label="Close sidebar" onClick={onNavigate}><PanelLeftClose size={17} /></Button>
        )}
      </div>
      <div className="px-3 pb-3">
        <Button variant="secondary" className="w-full justify-start gap-2" onClick={newChat}>
          <Plus size={16} /> New chat <span className="ml-auto text-[10px] text-muted-foreground">⌘K</span>
        </Button>
      </div>
      <div className="px-3 pb-3">
        <label className="flex items-center gap-2 rounded-xl border border-sidebar-border bg-background/40 px-3 py-2 text-sm">
          <Search size={14} className="text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search chats"
            aria-label="Search chats"
            className="w-full bg-transparent outline-none placeholder:text-muted-foreground"
          />
        </label>
      </div>
      <div className="px-4 pb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Recent chats</div>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2">
        {loading && (
          <div className="space-y-2 px-2" aria-hidden>
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-5/6" />
            <Skeleton className="h-9 w-4/6" />
          </div>
        )}
        <ThreadListPrimitive.Root>
          <ThreadListPrimitive.Items>
            {() => <ThreadRow query={query} onNavigate={onNavigate} />}
          </ThreadListPrimitive.Items>
        </ThreadListPrimitive.Root>
      </div>
      <div className="border-t border-sidebar-border p-3">
        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-xl px-2 py-2 text-left text-sm hover:bg-muted"
          aria-expanded={about}
          onClick={() => setAbout((value) => !value)}
        >
          <span className="flex-1">About this engine</span>
          <ChevronDown size={14} className={`transition-transform ${about ? "rotate-180" : ""}`} />
        </button>
        {about && (
          <p className="px-2 pb-2 text-xs leading-5 text-muted-foreground">
            No hosted language model is used. EvolveAI follows explicit rules, recalls saved context, and can search public sources. Chats stay in this browser.
          </p>
        )}
      </div>
    </div>
  );
}

function ThreadRow({ query, onNavigate }: { query: string; onNavigate?: () => void }) {
  const aui = useAui();
  const title = useAuiState((state) => state.threadListItem.title || "New conversation");
  const active = useAuiState((state) => state.threadListItem.id === state.threads.mainThreadId);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(title);
  if (query && !title.toLowerCase().includes(query.trim().toLowerCase())) return null;

  function commit() {
    const next = draft.trim().slice(0, 80);
    if (next) aui.threadListItem.rename(next);
    setRenaming(false);
  }

  return (
    <ThreadListItemPrimitive.Root className={`group flex items-center gap-1 rounded-lg ${active ? "bg-muted" : "hover:bg-muted/70"}`}>
      {renaming ? (
        <input
          autoFocus
          value={draft}
          aria-label="Rename conversation"
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") setRenaming(false);
          }}
          className="m-1 min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      ) : (
        <ThreadListItemPrimitive.Trigger
          className="min-w-0 flex-1 truncate px-3 py-2.5 text-left text-xs"
          onClick={() => onNavigate?.()}
        >
          {title}
        </ThreadListItemPrimitive.Trigger>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" aria-label="Conversation actions">
          <MoreHorizontal size={15} />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => { setDraft(title); setRenaming(true); }}>Rename</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ThreadListItemPrimitive.Delete className="mr-1 inline-flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-background hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" aria-label="Delete conversation">
        <Trash2 size={14} />
      </ThreadListItemPrimitive.Delete>
    </ThreadListItemPrimitive.Root>
  );
}
