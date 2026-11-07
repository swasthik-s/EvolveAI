"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { ThemeProvider, useTheme } from "next-themes";
import { AssistantRuntimeProvider, useAui, useAuiState, useRemoteThreadListRuntime } from "@assistant-ui/react";
import { BrainCircuit, Menu, Moon, PanelLeft, ShieldCheck, Sun } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/chat/sidebar";
import { ThreadView } from "@/components/chat/thread-view";
import { createEvolveThreadListAdapter, evolveModelAdapter, latestRemoteId, migrateLegacyThreads, rememberBranchHead } from "@/lib/evolve-runtime";
import { useLocalRuntime } from "@assistant-ui/react";

function useIsClient() {
  return useSyncExternalStore(() => () => {}, () => true, () => false);
}

export function ChatApp() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
      <TooltipProvider>
        <Toaster />
        <ChatRuntime />
      </TooltipProvider>
    </ThemeProvider>
  );
}

function useEvolveRuntime() {
  return useLocalRuntime(evolveModelAdapter);
}

function ChatRuntime() {
  const client = useIsClient();
  if (!client) return <AppSkeleton />;
  return <ChatRuntimeInner />;
}

function ChatRuntimeInner() {
  const [adapter] = useState(() => {
    migrateLegacyThreads();
    return createEvolveThreadListAdapter();
  });
  const [initialThreadId] = useState(() => latestRemoteId());
  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook: useEvolveRuntime,
    initialThreadId,
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Shell />
    </AssistantRuntimeProvider>
  );
}

function Shell() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const aui = useAui();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        aui.threads.switchToNewThread();
        setMobileOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [aui]);

  return (
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      {!collapsed && (
        <aside className="hidden w-[280px] shrink-0 border-r border-sidebar-border md:flex md:flex-col">
          <Sidebar onNavigate={() => setCollapsed(true)} />
        </aside>
      )}
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent aria-describedby={undefined}>
          <SheetTitle className="sr-only">Conversations</SheetTitle>
          <Sidebar onNavigate={() => setMobileOpen(false)} />
        </SheetContent>
      </Sheet>
      <section className="flex min-w-0 flex-1 flex-col">
        <Header
          showMenu={collapsed}
          onOpenDesktop={() => setCollapsed(false)}
          onOpenMobile={() => setMobileOpen(true)}
        />
        <BranchMemory />
        <ThreadView />
      </section>
    </main>
  );
}

function Header({ showMenu, onOpenDesktop, onOpenMobile }: { showMenu: boolean; onOpenDesktop: () => void; onOpenMobile: () => void }) {
  const health = useHealth();
  const { resolvedTheme, setTheme } = useTheme();
  const dark = resolvedTheme !== "light";
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border px-3 md:px-5">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open conversations" onClick={onOpenMobile}><Menu size={18} /></Button>
        {showMenu && <Button variant="ghost" size="icon" className="hidden md:inline-flex" aria-label="Open sidebar" onClick={onOpenDesktop}><PanelLeft size={18} /></Button>}
        <div>
          <div className="flex items-center gap-2 font-display text-sm font-semibold tracking-tight"><BrainCircuit size={16} className="text-primary" /> EvolveAI</div>
          <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span className={`h-1.5 w-1.5 rounded-full ${health === "online" ? "bg-emerald-500" : health === "offline" ? "bg-amber-500" : "bg-muted-foreground"}`} />
            {health === "online" ? "Ready" : health === "offline" ? "Connection issue" : "Checking"}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="hidden items-center gap-1.5 rounded-full border border-border px-2.5 py-1.5 text-[10px] text-muted-foreground sm:flex">
          <ShieldCheck size={12} className="text-emerald-500" /> No model API
        </div>
        <Button
          variant="ghost"
          size="icon"
          aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
          onClick={() => setTheme(dark ? "light" : "dark")}
        >
          {dark ? <Sun size={17} /> : <Moon size={17} />}
        </Button>
      </div>
    </header>
  );
}

function useHealth() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const response = await fetch("/api/chat");
        if (!response.ok) throw new Error("offline");
        const body = await response.json() as { status?: string };
        if (body.status !== "ok") throw new Error("offline");
        if (!cancelled) setStatus("online");
      } catch {
        if (cancelled) return;
        setStatus("offline");
        toast.error("EvolveAI is unreachable.", {
          id: "evolveai-health",
          action: { label: "Retry", onClick: () => { void check(); } },
        });
      }
    }
    void check();
    return () => { cancelled = true; };
  }, []);
  return status;
}

function BranchMemory() {
  const aui = useAui();
  const tailId = useAuiState((state) => state.thread.messages.at(-1)?.id ?? null);
  useEffect(() => {
    if (!tailId) return;
    const item = aui.optional.threadListItem;
    if (!item) return;
    const remoteId = item.getState().remoteId;
    if (remoteId) rememberBranchHead(remoteId, tailId);
  }, [aui, tailId]);
  return null;
}

function AppSkeleton() {
  return (
    <div className="flex h-dvh bg-background">
      <div className="hidden w-[280px] border-r border-border p-4 md:block">
        <Skeleton className="mb-4 h-8 w-32" />
        <Skeleton className="mb-2 h-10 w-full" />
        <Skeleton className="h-9 w-full" />
      </div>
      <div className="flex flex-1 flex-col">
        <div className="flex h-16 items-center border-b border-border px-4"><Skeleton className="h-6 w-28" /></div>
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-end gap-3 p-6">
          <Skeleton className="h-16 w-2/3 self-end" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-14 w-full rounded-3xl" />
        </div>
      </div>
    </div>
  );
}
