"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ArrowDown, ArrowUp, BrainCircuit, Check, ChevronDown, CircleHelp, Command,
  FileText, Globe2, Menu, MessageSquare, MoreHorizontal, PanelLeftClose,
  Plus, Search, Settings2, ShieldCheck, Sparkles, ThumbsDown, ThumbsUp, X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  taskId?: string;
  mode?: string;
  feedback?: "up" | "down";
};
type Thread = { id: string; title: string; messages: Message[]; updatedAt: number };

const starters = [
  { icon: BrainCircuit, title: "Get to know EvolveAI", prompt: "What can you do?" },
  { icon: MessageSquare, title: "Remember something", prompt: "Remember that my preferred response style is concise and practical." },
  { icon: Globe2, title: "Research a question", prompt: "What is the difference between memory and learning in a rule-based assistant?" },
  { icon: FileText, title: "Work through a task", prompt: "Help me break a complex task into clear, actionable steps." },
];

function makeId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function Home() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeId, setActiveId] = useState("");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [apiError, setApiError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("evolveai.threads.v1");
      if (saved) {
        const parsed = JSON.parse(saved) as Thread[];
        if (Array.isArray(parsed) && parsed.length) {
          setThreads(parsed);
          setActiveId(parsed[0].id);
          return;
        }
      }
    } catch { /* Ignore invalid local browser state and start clean. */ }
    const initial = { id: makeId(), title: "New conversation", messages: [], updatedAt: Date.now() };
    setThreads([initial]);
    setActiveId(initial.id);
  }, []);

  useEffect(() => {
    if (threads.length) {
      try { localStorage.setItem("evolveai.threads.v1", JSON.stringify(threads)); } catch { /* Storage may be disabled. */ }
    }
  }, [threads]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [activeId, threads, sending]);

  const activeThread = threads.find((thread) => thread.id === activeId);
  const messages = activeThread?.messages ?? [];

  function updateThread(threadId: string, updater: (thread: Thread) => Thread) {
    setThreads((current) => current.map((thread) => thread.id === threadId ? updater(thread) : thread));
  }

  function newChat() {
    const thread = { id: makeId(), title: "New conversation", messages: [], updatedAt: Date.now() };
    setThreads((current) => [thread, ...current]);
    setActiveId(thread.id);
    setInput("");
    setApiError("");
    window.setTimeout(() => inputRef.current?.focus(), 20);
  }

  async function sendMessage(raw?: string) {
    const content = (raw ?? input).trim();
    if (!content || sending || !activeThread) return;
    const threadId = activeThread.id;
    const userMessage: Message = { id: makeId(), role: "user", content };
    const history = [...activeThread.messages, userMessage];
    updateThread(threadId, (thread) => ({
      ...thread,
      title: thread.messages.length ? thread.title : content.slice(0, 42),
      messages: history,
      updatedAt: Date.now(),
    }));
    setInput("");
    setSending(true);
    setApiError("");
    setStatus("Understanding your message");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          history: history.slice(0, -1).slice(-12).map(({ role, content: text }) => ({ role, content: text })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || payload?.error || "The assistant could not process that request.");
      const answer = String(payload.answer ?? "I could not produce a response.");
      const assistantMessage: Message = {
        id: makeId(),
        role: "assistant",
        content: answer,
        taskId: payload.task_id,
        mode: payload.mode,
      };
      updateThread(threadId, (thread) => ({
        ...thread,
        messages: [...thread.messages, assistantMessage],
        updatedAt: Date.now(),
      }));
      setStatus(payload.mode === "web_research" ? "Research complete" : "Ready");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to reach EvolveAI.";
      setApiError(message);
      updateThread(threadId, (thread) => ({
        ...thread,
        messages: [...thread.messages, {
          id: makeId(),
          role: "assistant",
          content: "I couldn’t reach the EvolveAI engine. Check the deployment’s Python API logs and try again. Your message is still in this conversation.",
        }],
        updatedAt: Date.now(),
      }));
      setStatus("Connection issue");
    } finally {
      setSending(false);
      window.setTimeout(() => inputRef.current?.focus(), 20);
    }
  }

  async function rateMessage(message: Message, rating: "up" | "down") {
    if (!message.taskId || message.feedback) return;
    updateThread(activeId, (thread) => ({
      ...thread,
      messages: thread.messages.map((item) => item.id === message.id ? { ...item, feedback: rating } : item),
    }));
    try {
      await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "feedback", task_id: message.taskId, score: rating === "up" ? 1 : 0, feedback: rating === "up" ? "Helpful" : "Not helpful" }),
      });
    } catch { /* Feedback is best-effort; do not interrupt the conversation. */ }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  function removeThread(id: string) {
    const remaining = threads.filter((thread) => thread.id !== id);
    if (!remaining.length) {
      const fresh = { id: makeId(), title: "New conversation", messages: [], updatedAt: Date.now() };
      setThreads([fresh]);
      setActiveId(fresh.id);
    } else {
      setThreads(remaining);
      if (activeId === id) setActiveId(remaining[0].id);
    }
  }

  return (
    <main className="flex h-[100dvh] overflow-hidden bg-[#09090b] text-zinc-100">
      {sidebarOpen && (
        <aside className="z-20 flex w-[272px] shrink-0 flex-col border-r border-white/[0.07] bg-[#0c0c0f] max-md:absolute max-md:inset-y-0 max-md:left-0 max-md:shadow-2xl">
          <div className="flex h-[68px] items-center justify-between px-4">
            <button onClick={() => setActiveId(threads[0]?.id ?? activeId)} className="flex items-center gap-2.5 rounded-lg text-left">
              <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-violet-300 text-zinc-950"><BrainCircuit size={19} /></span>
              <span className="text-[15px] font-semibold tracking-tight">Evolve<span className="text-violet-300">AI</span></span>
            </button>
            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)}><PanelLeftClose size={17} /></Button>
          </div>
          <div className="px-3 pb-3">
            <Button variant="secondary" className="w-full justify-start gap-2.5" onClick={newChat}><Plus size={16} /> New chat <span className="ml-auto text-[10px] text-zinc-500">⌘ K</span></Button>
          </div>
          <div className="px-4 pb-2 pt-3 text-[10px] font-semibold uppercase tracking-[.16em] text-zinc-600">Workspace</div>
          <div className="space-y-1 px-2">
            <button onClick={newChat} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-[13px] text-zinc-300 hover:bg-white/[0.05]"><MessageSquare size={15} className="text-zinc-500" /> Conversations</button>
            <button onClick={() => setSettingsOpen((value) => !value)} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-[13px] text-zinc-300 hover:bg-white/[0.05]"><Settings2 size={15} className="text-zinc-500" /> Settings <ChevronDown size={13} className={`ml-auto transition-transform ${settingsOpen ? "rotate-180" : ""}`} /></button>
            {settingsOpen && <div className="mx-3 rounded-lg border border-white/[0.07] bg-white/[0.025] p-3 text-xs leading-5 text-zinc-400">No hosted language model is used. EvolveAI follows explicit rules, recalls saved context, and can search public sources.</div>}
          </div>
          <div className="mt-7 flex items-center justify-between px-4 pb-2 text-[10px] font-semibold uppercase tracking-[.16em] text-zinc-600">
            <span>Recent chats</span><Search size={13} />
          </div>
          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2">
            {threads.map((thread) => (
              <div key={thread.id} className={`group flex items-center gap-1 rounded-lg ${thread.id === activeId ? "bg-white/[0.07]" : "hover:bg-white/[0.035]"}`}>
                <button onClick={() => setActiveId(thread.id)} className="min-w-0 flex-1 truncate px-3 py-2.5 text-left text-[12px] text-zinc-300">{thread.title}</button>
                <button onClick={() => removeThread(thread.id)} aria-label="Delete conversation" className="mr-1 hidden rounded-md p-1.5 text-zinc-600 hover:bg-white/10 hover:text-zinc-200 group-hover:block"><X size={13} /></button>
              </div>
            ))}
          </div>
          <div className="border-t border-white/[0.07] p-3">
            <div className="flex items-center gap-3 rounded-xl p-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-violet-300/20 bg-violet-300/10 text-violet-200"><Sparkles size={17} /></div>
              <div className="min-w-0 flex-1"><div className="text-xs font-medium">Personal assistant</div><div className="mt-0.5 text-[10px] text-zinc-500">Instruction-driven engine</div></div>
              <MoreHorizontal size={16} className="text-zinc-600" />
            </div>
          </div>
        </aside>
      )}

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[68px] shrink-0 items-center justify-between border-b border-white/[0.06] px-4 md:px-7">
          <div className="flex items-center gap-3">
            {!sidebarOpen && <Button variant="ghost" size="icon" aria-label="Open sidebar" onClick={() => setSidebarOpen(true)}><Menu size={18} /></Button>}
            <div><div className="text-sm font-semibold tracking-tight">EvolveAI</div><div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-zinc-500"><span className={`h-1.5 w-1.5 rounded-full ${apiError ? "bg-amber-400" : "bg-emerald-400"}`} /> {status}</div></div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-1.5 rounded-full border border-white/[0.07] px-2.5 py-1.5 text-[10px] text-zinc-400 sm:flex"><ShieldCheck size={12} className="text-emerald-300" /> No model API</div>
            <Button variant="ghost" size="icon" aria-label="Help" onClick={() => setSettingsOpen((value) => !value)}><CircleHelp size={17} /></Button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className={`mx-auto flex min-h-full w-full max-w-[850px] flex-col px-4 pb-8 md:px-8 ${messages.length ? "pt-8" : "justify-center pt-10"}`}>
            {messages.length === 0 ? (
              <div className="animate-enter mx-auto w-full max-w-[660px] pb-10">
                <div className="mx-auto mb-7 flex h-[58px] w-[58px] items-center justify-center rounded-[20px] border border-violet-300/20 bg-violet-300/[0.09] text-violet-200 shadow-[0_0_70px_rgba(167,139,250,.09)]"><BrainCircuit size={29} strokeWidth={1.6} /></div>
                <div className="text-center">
                  <div className="mb-3 text-[11px] font-medium uppercase tracking-[.22em] text-violet-200/80">A system that learns from experience</div>
                  <h1 className="text-3xl font-semibold tracking-[-.045em] text-zinc-50 sm:text-[42px]">Think clearly. <span className="text-zinc-500">Evolve continuously.</span></h1>
                  <p className="mx-auto mt-4 max-w-[500px] text-sm leading-6 text-zinc-400">A personal assistant built on transparent rules, persistent memory, public research, and your feedback — not a hosted generative model.</p>
                </div>
                <div className="mt-9 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {starters.map(({ icon: Icon, title, prompt }) => (
                    <button key={title} onClick={() => void sendMessage(prompt)} className="group flex min-h-[94px] items-start gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.018] p-4 text-left transition hover:border-violet-300/25 hover:bg-violet-300/[0.035]">
                      <span className="mt-0.5 rounded-lg border border-white/[0.08] bg-white/[0.035] p-2 text-zinc-400 group-hover:text-violet-200"><Icon size={16} /></span>
                      <span><span className="block text-[13px] font-medium text-zinc-200">{title}</span><span className="mt-1.5 block text-xs leading-5 text-zinc-500">{prompt}</span></span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mx-auto w-full max-w-[720px] space-y-7">
                {messages.map((message) => (
                  <article key={message.id} className="animate-enter">
                    {message.role === "user" ? (
                      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-md border border-white/[0.07] bg-[#1b1b22] px-4 py-3 text-[13px] leading-6 text-zinc-100">{message.content}</div>
                    ) : (
                      <div className="flex gap-3.5">
                        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-violet-300/15 bg-violet-300/[0.08] text-violet-200"><BrainCircuit size={17} /></div>
                        <div className="min-w-0 flex-1 pt-1">
                          <div className="mb-2 flex items-center gap-2"><span className="text-xs font-semibold text-zinc-200">EvolveAI</span>{message.mode && <span className="rounded-full border border-white/[0.08] px-2 py-0.5 text-[9px] text-zinc-500">{message.mode.replaceAll("_", " ")}</span>}</div>
                          <div className="whitespace-pre-wrap break-words text-[13px] leading-7 text-zinc-300">{message.content.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g).map((part, index) => {
                            const bold = part.match(/^\*\*(.+)\*\*$/);
                            const italic = part.match(/^\*(.+)\*$/);
                            const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
                            if (bold) return <strong key={index} className="font-semibold text-zinc-100">{bold[1]}</strong>;
                            if (italic) return <em key={index} className="text-zinc-400">{italic[1]}</em>;
                            if (link) return <a key={index} href={link[2]} target="_blank" rel="noreferrer" className="text-violet-200 underline decoration-violet-300/30 underline-offset-4 hover:text-violet-100">{link[1]}</a>;
                            return <span key={index}>{part}</span>;
                          })}</div>
                          {message.taskId && <div className="mt-3 flex items-center gap-1 border-t border-white/[0.05] pt-2">
                            <span className="mr-2 text-[10px] text-zinc-600">Was this useful?</span>
                            <Button variant="ghost" size="icon" className={`h-7 w-7 ${message.feedback === "up" ? "text-emerald-300" : ""}`} aria-label="Helpful response" disabled={!!message.feedback} onClick={() => void rateMessage(message, "up")}><ThumbsUp size={13} /></Button>
                            <Button variant="ghost" size="icon" className={`h-7 w-7 ${message.feedback === "down" ? "text-rose-300" : ""}`} aria-label="Unhelpful response" disabled={!!message.feedback} onClick={() => void rateMessage(message, "down")}><ThumbsDown size={13} /></Button>
                            {message.feedback && <span className="ml-1 text-[10px] text-zinc-600">Feedback saved</span>}
                          </div>}
                        </div>
                      </div>
                    )}
                  </article>
                ))}
                {sending && <div className="flex items-center gap-3 text-xs text-zinc-500"><span className="flex gap-1"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-300" /><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-300 [animation-delay:120ms]" /><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-300 [animation-delay:240ms]" /></span>{status}…</div>}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </div>

        <div className="shrink-0 px-3 pb-[max(12px,env(safe-area-inset-bottom))] pt-2 md:px-6">
          <div className="mx-auto max-w-[760px]">
            {apiError && <div className="mb-2 flex items-center gap-2 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] px-3 py-2 text-xs text-amber-200/80"><CircleHelp size={14} /> {apiError}</div>}
            <form onSubmit={handleSubmit} className="rounded-[22px] border border-white/[0.12] bg-[#111115] p-2 shadow-[0_14px_50px_rgba(0,0,0,.22)] transition focus-within:border-violet-300/35">
              <Textarea ref={inputRef} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleKeyDown} placeholder="Message EvolveAI…" rows={1} disabled={sending} className="max-h-40 min-h-[48px] border-0 bg-transparent px-3 py-3 shadow-none focus-visible:ring-0" />
              <div className="flex items-center justify-between px-1 pb-0.5">
                <div className="flex items-center gap-1.5 pl-2 text-[10px] text-zinc-600"><Sparkles size={12} /> Rule-based · memory-enabled</div>
                <Button type="submit" size="icon" className="h-9 w-9 rounded-xl" disabled={sending || !input.trim()} aria-label="Send message">{sending ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-950/30 border-t-zinc-950" /> : <ArrowUp size={17} />}</Button>
              </div>
            </form>
            <p className="mt-2 text-center text-[10px] text-zinc-600">EvolveAI can make mistakes. Research is heuristic and should be checked against sources.</p>
          </div>
        </div>
      </section>
    </main>
  );
}
