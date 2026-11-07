"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ActionBarPrimitive,
  AuiIf,
  AttachmentPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
  type Attachment,
  type ReasoningMessagePartProps,
  type TextMessagePartProps,
} from "@assistant-ui/react";
import {
  ArrowDown, ArrowUp, BrainCircuit, Check, ChevronLeft, ChevronRight, Copy,
  FileText, Globe2, MessageSquare, Paperclip, Pencil, RefreshCw, Square, ThumbsDown, ThumbsUp, X,
} from "lucide-react";
import type { ReactElement } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { MarkdownText } from "@/components/chat/markdown-text";
import { registerRetry } from "@/lib/evolve-runtime";

const starters = [
  { icon: BrainCircuit, title: "Get to know EvolveAI", prompt: "What can you do?" },
  { icon: MessageSquare, title: "Remember something", prompt: "Remember that my preferred response style is concise and practical." },
  { icon: Globe2, title: "Research a question", prompt: "What is the difference between memory and learning in a rule-based assistant?" },
  { icon: FileText, title: "Work through a task", prompt: "Help me break a complex task into clear, actionable steps." },
];

const threadLoading = (state: { thread: { isLoading: boolean } }) => state.thread.isLoading;
const threadEmpty = (state: { thread: { isEmpty: boolean; isLoading: boolean } }) => state.thread.isEmpty && !state.thread.isLoading;

export function ThreadView() {
  const aui = useAui();
  useEffect(() => registerRetry(() => {
    const messages = aui.thread.getState().messages;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        aui.thread.message({ id: messages[index].id }).reload();
        return;
      }
    }
  }), [aui]);

  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
      <ThreadPrimitive.Viewport className="relative min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-6 md:px-8">
          <AuiIf condition={threadLoading}>
            <LoadingSkeleton />
          </AuiIf>
          <AuiIf condition={threadEmpty}>
            <EmptyState />
          </AuiIf>
          <div className="space-y-6">
            <ThreadPrimitive.Messages>
              {({ message }) => message.role === "user" ? <UserMessage /> : <AssistantMessage />}
            </ThreadPrimitive.Messages>
          </div>
        </div>
        <ThreadPrimitive.ViewportFooter className="sticky bottom-3 z-10 flex justify-center">
          <ThreadPrimitive.ScrollToBottom
            aria-label="Jump to latest"
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-popover px-3 py-1.5 text-xs text-popover-foreground shadow-lg disabled:hidden"
          >
            <ArrowDown size={14} /> Latest
          </ThreadPrimitive.ScrollToBottom>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
      <Composer />
    </ThreadPrimitive.Root>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4 py-6" aria-hidden>
      <div className="ml-auto h-12 w-2/3 animate-pulse rounded-2xl bg-muted" />
      <div className="h-24 w-full animate-pulse rounded-2xl bg-muted" />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="animate-message-in mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center pb-8">
      <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/30 bg-accent text-primary shadow-[0_0_60px_color-mix(in_srgb,var(--primary)_18%,transparent)]">
        <BrainCircuit size={28} strokeWidth={1.6} />
      </div>
      <p className="text-center text-[11px] font-medium uppercase tracking-[0.22em] text-primary">A system that learns from experience</p>
      <h1 className="mt-3 text-center font-display text-4xl font-semibold tracking-[-0.045em] text-foreground sm:text-5xl">
        Think clearly. <span className="text-muted-foreground">Evolve continuously.</span>
      </h1>
      <p className="mx-auto mt-4 max-w-lg text-center text-sm leading-6 text-muted-foreground">
        A personal assistant built on transparent rules, persistent memory, public research, and your feedback.
      </p>
      <div className="mt-8 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {starters.map(({ icon: Icon, title, prompt }) => (
          <ThreadPrimitive.Suggestion
            key={title}
            prompt={prompt}
            send
            className="group flex min-h-24 items-start gap-3 rounded-2xl border border-border bg-card p-4 text-left transition hover:border-primary/40 hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span className="mt-0.5 rounded-lg border border-border bg-muted p-2 text-muted-foreground group-hover:text-primary"><Icon size={16} /></span>
            <span>
              <span className="block text-[13px] font-medium text-foreground">{title}</span>
              <span className="mt-1.5 block text-xs leading-5 text-muted-foreground">{prompt}</span>
            </span>
          </ThreadPrimitive.Suggestion>
        ))}
      </div>
    </div>
  );
}

function UserMessage() {
  const editing = useAuiState((state) => state.message.composer.isEditing);
  if (editing) {
    return (
      <MessagePrimitive.Root className="animate-message-in ml-auto w-full max-w-[85%]">
        <ComposerPrimitive.Root className="rounded-2xl border border-primary/40 bg-card p-2">
          <ComposerPrimitive.Input className="max-h-40 w-full resize-none bg-transparent px-2 py-2 text-sm outline-none" aria-label="Edit message" autoFocus />
          <div className="flex justify-end gap-2 px-1 pb-1">
            <ComposerPrimitive.Cancel className="rounded-lg px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted">Cancel</ComposerPrimitive.Cancel>
            <ComposerPrimitive.Send className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-40">Save</ComposerPrimitive.Send>
          </div>
        </ComposerPrimitive.Root>
      </MessagePrimitive.Root>
    );
  }
  return (
    <MessagePrimitive.Root className="animate-message-in ml-auto flex w-full max-w-[85%] flex-col items-end gap-1.5">
      <MessagePrimitive.Attachments>{({ attachment }) => <AttachmentCard attachment={attachment} />}</MessagePrimitive.Attachments>
      <div className="rounded-2xl rounded-br-md border border-border bg-accent px-4 py-3 text-sm leading-6 text-accent-foreground">
        <MessagePrimitive.Parts components={{ Text: UserText }} />
      </div>
      <div className="flex items-center gap-1">
        <ActionBarPrimitive.Root className="flex items-center gap-0.5">
          <Tip label="Edit">
            <ActionBarPrimitive.Edit className={iconClass} aria-label="Edit message"><Pencil size={14} /></ActionBarPrimitive.Edit>
          </Tip>
          <Tip label="Copy">
            <ActionBarPrimitive.Copy className={iconClass} aria-label="Copy message"><CopyOrCheck /></ActionBarPrimitive.Copy>
          </Tip>
        </ActionBarPrimitive.Root>
        <BranchControls />
      </div>
    </MessagePrimitive.Root>
  );
}

function UserText({ text }: TextMessagePartProps) {
  return <span className="whitespace-pre-wrap">{text}</span>;
}

function AssistantMessage() {
  const mode = useAuiState((state) => {
    const value = state.message.metadata.custom.mode;
    return typeof value === "string" ? value.replaceAll("_", " ") : "";
  });
  const feedback = useAuiState((state) => state.message.metadata.submittedFeedback?.type);
  return (
    <MessagePrimitive.Root className="animate-message-in flex gap-3">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-primary/25 bg-accent text-primary">
        <BrainCircuit size={16} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="font-display text-sm font-semibold">EvolveAI</span>
          {mode && <span className="rounded-full border border-border px-2 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">{mode}</span>}
        </div>
        <MessagePrimitive.Parts components={{ Text: MarkdownText, Reasoning: ReasoningBlock, Empty: Thinking }} />
        <MessageError />
        <div className="mt-2 flex flex-wrap items-center gap-1">
          <ActionBarPrimitive.Root className="flex items-center gap-0.5">
            <Tip label="Copy">
              <ActionBarPrimitive.Copy className={iconClass} aria-label="Copy response"><CopyOrCheck /></ActionBarPrimitive.Copy>
            </Tip>
            <Tip label="Regenerate">
              <ActionBarPrimitive.Reload className={iconClass} aria-label="Regenerate response"><RefreshCw size={14} /></ActionBarPrimitive.Reload>
            </Tip>
            <Tip label="Helpful">
              <ActionBarPrimitive.FeedbackPositive disabled={!!feedback} className={iconClass} data-on={feedback === "positive" || undefined} aria-label="Helpful response">
                <ThumbsUp size={14} />
              </ActionBarPrimitive.FeedbackPositive>
            </Tip>
            <Tip label="Not helpful">
              <ActionBarPrimitive.FeedbackNegative disabled={!!feedback} className={iconClass} data-on={feedback === "negative" || undefined} aria-label="Unhelpful response">
                <ThumbsDown size={14} />
              </ActionBarPrimitive.FeedbackNegative>
            </Tip>
          </ActionBarPrimitive.Root>
          {feedback && <span className="text-[11px] text-muted-foreground">Feedback saved</span>}
          <BranchControls />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

function MessageError() {
  const text = useAuiState((state) => {
    const status = state.message.status;
    if (!status || status.type !== "incomplete" || status.reason !== "error") return "";
    const error = "error" in status ? status.error : undefined;
    if (typeof error === "string") return error;
    if (error && typeof error === "object" && "message" in error && typeof error.message === "string") return error.message;
    return "EvolveAI could not finish that response.";
  });
  if (!text) return null;
  return (
    <ErrorPrimitive.Root className="mt-2 flex flex-wrap items-center gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
      <ErrorPrimitive.Message />
      <ActionBarPrimitive.Reload className="rounded-lg border border-destructive/30 px-2 py-1 text-xs">Retry</ActionBarPrimitive.Reload>
    </ErrorPrimitive.Root>
  );
}

function ReasoningBlock({ text, status }: ReasoningMessagePartProps) {
  const running = status.type === "running";
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const open = userOpen ?? running;
  return (
    <div className="mb-3 rounded-xl border border-border bg-muted/50">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-expanded={open}
        onClick={() => setUserOpen(!open)}
      >
        <ChevronRight size={14} className={`transition-transform ${open ? "rotate-90" : ""}`} />
        {running ? <ThinkingLabel /> : <span>Reasoning</span>}
      </button>
      {open && <pre className="whitespace-pre-wrap px-3 pb-3 font-sans text-xs leading-5 text-muted-foreground">{text}</pre>}
    </div>
  );
}

function Thinking() {
  return (
    <div className="flex items-center gap-2 py-1" aria-live="polite">
      <ThinkingLabel />
    </div>
  );
}

function ThinkingLabel() {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="flex gap-1" aria-hidden>
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-primary" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-primary [animation-delay:160ms]" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-primary [animation-delay:320ms]" />
      </span>
      <span className="shimmer-text text-xs font-medium">Thinking</span>
    </span>
  );
}

function BranchControls() {
  return (
    <BranchPickerPrimitive.Root hideWhenSingleBranch className="ml-1 inline-flex items-center gap-0.5 text-xs text-muted-foreground">
      <BranchPickerPrimitive.Previous className={iconClass} aria-label="Previous branch"><ChevronLeft size={14} /></BranchPickerPrimitive.Previous>
      <span className="min-w-8 text-center tabular-nums"><BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count /></span>
      <BranchPickerPrimitive.Next className={iconClass} aria-label="Next branch"><ChevronRight size={14} /></BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
}

function CopyOrCheck() {
  const copied = useAuiState((state) => state.message.isCopied);
  return copied ? <Check size={14} /> : <Copy size={14} />;
}

function Composer() {
  return (
    <div className="shrink-0 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-1 md:px-6">
      <ComposerPrimitive.AttachmentDropzone className="mx-auto block max-w-3xl rounded-[22px] data-[dragging]:bg-accent/40">
        <ComposerPrimitive.Root className="rounded-[22px] border border-border bg-card p-2 shadow-[0_14px_40px_rgba(0,0,0,0.06)] transition focus-within:border-primary/50 dark:shadow-[0_14px_50px_rgba(0,0,0,0.28)]">
          <ComposerPrimitive.Attachments>
            {({ attachment }) => <AttachmentCard attachment={attachment} removable />}
          </ComposerPrimitive.Attachments>
          <ComposerPrimitive.Input
            placeholder="Message EvolveAI…"
            aria-label="Message EvolveAI"
            autoFocus
            minRows={1}
            maxRows={8}
            className="max-h-48 w-full resize-none bg-transparent px-3 py-3 text-sm outline-none placeholder:text-muted-foreground"
          />
          <div className="flex items-center justify-between px-1 pb-0.5">
            <div className="flex items-center gap-1 pl-1">
              <ComposerPrimitive.AddAttachment className={iconClass} aria-label="Attach files" multiple>
                <Paperclip size={16} />
              </ComposerPrimitive.AddAttachment>
              <span className="hidden text-[11px] text-muted-foreground sm:inline">Rule-based · memory-enabled</span>
            </div>
            <div className="flex items-center gap-1.5">
              <AuiIf condition={(state) => state.composer.canCancel}>
                <ComposerPrimitive.Cancel className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-foreground text-background" aria-label="Stop generating">
                  <Square size={14} fill="currentColor" />
                </ComposerPrimitive.Cancel>
              </AuiIf>
              <AuiIf condition={(state) => !state.composer.canCancel}>
                <ComposerPrimitive.Send className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground disabled:opacity-40" aria-label="Send message">
                  <ArrowUp size={17} />
                </ComposerPrimitive.Send>
              </AuiIf>
            </div>
          </div>
        </ComposerPrimitive.Root>
      </ComposerPrimitive.AttachmentDropzone>
      <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground">EvolveAI can make mistakes. Research is heuristic and should be checked against sources.</p>
    </div>
  );
}

function AttachmentCard({ attachment, removable = false }: { attachment: Attachment; removable?: boolean }) {
  const preview = usePreview(attachment);
  return (
    <AttachmentPrimitive.Root className="mb-2 mr-2 inline-flex max-w-full items-center gap-2 rounded-xl border border-border bg-muted/70 py-1 pl-1 pr-2 text-xs">
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="" className="h-10 w-10 rounded-lg object-cover" />
      ) : (
        <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-background text-muted-foreground"><FileText size={16} /></span>
      )}
      <span className="max-w-40 truncate"><AttachmentPrimitive.Name /></span>
      {removable && attachment.status.type !== "complete" && (
        <AttachmentPrimitive.Remove className="rounded-md p-1 text-muted-foreground hover:bg-background hover:text-foreground" aria-label={`Remove ${attachment.name}`}>
          <X size={13} />
        </AttachmentPrimitive.Remove>
      )}
    </AttachmentPrimitive.Root>
  );
}

function usePreview(attachment: Attachment) {
  const file = attachment.file?.type.startsWith("image/") ? attachment.file : undefined;
  const objectUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    if (!objectUrl) return undefined;
    return () => URL.revokeObjectURL(objectUrl);
  }, [objectUrl]);
  if (objectUrl) return objectUrl;
  const image = attachment.content?.find((part) => part.type === "image");
  return image && image.type === "image" ? image.image : null;
}

const iconClass = "inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-30 data-[on=true]:text-primary";

function Tip({ label, children }: { label: string; children: ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
