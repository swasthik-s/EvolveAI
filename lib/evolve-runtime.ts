"use client";

import { useMemo } from "react";
import { toast } from "sonner";
import {
  createLocalStorageAdapter,
  type AsyncStorageLike,
} from "@assistant-ui/core/react";
import type {
  AttachmentAdapter,
  ChatModelAdapter,
  CompleteAttachment,
  FeedbackAdapter,
  PendingAttachment,
  RemoteThreadListAdapter,
  ThreadMessage,
  ThreadUserMessagePart,
} from "@assistant-ui/react";

export const THREAD_PREFIX = "evolveai.threads.v2:";
const LEGACY_KEY = "evolveai.threads.v1";
const LEGACY_FLAG = "evolveai.threads.v1.migrated";
const MAX_FILE_BYTES = 1_500_000;
const MAX_PAYLOAD = 80_000;

type LegacyMessage = {
  id?: string;
  role?: string;
  content?: string;
  taskId?: string;
  mode?: string;
  feedback?: "up" | "down";
};

type LegacyThread = {
  id?: string;
  title?: string;
  updatedAt?: number;
  messages?: LegacyMessage[];
};

let retryLast: (() => void) | null = null;

export function registerRetry(handler: () => void) {
  retryLast = handler;
  return () => {
    if (retryLast === handler) retryLast = null;
  };
}

const browserStorage: AsyncStorageLike = {
  async getItem(key) {
    return localStorage.getItem(key);
  },
  async setItem(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      try {
        localStorage.setItem(key, value.replace(/data:[^"]{80000,}/g, ""));
      } catch {
        console.warn("EvolveAI could not save this conversation in local storage.");
      }
    }
  },
  async removeItem(key) {
    localStorage.removeItem(key);
  },
};

function textOf(message: ThreadMessage) {
  return message.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

function legacyToRepository(thread: LegacyThread) {
  const messages = Array.isArray(thread.messages) ? thread.messages : [];
  const createdAt = new Date(thread.updatedAt || Date.now()).toISOString();
  let previousId: string | null = null;
  const items: object[] = [];
  for (const message of messages) {
    const content = typeof message.content === "string" ? message.content : "";
    if (!content || (message.role !== "user" && message.role !== "assistant")) continue;
    const id = message.id || crypto.randomUUID();
    const parentId = previousId;
    previousId = id;
    items.push(message.role === "user" ? {
      parentId,
      message: {
        id,
        role: "user",
        createdAt,
        content: [{ type: "text", text: content }],
        attachments: [],
        metadata: { custom: {} },
      },
    } : {
      parentId,
      message: {
        id,
        role: "assistant",
        createdAt,
        content: [{ type: "text", text: content }],
        status: { type: "complete", reason: "stop" },
        metadata: {
          custom: {
            ...(message.taskId ? { taskId: message.taskId } : {}),
            ...(message.mode ? { mode: message.mode } : {}),
          },
          unstable_state: null,
          unstable_annotations: [],
          unstable_data: [],
          steps: [],
          ...(message.feedback === "up"
            ? { submittedFeedback: { type: "positive" } }
            : message.feedback === "down"
              ? { submittedFeedback: { type: "negative" } }
              : {}),
        },
      },
    });
  }
  return { headId: previousId, messages: items };
}

/** Move the previous single-array thread store into assistant-ui's local records once. */
export function migrateLegacyThreads() {
  if (typeof window === "undefined") return;
  try {
    if (localStorage.getItem(LEGACY_FLAG) === "1") return;
    localStorage.setItem(LEGACY_FLAG, "1");
    const threadsKey = `${THREAD_PREFIX}threads`;
    if (localStorage.getItem(threadsKey)) return;
    const raw = localStorage.getItem(LEGACY_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return;
    const metadata = [];
    for (const entry of parsed as LegacyThread[]) {
      if (!entry || typeof entry !== "object") continue;
      const remoteId = entry.id || crypto.randomUUID();
      metadata.push({
        remoteId,
        status: "regular",
        title: (entry.title || "New conversation").slice(0, 80),
      });
      const repository = legacyToRepository({ ...entry, id: remoteId });
      localStorage.setItem(`${THREAD_PREFIX}messages:${remoteId}`, JSON.stringify(repository));
    }
    if (metadata.length) localStorage.setItem(threadsKey, JSON.stringify(metadata));
  } catch {
    /* A bad local record should not block a new conversation. */
  }
}

export function latestRemoteId() {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = localStorage.getItem(`${THREAD_PREFIX}threads`);
    const parsed = raw ? JSON.parse(raw) as { remoteId?: string; status?: string }[] : [];
    return parsed.find((thread) => thread.status !== "archived")?.remoteId;
  } catch {
    return undefined;
  }
}

export function rememberBranchHead(remoteId: string, headId: string) {
  const key = `${THREAD_PREFIX}messages:${remoteId}`;
  const raw = localStorage.getItem(key);
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw) as { headId?: string | null; messages?: { message?: { id?: string } }[] };
    if (!Array.isArray(parsed.messages) || parsed.headId === headId) return;
    if (!parsed.messages.some((item) => item?.message?.id === headId)) return;
    parsed.headId = headId;
    localStorage.setItem(key, JSON.stringify(parsed));
  } catch {
    /* The next completed message rewrites the record. */
  }
}

function titleFrom(messages: readonly ThreadMessage[]) {
  const first = messages.find((message) => message.role === "user");
  const text = first ? textOf(first).replace(/\s+/g, " ").trim() : "";
  return text ? text.slice(0, 42) : "New conversation";
}

export function createEvolveThreadListAdapter(): RemoteThreadListAdapter {
  const base = createLocalStorageAdapter({
    storage: browserStorage,
    prefix: THREAD_PREFIX,
    titleGenerator: { async generateTitle(messages) { return titleFrom(messages); } },
  });
  const useBaseAdapters = base.unstable_useAdapters;
  if (!useBaseAdapters) return base;
  return {
    ...base,
    unstable_useAdapters: function useEvolveAdapters() {
      const historyAdapters = useBaseAdapters();
      const attachments = useMemo(() => createAttachmentAdapter(), []);
      const feedback = useMemo(() => createFeedbackAdapter(), []);
      return useMemo(
        () => ({ ...historyAdapters, attachments, feedback }),
        [historyAdapters, attachments, feedback],
      );
    },
  };
}

const TEXT_EXT = /\.(txt|md|markdown|csv|json|py|ts|tsx|js|jsx|mjs|css|html|xml|yml|yaml|toml|rs|go|java|rb|sh|c|h|cpp|sql)$/i;

function isTextFile(file: File) {
  return file.type.startsWith("text/") || file.type === "application/json" || TEXT_EXT.test(file.name);
}

async function attachmentContent(file: File, kind: string): Promise<ThreadUserMessagePart[]> {
  if (kind === "image") {
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error ?? new Error("Could not read the image."));
      reader.readAsDataURL(file);
    });
    if (dataUrl.length <= MAX_PAYLOAD) return [{ type: "image", image: dataUrl, filename: file.name }];
    return [{ type: "text", text: "" }];
  }
  if (kind === "document") {
    const text = (await file.text()).slice(0, MAX_PAYLOAD);
    return [{ type: "text", text }];
  }
  return [{ type: "text", text: "" }];
}

export function createAttachmentAdapter(): AttachmentAdapter {
  return {
    accept: "image/*,text/*,.md,.json,.csv,.py,.ts,.tsx,.js,.jsx,.css,.html,.xml,.yml,.yaml,.txt,.pdf,.toml,.sql",
    async add({ file }) {
      if (file.size > MAX_FILE_BYTES) {
        const message = `${file.name || "That file"} is larger than 1.5 MB.`;
        toast.error(message);
        throw new Error(message);
      }
      const type = file.type.startsWith("image/") ? "image" : isTextFile(file) ? "document" : "file";
      const pending: PendingAttachment = {
        id: crypto.randomUUID(),
        type,
        name: file.name || "attachment",
        contentType: file.type || "application/octet-stream",
        file,
        status: { type: "requires-action", reason: "composer-send" },
      };
      return pending;
    },
    async send(attachment) {
      const content = attachment.file
        ? await attachmentContent(attachment.file, attachment.type)
        : [{ type: "text" as const, text: "" }];
      const complete: CompleteAttachment = { ...attachment, content, status: { type: "complete" } };
      return complete;
    },
    async remove() {},
  };
}

export function createFeedbackAdapter(): FeedbackAdapter {
  return {
    async submit({ message, type }) {
      const taskId = message.metadata.custom.taskId;
      if (typeof taskId !== "string" || !taskId) return;
      try {
        await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "feedback",
            task_id: taskId,
            score: type === "positive" ? 1 : 0,
            feedback: type === "positive" ? "Helpful" : "Not helpful",
          }),
        });
      } catch {
        /* Feedback is best-effort and should not interrupt the conversation. */
      }
    },
  };
}

type Plan = {
  intent?: string;
  follow_up?: boolean;
  speech_act?: string;
  communication_action?: string;
  communication_reason?: string;
  steps?: unknown[];
};

type ChatPayload = {
  answer?: string;
  task_id?: string;
  mode?: string;
  reasoning_plan?: Plan;
  detail?: string;
};

function formatReasoning(stages: string[], plan: Plan | null) {
  const lines = [...stages];
  if (plan) {
    const facts = [
      plan.intent ? `Intent: ${String(plan.intent).replaceAll("_", " ")}` : "",
      typeof plan.follow_up === "boolean" ? `Follow-up: ${plan.follow_up ? "yes" : "no"}` : "",
      plan.speech_act ? `Speech act: ${plan.speech_act}` : "",
      plan.communication_action
        ? `Communication: ${String(plan.communication_action).replaceAll("_", " ")}${plan.communication_reason ? ` — ${plan.communication_reason}` : ""}`
        : "",
    ].filter(Boolean);
    if (facts.length) {
      if (lines.length) lines.push("");
      lines.push(...facts);
    }
    if (Array.isArray(plan.steps) && plan.steps.length) {
      lines.push("", "Steps:");
      for (const step of plan.steps) lines.push(`- ${String(step).replaceAll("_", " ")}`);
    }
  }
  return lines.join("\n").trim();
}

function contentParts(reasoning: string, text: string) {
  return [
    ...(reasoning ? [{ type: "reasoning" as const, text: reasoning }] : []),
    ...(text ? [{ type: "text" as const, text }] : []),
  ];
}

function apiAttachments(message: ThreadMessage) {
  if (message.role !== "user") return [];
  return message.attachments.slice(0, 4).map((attachment) => {
    const name = attachment.name || "file";
    const type = attachment.contentType || "application/octet-stream";
    const size = attachment.file?.size ?? 0;
    const image = attachment.content.find((part) => part.type === "image");
    if (image && image.type === "image" && image.image.startsWith("data:") && image.image.length <= MAX_PAYLOAD) {
      return { name, type, size: size || image.image.length, encoding: "base64" as const, content: image.image };
    }
    if (attachment.type === "image" || type.startsWith("image/")) {
      return { name, type, size, encoding: "omitted" as const, content: "" };
    }
    const text = attachment.content
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("\n")
      .slice(0, MAX_PAYLOAD);
    if (!text) return { name, type, size, encoding: "omitted" as const, content: "" };
    return { name, type: type || "text/plain", size: size || text.length, encoding: "text" as const, content: text };
  });
}

async function* readSse(response: Response, signal: AbortSignal) {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      yield { event, data: JSON.parse(dataLines.join("\n")) as ChatPayload & { stage?: string; text?: string; detail?: string } };
    }
  }
}

function isAbort(error: unknown) {
  return (error instanceof DOMException || error instanceof Error) && error.name === "AbortError";
}

function reportFailure(error: unknown) {
  const message = error instanceof Error ? error.message : "Unable to reach EvolveAI.";
  toast.error(message, {
    id: "evolveai-chat-error",
    action: { label: "Retry", onClick: () => retryLast?.() },
  });
}

export const evolveModelAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const current = [...messages].reverse().find((message) => message.role === "user") ?? messages.at(-1);
    if (!current || current.role !== "user") throw new Error("There is no message to send.");
    const index = messages.findIndex((message) => message.id === current.id);
    const history = (index >= 0 ? messages.slice(0, index) : [])
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-12)
      .flatMap((message) => {
        const content = textOf(message).slice(0, 8000);
        if (!content || (message.role !== "user" && message.role !== "assistant")) return [];
        return [{ role: message.role, content }];
      });

    let response: Response;
    try {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream, application/json" },
        body: JSON.stringify({
          message: textOf(current),
          history,
          stream: true,
          attachments: apiAttachments(current),
        }),
        signal: abortSignal,
      });
    } catch (error) {
      if (isAbort(error) || abortSignal.aborted) return;
      reportFailure(error);
      throw error;
    }

    if (!response.ok) {
      let detail = "The assistant could not process that request.";
      try {
        const payload = await response.json() as ChatPayload;
        detail = String(payload.detail || detail);
      } catch { /* The error body is not JSON. */ }
      const error = new Error(detail);
      reportFailure(error);
      throw error;
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/event-stream")) {
      const payload = await response.json() as ChatPayload;
      const text = String(payload.answer ?? "");
      const reasoning = formatReasoning([], payload.reasoning_plan ?? null);
      yield {
        content: contentParts(reasoning, text),
        metadata: { custom: { taskId: payload.task_id ?? "", mode: payload.mode ?? "rules" } },
      };
      return;
    }

    let text = "";
    const stages: string[] = [];
    let plan: Plan | null = null;
    let taskId = "";
    let mode = "rules";
    try {
      for await (const evt of readSse(response, abortSignal)) {
        if (abortSignal.aborted) return;
        if (evt.event === "status" && evt.data.stage) {
          stages.push(evt.data.stage);
          const reasoning = formatReasoning(stages, plan);
          if (reasoning) yield { content: contentParts(reasoning, text) };
        } else if (evt.event === "delta" && evt.data.text) {
          text += evt.data.text;
          yield { content: contentParts(formatReasoning(stages, plan), text) };
        } else if (evt.event === "error") {
          throw new Error(evt.data.detail || "EvolveAI could not finish that response.");
        } else if (evt.event === "done") {
          text = typeof evt.data.answer === "string" ? evt.data.answer : text;
          plan = evt.data.reasoning_plan ?? plan;
          taskId = evt.data.task_id ?? taskId;
          mode = evt.data.mode ?? mode;
          yield {
            content: contentParts(formatReasoning(stages, plan), text),
            metadata: { custom: { taskId, mode } },
          };
        }
      }
    } catch (error) {
      if (isAbort(error) || abortSignal.aborted) return;
      reportFailure(error);
      throw error;
    }
  },
};
