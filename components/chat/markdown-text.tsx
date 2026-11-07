"use client";

import { useState, useSyncExternalStore, type ComponentProps } from "react";
import { useTheme } from "next-themes";
import { Check, Copy } from "lucide-react";
import { MarkdownTextPrimitive, type CodeHeaderProps, type SyntaxHighlighterProps } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { PrismLight } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import oneDark from "react-syntax-highlighter/dist/esm/styles/prism/one-dark";
import oneLight from "react-syntax-highlighter/dist/esm/styles/prism/one-light";

const languages: [string, { displayName: string }][] = [
  ["python", python], ["py", python],
  ["typescript", typescript], ["ts", typescript],
  ["tsx", tsx],
  ["javascript", javascript], ["js", javascript],
  ["jsx", jsx],
  ["bash", bash], ["sh", bash], ["shell", bash],
  ["json", json],
  ["markdown", markdown], ["md", markdown],
  ["css", css],
  ["html", markup], ["xml", markup],
  ["sql", sql],
  ["yaml", yaml], ["yml", yaml],
];
for (const [name, language] of languages) PrismLight.registerLanguage(name, language);
const KNOWN = new Set(languages.map(([name]) => name));

function useIsClient() {
  return useSyncExternalStore(() => () => {}, () => true, () => false);
}

function CodeHeader({ language, code }: CodeHeaderProps) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-header flex items-center justify-between px-3 py-1.5 text-[11px] text-muted-foreground">
      <span className="font-mono uppercase tracking-wide">{language || "text"}</span>
      <button
        type="button"
        className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label="Copy code"
        onClick={() => {
          void navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

function CodeBlock({ language, code, components }: SyntaxHighlighterProps) {
  const client = useIsClient();
  const { resolvedTheme } = useTheme();
  const lang = (language || "").toLowerCase();
  if (!client || !KNOWN.has(lang)) {
    return (
      <components.Pre className="overflow-x-auto px-4 py-3 font-mono text-[13px] leading-6">
        <components.Code>{code}</components.Code>
      </components.Pre>
    );
  }
  return (
    <PrismLight
      language={lang}
      style={resolvedTheme === "light" ? oneLight : oneDark}
      PreTag={components.Pre}
      CodeTag={components.Code}
      customStyle={{ margin: 0, padding: "0.9rem 1rem", background: "transparent", fontSize: "13px" }}
      codeTagProps={{ className: "font-mono" }}
    >
      {code}
    </PrismLight>
  );
}

function Link({ node: _node, ...props }: ComponentProps<"a"> & { node?: unknown }) {
  return <a {...props} target="_blank" rel="noreferrer" />;
}

export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      className="aui-md"
      smooth
      components={{ CodeHeader, SyntaxHighlighter: CodeBlock, a: Link }}
    />
  );
}
