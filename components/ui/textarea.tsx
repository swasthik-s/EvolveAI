"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "flex min-h-[52px] w-full resize-none rounded-xl border border-white/10 bg-transparent px-4 py-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-500 focus-visible:border-violet-300/60 focus-visible:ring-2 focus-visible:ring-violet-300/10 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";

export { Textarea };
