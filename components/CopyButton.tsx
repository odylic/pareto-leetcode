"use client";

import { useState } from "react";

interface CopyButtonProps {
  code: string;
}

export function CopyButton({ code }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      className="text-sm text-foreground/60 hover:text-foreground transition-colors px-2 py-1 rounded hover:bg-card-border/50"
      onClick={handleCopy}
      title="Copy code"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}
