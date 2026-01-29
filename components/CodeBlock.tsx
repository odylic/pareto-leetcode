import { codeToHtml } from "shiki";
import { CopyButton } from "./CopyButton";

interface CodeBlockProps {
  code: string;
  language?: string;
}

export async function CodeBlock({ code, language = "python" }: CodeBlockProps) {
  const html = await codeToHtml(code, {
    lang: language,
    theme: "github-dark",
  });

  return (
    <div className="relative rounded-lg overflow-hidden bg-[#0d1117] border border-card-border">
      <div className="flex items-center justify-between px-4 py-2.5 bg-card-bg border-b border-card-border">
        <span className="text-sm text-foreground/60 font-mono uppercase">
          {language}
        </span>
        <CopyButton code={code} />
      </div>
      <div
        className="p-4 overflow-x-auto text-base [&>pre]:!bg-transparent [&>pre]:!p-0 [&>pre]:!m-0"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
