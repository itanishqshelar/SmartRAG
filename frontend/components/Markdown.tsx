"use client";

import { memo, useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/lib/api";
import { useLibrary } from "@/lib/documents";
import { cn } from "@/lib/utils";

/** Turn [n] citations into links (#cite-n) outside of code, so the renderer can show them as chips. */
function linkCitations(text: string, indexes: Set<number>) {
  if (!indexes.size) return text;
  return text
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) =>
      i % 2 ? part : part.replace(/\[(\d{1,2})\](?!\()/g, (m, n) => (indexes.has(+n) ? `[${n}](#cite-${n})` : m)),
    )
    .join("");
}

function Citation({ source }: { source: Source }) {
  const preview = useLibrary((s) => s.preview);
  return (
    <button
      data-citation
      onClick={() => source.document_id && preview(source.document_id, source.excerpt, source.page)}
      title={`${source.filename}${source.page ? ` · page ${source.page}` : ""}`}
      className="relative -top-[0.35em] mx-[1px] inline-grid h-[1.35em] min-w-[1.35em] cursor-pointer place-items-center rounded-md bg-accent-soft px-1 align-baseline font-mono text-[0.68em] font-semibold text-accent transition-colors hover:bg-accent hover:text-accent-fg"
    >
      {source.index}
    </button>
  );
}

export const Markdown = memo(function Markdown({
  content,
  sources = [],
  streaming,
}: {
  content: string;
  sources?: Source[];
  streaming?: boolean;
}) {
  const byIndex = useMemo(() => new Map(sources.map((s) => [s.index, s])), [sources]);
  const text = useMemo(() => linkCitations(content, new Set(byIndex.keys())), [content, byIndex]);

  const components = useMemo<Components>(
    () => ({
      a: ({ href, children, ...props }) => {
        const source = href?.startsWith("#cite-") ? byIndex.get(Number(href.slice(6))) : undefined;
        if (source) return <Citation source={source} />;
        return (
          <a href={href} target="_blank" rel="noreferrer" {...props}>
            {children}
          </a>
        );
      },
    }),
    [byIndex],
  );

  return (
    <div className={cn("prose-chat", streaming && "streaming-caret")}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
});
