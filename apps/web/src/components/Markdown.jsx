import React from "react";

// Compact, dependency-free Markdown renderer for first-party prompt-guide
// content shipped from /public/prompt-guides. It covers the subset those files
// use — ATX headings, paragraphs, ordered/unordered lists, blockquotes, fenced
// code blocks, and inline bold/italic/code/links. It builds React elements
// directly (never dangerouslySetInnerHTML), so untrusted markup cannot inject
// HTML, and link hrefs are restricted to safe schemes.

const HEADING = /^(#{1,6})\s+(.*)$/;
const LIST_ITEM = /^\s*([-*+]|\d+\.)\s+/;
const ORDERED_ITEM = /^\s*\d+\.\s+/;
const FENCE = /^```/;
const QUOTE = /^\s*>\s?/;

function safeHref(url) {
  const trimmed = (url || "").trim();
  if (/^(https?:|mailto:)/i.test(trimmed)) return trimmed;
  if (/^[/#]/.test(trimmed)) return trimmed;
  return undefined;
}

const INLINE_RULES = [
  { re: /`([^`]+)`/, node: (m, key) => <code key={key}>{m[1]}</code> },
  {
    re: /\[([^\]]+)\]\(([^)\s]+)\)/,
    node: (m, key) => {
      const href = safeHref(m[2]);
      if (!href) return <span key={key}>{renderInline(m[1], key)}</span>;
      return (
        <a key={key} href={href} target="_blank" rel="noopener noreferrer">
          {renderInline(m[1], key)}
        </a>
      );
    },
  },
  { re: /\*\*([^*]+)\*\*/, node: (m, key) => <strong key={key}>{renderInline(m[1], key)}</strong> },
  { re: /\*([^*]+)\*/, node: (m, key) => <em key={key}>{renderInline(m[1], key)}</em> },
  { re: /_([^_]+)_/, node: (m, key) => <em key={key}>{renderInline(m[1], key)}</em> },
];

function renderInline(text, keyPrefix) {
  let earliest = null;
  for (const rule of INLINE_RULES) {
    const match = rule.re.exec(text);
    if (match && (!earliest || match.index < earliest.match.index)) {
      earliest = { rule, match };
    }
  }
  if (!earliest) return text;
  const { rule, match } = earliest;
  const before = text.slice(0, match.index);
  const after = text.slice(match.index + match[0].length);
  const nodes = [];
  if (before) nodes.push(before);
  nodes.push(rule.node(match, `${keyPrefix}-${match.index}`));
  const rest = renderInline(after, `${keyPrefix}-r`);
  if (Array.isArray(rest)) nodes.push(...rest);
  else if (rest) nodes.push(rest);
  return nodes;
}

function parseBlocks(content) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
      i += 1;
      continue;
    }
    if (FENCE.test(line.trim())) {
      i += 1;
      const code = [];
      while (i < lines.length && !FENCE.test(lines[i].trim())) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1; // skip closing fence
      blocks.push({ type: "code", text: code.join("\n") });
      continue;
    }
    if (LIST_ITEM.test(line)) {
      const ordered = ORDERED_ITEM.test(line);
      const items = [];
      while (i < lines.length && LIST_ITEM.test(lines[i])) {
        items.push(lines[i].replace(LIST_ITEM, ""));
        i += 1;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    if (QUOTE.test(line)) {
      const quote = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quote.push(lines[i].replace(QUOTE, ""));
        i += 1;
      }
      blocks.push({ type: "quote", text: quote.join(" ") });
      continue;
    }
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !HEADING.test(lines[i]) &&
      !LIST_ITEM.test(lines[i]) &&
      !FENCE.test(lines[i].trim()) &&
      !QUOTE.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ type: "paragraph", text: para.join(" ") });
  }
  return blocks;
}

function renderBlock(block, key) {
  switch (block.type) {
    case "heading": {
      const Tag = `h${Math.min(block.level, 6)}`;
      return <Tag key={key}>{renderInline(block.text, key)}</Tag>;
    }
    case "list": {
      const Tag = block.ordered ? "ol" : "ul";
      return (
        <Tag key={key}>
          {block.items.map((item, index) => (
            <li key={`${key}-${index}`}>{renderInline(item, `${key}-${index}`)}</li>
          ))}
        </Tag>
      );
    }
    case "quote":
      return (
        <blockquote key={key}>
          <p>{renderInline(block.text, key)}</p>
        </blockquote>
      );
    case "code":
      return (
        <pre key={key}>
          <code>{block.text}</code>
        </pre>
      );
    case "paragraph":
    default:
      return <p key={key}>{renderInline(block.text, key)}</p>;
  }
}

export function Markdown({ content }) {
  const blocks = parseBlocks(content || "");
  return <div className="markdown-body">{blocks.map((block, index) => renderBlock(block, `b${index}`))}</div>;
}
