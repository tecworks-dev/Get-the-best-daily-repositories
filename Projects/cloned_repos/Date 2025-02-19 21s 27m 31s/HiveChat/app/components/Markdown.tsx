import React, { useMemo } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import RemarkMath from "remark-math";
import RemarkBreaks from "remark-breaks";
import RehypeKatex from "rehype-katex";
import rehypeHighlight from 'rehype-highlight';
import CodeBlock from '@/app/components/CodeBlock';
import 'highlight.js/styles/github.css';
import "katex/dist/katex.min.css";
import 'github-markdown-css/github-markdown-light.css';
const MarkdownRender = (props: {
  content: string,
}
) => {
  function escapeBrackets(text: string) {
    const pattern =
      /(```[\s\S]*?```|`.*?`)|\\\[([\s\S]*?[^\\])\\\]|\\\((.*?)\\\)/g;
    return text.replace(
      pattern,
      (match, codeBlock, squareBracket, roundBracket) => {
        if (codeBlock) {
          return codeBlock;
        } else if (squareBracket) {
          return `$$${squareBracket}$$`;
        } else if (roundBracket) {
          return `$${roundBracket}$`;
        }
        return match;
      },
    );
  }

  const escapedContent = useMemo(() => {
    return escapeBrackets(props.content);
  }, [props.content]);

  return (
    <Markdown
      remarkPlugins={[RemarkMath, remarkGfm, RemarkBreaks]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <CodeBlock language={match[1]}>{children}</CodeBlock>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        a: (aProps) => {
          const href = aProps.href || "";
          if (/\.(aac|mp3|opus|wav)$/.test(href)) {
            return (
              <figure>
                <audio controls src={href}></audio>
              </figure>
            );
          }
          if (/\.(3gp|3g2|webm|ogv|mpeg|mp4|avi)$/.test(href)) {
            return (
              <video controls width="99.9%">
                <source src={href} />
              </video>
            );
          }
          const isInternal = /^\/#/i.test(href);
          const target = isInternal ? "_self" : aProps.target ?? "_blank";
          return <a {...aProps} target={target} />;
        },
      }}
      rehypePlugins={[
        RehypeKatex,
        [
          rehypeHighlight,
          {
            detect: false,
            ignoreMissing: true,
          },
        ],
      ]}
    >{escapedContent}</Markdown>
  )
}

export default MarkdownRender
