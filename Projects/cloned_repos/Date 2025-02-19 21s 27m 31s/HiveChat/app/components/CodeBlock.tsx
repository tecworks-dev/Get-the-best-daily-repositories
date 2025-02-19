import React, { useState } from "react";
import { CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { Button, Tooltip, message } from "antd";

const CodeBlock: React.FC<{ children: React.ReactNode; language: string }> = ({ children, language }) => {
  const [messageApi, contextHolderMessage] = message.useMessage();
  const reactNodeToText = (node: React.ReactNode): string => {
    if (typeof node === 'string' || typeof node === 'number' || typeof node === 'boolean') {
      return String(node);
    }

    if (Array.isArray(node)) {
      return node.map(reactNodeToText).join('');
    }

    if (typeof node === 'object' && node !== null && 'props' in node) {
      const { children } = node.props as { children?: React.ReactNode };
      return children ? reactNodeToText(children) : '';
    }

    return '';
  };

  return (
    <>
      {contextHolderMessage}
      <div className="bg-slate-200 -mt-4 -mx-4 mb-4 py-2 px-4 flex justify-between items-center">
        <span className="text-xs text-gray-500">{language}</span>
        <CopyToClipboard text={reactNodeToText(children)}
          onCopy={() => {
            messageApi.success('复制成功');
          }}>
          <Tooltip title="复制">
            <Button type="text" size='small'>
              <CopyOutlined style={{ color: 'gray' }} />
            </Button>
          </Tooltip>
        </CopyToClipboard>
      </div>
      <pre style={{margin: '0px', padding: '0px'}}>
        <code className={`language-${language}`}>{children}</code>
      </pre>
    </>
  );
};

export default CodeBlock;
