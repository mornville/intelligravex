import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type MarkdownTextProps = {
  content: string
  className?: string
}

const REMARK_PLUGINS = [remarkGfm]

const RENDER_COMPONENTS = {
  a: ({ node, ...props }: any) => (
    <a {...props} target="_blank" rel="noreferrer" />
  ),
  code: (props: any) => {
    const { inline, className, children, ...rest } = props
    if (inline) {
      return (
        <code className={`md-inline-code ${className || ''}`.trim()} {...rest}>
          {children}
        </code>
      )
    }
    return (
      <code className={`md-code ${className || ''}`.trim()} {...rest}>
        {children}
      </code>
    )
  },
  pre: ({ node, className, children, ...props }: any) => (
    <pre className={`md-pre ${className || ''}`.trim()} {...props}>
      {children}
    </pre>
  ),
}

function MarkdownText({ content, className }: MarkdownTextProps) {
  if (!content) return null
  return (
    <ReactMarkdown
      className={className || 'markdown'}
      remarkPlugins={REMARK_PLUGINS}
      components={RENDER_COMPONENTS}
    >
      {content}
    </ReactMarkdown>
  )
}

export default memo(MarkdownText, (prev, next) => prev.content === next.content && prev.className === next.className)
