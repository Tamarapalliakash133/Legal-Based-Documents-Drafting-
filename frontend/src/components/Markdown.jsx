import ReactMarkdown from 'react-markdown'

// "[p.12]" -> inline code, which the stylesheet turns into a brass page-tab
const tabify = (t) => t.replace(/\[(p\.[^\]]+)\]/g, '`$1`')

export default function Markdown({ children, cite = true }) {
  return <ReactMarkdown>{cite ? tabify(children || '') : children || ''}</ReactMarkdown>
}
