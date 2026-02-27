export default function InlineHelpTip({ text }: { text: string }) {
  return (
    <span className="helpTipWrap" tabIndex={0} role="button" aria-label="Info">
      <span className="helpTipIcon">?</span>
      <span className="helpTipBubble">{text}</span>
    </span>
  )
}
