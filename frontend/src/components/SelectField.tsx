import type { SelectHTMLAttributes } from 'react'
import { ChevronDownIcon } from '@heroicons/react/24/solid'

type SelectFieldProps = SelectHTMLAttributes<HTMLSelectElement> & {
  wrapClassName?: string
}

export default function SelectField({ wrapClassName, className, children, ...props }: SelectFieldProps) {
  return (
    <div className={`selectField ${wrapClassName || ''}`.trim()}>
      <select className={className} {...props}>
        {children}
      </select>
      <ChevronDownIcon className="selectIcon" aria-hidden="true" />
    </div>
  )
}
