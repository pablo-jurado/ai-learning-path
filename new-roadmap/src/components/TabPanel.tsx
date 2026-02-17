import { useState } from 'react'
import type { DataNode } from '../data'
import { isLeaf } from '../utils'
import { Accordion } from './Accordion'

interface TabPanelProps {
  tab: DataNode
  colors: { color: string; light: string; border: string }
  isActive: boolean
  activeLeaf: string | null
  onLeafClick: (node: DataNode, breadcrumb: string) => void
}

export function TabPanel({ tab, colors, isActive, activeLeaf, onLeafClick }: TabPanelProps) {
  const [expandVer, setExpandVer] = useState(0)
  const [collapseVer, setCollapseVer] = useState(0)

  return (
    <div className={`tab-content${isActive ? ' active' : ''}`}>
      {tab.goal && <div className="module-goal">{tab.goal}</div>}
      <div className="toolbar">
        <button className="toolbar-btn" onClick={() => setExpandVer(v => v + 1)}>Expand all</button>
        <button className="toolbar-btn" onClick={() => setCollapseVer(v => v + 1)}>Collapse all</button>
      </div>
      <div className="accordion-group">
        {tab.children?.filter(s => !isLeaf(s)).map(section => (
          <Accordion
            key={section.name}
            node={section}
            depth={0}
            accentColor={colors.color}
            isTopLevel={true}
            breadcrumb={tab.name}
            expandVer={expandVer}
            collapseVer={collapseVer}
            activeLeaf={activeLeaf}
            onLeafClick={onLeafClick}
          />
        ))}
      </div>
    </div>
  )
}
