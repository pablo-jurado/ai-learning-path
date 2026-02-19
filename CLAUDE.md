# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI Learning Path web app — a single-page React application that renders interactive, hierarchical learning roadmaps for AI topics (Developer, Business, Certifications). All application code lives in the `new-roadmap/` subdirectory.

## Commands

All commands must be run from the `new-roadmap/` directory:

```bash
npm run dev        # Vite dev server with HMR
npm run build      # TypeScript type-check (tsc -b) + Vite production build → dist/
npm run lint       # ESLint on all TS/TSX files
npm run preview    # Serve production build locally
```

There is no test framework configured.

## Tech Stack

- React 19 + TypeScript 5.9 (strict mode) + Vite 7
- React Router DOM 7 (BrowserRouter, client-side routing)
- Plain CSS with custom properties (single `src/index.css`, no CSS framework)
- ESLint 9 flat config with typescript-eslint, react-hooks, react-refresh plugins

## Architecture

### Data-driven UI

All learning content is defined as static `DataNode` trees in `src/data/data.ts` — three exports: `DEV_PATH`, `BUSINESS_PATH`, `CERTIFICATIONS_PATH`. The `DataNode` interface (in `src/types/index.ts`) is recursive with optional `children`, `sources`, `tags`, `icon`, and `color` fields.

**To add or modify content, edit `src/data/data.ts` — not components.**

### Component patterns

- **`PathWrapper`** — shared page shell used by all learning-path pages. Receives a `DataNode` root and renders Header, Tabs, TabPanels with recursive Accordions, and the Sidebar detail panel.
- **`Accordion`** — recursive component that renders branch nodes as collapsible sections and leaf nodes as `LeafCard` components.
- **`Sidebar`** — slide-in detail panel (right side, 440px) showing node details and resource links. Opens on leaf card click, closes on ESC or backdrop click.
- **`SideNav`** — hamburger-triggered navigation overlay wrapping all pages in `App.tsx`.

### Routes

| Route | Page |
|---|---|
| `/` | Home (landing with track cards) |
| `/developer` | DeveloperPath |
| `/business` | BusinessPath |
| `/certifications` | Certifications |
| `/project-guidelines` | AgentProjectGuidelines |

### State management

Local React state only (`useState`/`useCallback`/`useEffect`). No state library. `AgentProjectGuidelines` uses `localStorage` to persist checklist completion state.

### Styling

Single global CSS file (`src/index.css`, ~1095 lines) using CSS custom properties on `:root`. Per-instance color theming is done by injecting `--accent-color` via inline `style` props cast as `React.CSSProperties`.

## TypeScript

Strict mode enabled with `noUnusedLocals` and `noUnusedParameters`. Target ES2022, JSX react-jsx, bundler module resolution.
