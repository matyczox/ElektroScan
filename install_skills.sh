#!/usr/bin/env bash
# Instalacja skilsów dla ElektroScan
# Użycie: bash install_skills.sh
# Wymaga: npx (Node.js)

set -e

# ─── detector-engineer ───────────────────────────────────────────────────────
npx skills add mindrally/skills@computer-vision-opencv -y
npx skills add wshobson/agents@python-performance-optimization -y
npx skills add davila7/claude-code-templates@senior-computer-vision -y

# ─── hitl-frontend-dev ───────────────────────────────────────────────────────
npx skills add vercel-labs/agent-skills@vercel-react-best-practices -y
npx skills add wshobson/agents@typescript-advanced-types -y
npx skills add dotneet/claude-code-marketplace@typescript-react-reviewer -y

# ─── qa-engineer ─────────────────────────────────────────────────────────────
npx skills add anthropics/skills@webapp-testing -y
npx skills add wshobson/agents@python-testing-patterns -y
npx skills add wshobson/agents@e2e-testing-patterns -y

# ─── devops-engineer ─────────────────────────────────────────────────────────
npx skills add xixu-me/skills@github-actions-docs -y
npx skills add sickn33/antigravity-awesome-skills@docker-expert -y
npx skills add github/awesome-copilot@multi-stage-dockerfile -y

# ─── debug-analyst ───────────────────────────────────────────────────────────
npx skills add obra/superpowers@systematic-debugging -y
npx skills add wshobson/agents@debugging-strategies -y
npx skills add wshobson/agents@parallel-debugging -y

echo "Wszystkie skille zainstalowane."
