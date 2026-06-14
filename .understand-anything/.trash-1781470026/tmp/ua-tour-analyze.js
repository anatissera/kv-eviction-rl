#!/usr/bin/env node
const fs = require('fs');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-tour-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let data;
try {
  data = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to read/parse input:', e.message);
  process.exit(1);
}

const { nodes, edges, layers } = data;

// Build node map
const nodeMap = {};
for (const n of nodes) {
  nodeMap[n.id] = n;
}

// A. Fan-In Ranking
const fanInMap = {};
const fanOutMap = {};
for (const n of nodes) {
  fanInMap[n.id] = 0;
  fanOutMap[n.id] = 0;
}
for (const e of edges) {
  if (fanInMap[e.target] !== undefined) fanInMap[e.target]++;
  if (fanOutMap[e.source] !== undefined) fanOutMap[e.source]++;
}

const fanInRanking = Object.entries(fanInMap)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, fanIn]) => ({ id, fanIn, name: nodeMap[id]?.name || id }));

const fanOutRanking = Object.entries(fanOutMap)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, fanOut]) => ({ id, fanOut, name: nodeMap[id]?.name || id }));

// B. Entry Point Candidates
const totalNodes = nodes.length;
const fanInValues = Object.values(fanInMap).sort((a, b) => a - b);
const fanOutValues = Object.values(fanOutMap).sort((a, b) => a - b);
const top10PctFanOutThreshold = fanOutValues[Math.floor(totalNodes * 0.9)];
const bottom25PctFanInThreshold = fanInValues[Math.floor(totalNodes * 0.25)];

const entryNamePatterns = [
  'index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js',
  'mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py',
  'run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru',
  'index.php','App.swift','Application.kt','main.cpp','main.c'
];

const candidateScores = {};
for (const n of nodes) {
  let score = 0;
  if (n.type === 'document' && n.name === 'README.md') {
    const depth = (n.filePath.match(/\//g) || []).length;
    if (depth === 0) score += 5;
    else score += 2;
  } else if (n.type === 'document' && n.name.endsWith('.md')) {
    const depth = (n.filePath.match(/\//g) || []).length;
    if (depth === 0) score += 2;
  } else if (n.type === 'file') {
    if (entryNamePatterns.includes(n.name)) score += 3;
    const depth = (n.filePath.match(/\//g) || []).length;
    if (depth <= 1) score += 1;
    if (fanOutMap[n.id] >= top10PctFanOutThreshold) score += 1;
    if (fanInMap[n.id] <= bottom25PctFanInThreshold) score += 1;
  }
  candidateScores[n.id] = score;
}

const entryPointCandidates = Object.entries(candidateScores)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 5)
  .map(([id, score]) => ({
    id,
    score,
    name: nodeMap[id]?.name || id,
    summary: nodeMap[id]?.summary || ''
  }));

// C. BFS from top code entry point
const codeEntryPoint = entryPointCandidates.find(c => nodeMap[c.id]?.type === 'file');
const bfsStart = codeEntryPoint ? codeEntryPoint.id : null;

const bfsResult = { startNode: bfsStart, order: [], depthMap: {}, byDepth: {} };
if (bfsStart) {
  // Build adjacency: imports and calls edges forward
  const adj = {};
  for (const n of nodes) adj[n.id] = [];
  for (const e of edges) {
    if ((e.type === 'imports' || e.type === 'calls') && adj[e.source]) {
      adj[e.source].push(e.target);
    }
  }

  const visited = new Set();
  const queue = [{ id: bfsStart, depth: 0 }];
  visited.add(bfsStart);

  while (queue.length > 0) {
    const { id, depth } = queue.shift();
    bfsResult.order.push(id);
    bfsResult.depthMap[id] = depth;
    if (!bfsResult.byDepth[depth]) bfsResult.byDepth[depth] = [];
    bfsResult.byDepth[depth].push(id);

    for (const neighbor of (adj[id] || [])) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ id: neighbor, depth: depth + 1 });
      }
    }
  }
}

// D. Non-code files
const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
for (const n of nodes) {
  if (n.type === 'document') {
    nonCodeFiles.documentation.push({ id: n.id, name: n.name, summary: n.summary });
  } else if (['service', 'pipeline', 'resource'].includes(n.type)) {
    nonCodeFiles.infrastructure.push({ id: n.id, name: n.name, summary: n.summary });
  } else if (['table', 'schema', 'endpoint'].includes(n.type)) {
    nonCodeFiles.data.push({ id: n.id, name: n.name, summary: n.summary });
  } else if (n.type === 'config') {
    nonCodeFiles.config.push({ id: n.id, name: n.name, summary: n.summary });
  }
}

// E. Clusters: bidirectional edges
const edgeSet = new Set();
const bidir = new Map();
for (const e of edges) {
  const key = `${e.source}|||${e.target}`;
  edgeSet.add(key);
}

const pairs = [];
for (const e of edges) {
  const reverse = `${e.target}|||${e.source}`;
  if (edgeSet.has(reverse) && e.source < e.target) {
    pairs.push([e.source, e.target]);
  }
}

// Build initial clusters from pairs, then expand
const clusterMap = new Map();
for (const [a, b] of pairs) {
  let found = null;
  for (const [key, members] of clusterMap.entries()) {
    if (members.has(a) || members.has(b)) {
      found = key;
      break;
    }
  }
  if (found) {
    clusterMap.get(found).add(a);
    clusterMap.get(found).add(b);
  } else {
    const s = new Set([a, b]);
    clusterMap.set(`${a}+${b}`, s);
  }
}

// Also find clusters via shared connections (nodes connected to 2+ cluster members)
// Expand each cluster
for (const [key, members] of clusterMap.entries()) {
  const memberArr = Array.from(members);
  // Count how many edges from a non-member node point to members
  const outsiderConnections = {};
  for (const e of edges) {
    if (memberArr.includes(e.target) && !members.has(e.source)) {
      outsiderConnections[e.source] = (outsiderConnections[e.source] || 0) + 1;
    }
    if (memberArr.includes(e.source) && !members.has(e.target)) {
      outsiderConnections[e.target] = (outsiderConnections[e.target] || 0) + 1;
    }
  }
  for (const [node, count] of Object.entries(outsiderConnections)) {
    if (count >= 2 && members.size < 5) {
      members.add(node);
    }
  }
}

// Count edges within each cluster
const clusters = [];
for (const [, members] of clusterMap.entries()) {
  const memberArr = Array.from(members);
  let edgeCount = 0;
  for (const e of edges) {
    if (members.has(e.source) && members.has(e.target)) edgeCount++;
  }
  clusters.push({ nodes: memberArr, edgeCount });
}
clusters.sort((a, b) => b.edgeCount - a.edgeCount);
const topClusters = clusters.slice(0, 10);

// F. Node Summary Index
const nodeSummaryIndex = {};
for (const n of nodes) {
  nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary };
}

const result = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal: bfsResult,
  nonCodeFiles,
  clusters: topClusters,
  layers: {
    count: layers.length,
    list: layers.map(l => ({ id: l.id, name: l.name, description: l.description }))
  },
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  console.log('Analysis complete. Output written to', outputPath);
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}
