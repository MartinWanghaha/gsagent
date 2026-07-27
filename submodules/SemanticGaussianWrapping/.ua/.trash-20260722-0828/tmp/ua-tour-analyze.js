#!/usr/bin/env node
'use strict';

const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) fail('usage: ua-tour-analyze.js INPUT OUTPUT');

  const graph = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const layers = Array.isArray(graph.layers) ? graph.layers : [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));

  for (const edge of edges) {
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
  }

  const rank = (counts, field) => nodes
    .map((node) => ({id: node.id, [field]: counts.get(node.id), name: node.name}))
    .sort((a, b) => b[field] - a[field] || a.id.localeCompare(b.id))
    .slice(0, 20);
  const fanInRanking = rank(fanIn, 'fanIn');
  const fanOutRanking = rank(fanOut, 'fanOut');

  const codeNodes = nodes.filter((node) => ['file', 'class', 'function'].includes(node.type));
  const topOutThreshold = [...codeNodes].sort((a, b) => fanOut.get(b.id) - fanOut.get(a.id))[
    Math.max(0, Math.ceil(codeNodes.length * 0.10) - 1)
  ];
  const topOutMinimum = topOutThreshold ? fanOut.get(topOutThreshold.id) : Infinity;
  const lowInThreshold = [...codeNodes].sort((a, b) => fanIn.get(a.id) - fanIn.get(b.id))[
    Math.max(0, Math.ceil(codeNodes.length * 0.25) - 1)
  ];
  const lowInMaximum = lowInThreshold ? fanIn.get(lowInThreshold.id) : -Infinity;
  const entryNames = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js', 'server.ts',
    'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs', 'manage.py', 'app.py',
    'wsgi.py', 'asgi.py', 'run.py', 'train.py', '__main__.py', 'Application.java', 'Main.java',
    'Program.cs', 'config.ru', 'index.php', 'App.swift', 'Application.kt', 'main.cpp', 'main.c'
  ]);

  const entryPointCandidates = nodes.map((node) => {
    let score = 0;
    const path = node.filePath || '';
    const depth = path.split('/').filter(Boolean).length;
    if (codeNodes.includes(node)) {
      if (entryNames.has(node.name)) score += 3;
      if (depth <= 2) score += 1;
      if (fanOut.get(node.id) >= topOutMinimum) score += 1;
      if (fanIn.get(node.id) <= lowInMaximum) score += 1;
    } else if (node.type === 'document') {
      if (path === 'README.md') score += 5;
      else if (depth === 1 && path.endsWith('.md')) score += 2;
    }
    return {id: node.id, score, name: node.name, summary: node.summary || ''};
  }).filter((candidate) => candidate.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, 5);

  const codeEntry = entryPointCandidates.find((candidate) => {
    const node = nodeById.get(candidate.id);
    return node && node.type !== 'document';
  });
  const bfsTraversal = {startNode: codeEntry ? codeEntry.id : null, order: [], depthMap: {}, byDepth: {}};
  if (codeEntry) {
    const adjacency = new Map(nodes.map((node) => [node.id, []]));
    for (const edge of edges) {
      if (['imports', 'calls'].includes(edge.type) && adjacency.has(edge.source) && nodeById.has(edge.target)) {
        adjacency.get(edge.source).push(edge.target);
      }
    }
    for (const targets of adjacency.values()) targets.sort();
    const queue = [codeEntry.id];
    bfsTraversal.depthMap[codeEntry.id] = 0;
    while (queue.length) {
      const current = queue.shift();
      bfsTraversal.order.push(current);
      const depth = bfsTraversal.depthMap[current];
      if (!bfsTraversal.byDepth[depth]) bfsTraversal.byDepth[depth] = [];
      bfsTraversal.byDepth[depth].push(current);
      for (const target of adjacency.get(current) || []) {
        if (bfsTraversal.depthMap[target] !== undefined) continue;
        bfsTraversal.depthMap[target] = depth + 1;
        queue.push(target);
      }
    }
  }

  const inventory = (types) => nodes.filter((node) => types.includes(node.type)).map((node) => ({
    id: node.id, name: node.name, type: node.type, summary: node.summary || ''
  }));
  const nonCodeFiles = {
    documentation: inventory(['document']),
    infrastructure: inventory(['service', 'pipeline', 'resource']),
    data: inventory(['table', 'schema', 'endpoint']),
    config: inventory(['config'])
  };

  const relationKeys = new Set(edges.filter((edge) => ['imports', 'calls'].includes(edge.type))
    .map((edge) => `${edge.source}\u0000${edge.target}\u0000${edge.type}`));
  const pairs = [];
  for (const edge of edges) {
    if (!['imports', 'calls'].includes(edge.type)) continue;
    if (edge.source >= edge.target) continue;
    if (relationKeys.has(`${edge.target}\u0000${edge.source}\u0000${edge.type}`)) {
      pairs.push([edge.source, edge.target]);
    }
  }
  const candidateClusters = pairs.map(([a, b]) => new Set([a, b]));
  for (const cluster of candidateClusters) {
    let changed = true;
    while (changed && cluster.size < 5) {
      changed = false;
      for (const node of nodes) {
        if (cluster.has(node.id)) continue;
        let links = 0;
        for (const member of cluster) {
          if (edges.some((edge) => (edge.source === node.id && edge.target === member) ||
              (edge.source === member && edge.target === node.id))) links += 1;
        }
        if (links >= 2) {
          cluster.add(node.id);
          changed = true;
          if (cluster.size >= 5) break;
        }
      }
    }
  }
  const seenClusters = new Set();
  const clusters = candidateClusters.map((cluster) => [...cluster].sort()).filter((members) => {
    const key = members.join('\u0000');
    if (seenClusters.has(key)) return false;
    seenClusters.add(key);
    return true;
  }).map((members) => ({
    nodes: members,
    edgeCount: edges.filter((edge) => members.includes(edge.source) && members.includes(edge.target)).length
  })).sort((a, b) => b.edgeCount - a.edgeCount || b.nodes.length - a.nodes.length).slice(0, 10);

  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [node.id, {
    name: node.name, type: node.type, summary: node.summary || ''
  }]));
  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal,
    nonCodeFiles,
    clusters,
    layers: {count: layers.length, list: layers.map(({id, name, description}) => ({id, name, description}))},
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  fail(error && error.stack ? error.stack : String(error));
}
